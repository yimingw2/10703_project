import numpy as np
import argparse
import re
from utils.feature_extraction import *
from parser_model import *
np.set_printoptions(threshold=np.nan) 


class ParserEnv():

	def __init__(self, dataset, arc_only):

		self.dataset = dataset
		self.num_classes = self.dataset.model_config.num_classes
		self.arc_only = arc_only

	
	def step(self, sentence, action, num):

		# update state inside sentence
		if self.arc_only:
			real_action = action
			arc_label = -1
		else:
			if action == 0:
				real_action = 0
				arc_label = -1
			else:
				action_temp = action - 1
				real_action = action_temp % 2 + 1 
				arc_label = action_temp / 2

		# judge if real_action is legal
		# if real_action == 0 and len(sentence.buff) == 0: # shift
		# 	legal_label = False
		# elif real_action == 1 and len(sentence.stack) <= 2: # left-arc
		# 	legal_label = False
		# elif real_action == 2 and len(sentence.stack) < 2: # right-arc
		# 	legal_label = False
		# else:
		# 	legal_label = True
		
		# if legal_label:
		if real_action != 0:
			sentence.update_child_dependencies(real_action, arc_label)
		sentence.update_state_by_transition(real_action, arc_label, gold=False)
		# extract new state from sentence
		new_state = self.dataset.feature_extractor.extract_for_current_state(sentence, \
																			 self.dataset.word2idx, \
																			 self.dataset.pos2idx, \
																			 self.dataset.dep2idx)

		# compute reward
		# if num >= 3 * len(sentence.tokens):
		# 	reward = -50
		# elif not legal_label:
		# 	reward = -10
		# elif real_action == 0:
		# 	reward = 0
		# else:
		# 	h, t, l = sentence.predicted_dependencies[-1] # [h, t]
		# 	if t.head_id != h.token_id:
		# 		reward = -1
		# 	else:
		# 		reward = 4
		# 		if not self.arc_only and l == self.dataset.dep2idx[t.dep]:
		# 			reward += 1
		if real_action != 0:
			h, t, l = sentence.predicted_dependencies[-1] # [h, t]
			if t.head_id != h.token_id:
				sentence.incorrect_tokens_UAS += 1
		reward = -sentence.incorrect_tokens_UAS

		# done
		if (len(sentence.stack) == 1 and len(sentence.buff) == 0): # or (num >= 3 * len(sentence.tokens)):
			done = True
		else:
			done = False
		return new_state, reward, done
	

	def reset(self, sentence):

		sentence.reset_to_initial_state()
		sentence.clear_prediction_dependencies()
		sentence.clear_children_info()
		init_state = self.dataset.feature_extractor.extract_for_current_state(sentence, \
																			  self.dataset.word2idx, \
																			  self.dataset.pos2idx, \
																			  self.dataset.dep2idx)
		return init_state



class Critic():

	def __init__(self, config, word_embeddings, pos_embeddings, dep_embeddings):
		self.config = config
		self.word_embeddings = word_embeddings
		self.pos_embeddings = pos_embeddings
		self.dep_embeddings = dep_embeddings
		self.output = self._build()


	def _build(self):
		self._add_placeholders()
		self.embeddings = self._add_embedding()

		dense = tf.layers.dense(self.embeddings, 128, activation=tf.nn.relu, name='critic_dense_1')
		dense = tf.layers.dense(dense, 128, activation=tf.nn.relu, name='critic_dense_2')
		dense = tf.layers.dense(dense, 64, activation=tf.nn.relu, name='critic_dense_3')
		dense = tf.layers.dense(dense, 32, activation=tf.nn.relu, name='critic_dense_4')
		output = tf.layers.dense(dense, 1, name='critic_dense_5')
		return output


	def create_feed_dict(self, inputs_batch):
		feed_dict = {self.word_input_placeholder: inputs_batch[0],
					 self.pos_input_placeholder: inputs_batch[1],
					 self.dep_input_placeholder: inputs_batch[2]}
		return feed_dict


	def _add_placeholders(self):

		with tf.variable_scope("critic_input_placeholders"):
			self.word_input_placeholder = tf.placeholder(shape=[None, self.config.word_features_types],
														 dtype=tf.int32, name="critic_batch_word_indices")
			self.pos_input_placeholder = tf.placeholder(shape=[None, self.config.pos_features_types],
														dtype=tf.int32, name="critic_batch_pos_indices")
			self.dep_input_placeholder = tf.placeholder(shape=[None, self.config.dep_features_types],
														dtype=tf.int32, name="critic_batch_dep_indices")


	def _add_embedding(self):
		with tf.variable_scope("critic_feature_lookup"):
			self.word_embedding_matrix = random_uniform_initializer(self.word_embeddings.shape, "critic_word_embedding_matrix",
																	0.01, trainable=True)
			self.pos_embedding_matrix = random_uniform_initializer(self.pos_embeddings.shape, "critic_pos_embedding_matrix",
																   0.01, trainable=True)
			self.dep_embedding_matrix = random_uniform_initializer(self.dep_embeddings.shape, "critic_dep_embedding_matrix",
																   0.01, trainable=True)

			word_context_embeddings = tf.nn.embedding_lookup(self.word_embedding_matrix, self.word_input_placeholder)
			pos_context_embeddings = tf.nn.embedding_lookup(self.pos_embedding_matrix, self.pos_input_placeholder)
			dep_context_embeddings = tf.nn.embedding_lookup(self.dep_embedding_matrix, self.dep_input_placeholder)

			word_embeddings = tf.reshape(word_context_embeddings,
										 [-1, self.config.word_features_types * self.config.embedding_dim],
										 name="critic_word_context_embeddings")
			pos_embeddings = tf.reshape(pos_context_embeddings,
										[-1, self.config.pos_features_types * self.config.embedding_dim],
										name="critic_pos_context_embeddings")
			dep_embeddings = tf.reshape(dep_context_embeddings,
										[-1, self.config.dep_features_types * self.config.embedding_dim],
										name="critic_dep_context_embeddings")

		with tf.variable_scope("critic_batch_inputs"):
			embeddings = tf.concat([word_embeddings, pos_embeddings, dep_embeddings], 1, name="critic_batch_feature_matrix")

		return embeddings # word_embeddings, pos_embeddings, dep_embeddings



class A2C():

	def __init__(self, dataset, env, action_num, actor_model, lr, critic_model, critic_lr, num_epoch, epsilon, e_sm=1e-8, n=20, name=None):     
		# Initializes A2C.
		# :param model: The actor model.
		# :param lr: Learning rate for the actor model.
		# :param critic_model: The critic model.
		# :param critic_lr: Learning rate for the critic model.
		# :param num_episodes: number of episode
		# :param env:
		# :param n: The value of N in N-step A2C.   
		self.dataset = dataset
		self.env = env
		self.action_num = action_num

		self.actor_model = actor_model
		self.critic_model = critic_model
		self.actor_lr = lr
		self.critic_lr = critic_lr

		self.num_epoch = num_epoch
		self.n = n
		self.name = name
		self.epsilon = epsilon
		self.e_sm = e_sm

		self._add_placeholder()
		self.train_op_actor, self.train_op_critic = self._build_train_op()


	def _add_placeholder(self):
		self.actions = tf.placeholder(tf.int32, shape=[None, self.action_num], name='actions') # one-hot
		self.Rt = tf.placeholder(tf.float32, shape=[None, 1], name='Rt')
		self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='advantage')
		self.actor_learning_rate = tf.placeholder(tf.float32, shape=[], name='actor_lr')
		self.critic_learning_rate = tf.placeholder(tf.float32, shape=[], name='critic_lr')


	def _build_train_op(self):
		# loss for actor

		self.actor_pred = tf.nn.softmax(logits=self.actor_model.pred, axis=1)
		self.neg_log_prob = -tf.log(tf.expand_dims(tf.reduce_sum(self.actor_pred * tf.cast(self.actions, dtype=tf.float32), axis=1) + self.e_sm, axis=1))
		self.loss_a = tf.reduce_mean(self.neg_log_prob * self.advantage)
		# train_op_actor = tf.train.AdamOptimizer(self.actor_lr).minimize(self.loss_a)
		optimizer_actor = tf.train.AdamOptimizer(self.actor_lr)
		gvs_actor = optimizer_actor.compute_gradients(self.loss_a)
		capped_gvs_actor = [(tf.clip_by_norm(grad, 1), var) for grad, var in gvs_actor]
		train_op_actor = optimizer_actor.apply_gradients(capped_gvs_actor)
		# loss for critic
		self.loss_c = tf.reduce_mean(tf.squared_difference(self.Rt, self.critic_model.output))
		# train_op_critic = tf.train.AdamOptimizer(self.critic_lr).minimize(self.loss_c)
		optimizer_critic = tf.train.AdamOptimizer(self.critic_lr)
		gvs_critic = optimizer_critic.compute_gradients(self.loss_c)
		capped_gvs_critic = [(tf.clip_by_norm(grad, 1), var) for grad, var in gvs_critic]
		train_op_critic = optimizer_critic.apply_gradients(capped_gvs_critic)
		return train_op_actor, train_op_critic


	def _epsilon_greedy_policy(self, q_values):
		"""
		Creating epsilon greedy probabilities to sample from.
		:param q_values: the q value function
		:param epsilon: exploration probability
		:return: epsilon greedy probabilities
		"""
		e_prob = np.ones(self.action_num, dtype=float) * self.epsilon / float(self.action_num)
		a_max = np.argmax(q_values)
		e_prob[a_max] += (1.0 - self.epsilon)
		return e_prob


	def generate_episode(self, sess, sentence, train=True):

		states = []
		state_word = []
		state_pos = []
		state_dep = []
		actions = []
		rewards = []

		state = self.env.reset(sentence)
		done = False
		step_num = 0
		while not done:
			word_inputs_batch = [state[0]]
			pos_inputs_batch = [state[1]]
			dep_inputs_batch = [state[2]]
			action_prob = sess.run(self.actor_pred, feed_dict=self.actor_model.create_feed_dict([word_inputs_batch, \
																								 pos_inputs_batch, \
																								 dep_inputs_batch]))[0]
			# action_prob_e = self._epsilon_greedy_policy(action_prob)
			legal_labels = sentence.get_legal_labels(len(self.dataset.dep2idx), self.dataset.model_config.arc_only)
			prob_temp = action_prob * legal_labels  # remove illegal label
			if np.sum(prob_temp) == 0:
				prob_rescale = legal_labels / np.sum(legal_labels)
			else:
				prob_rescale = prob_temp / np.sum(prob_temp)
			# if np.sum(prob_rescale) == 0:
			# 	print(action_prob)
			# 	print(legal_labels)
			# 	print(prob_temp)
			action = np.random.choice(self.action_num, 1, p=action_prob)[0]

			step_num += 1
			next_state, reward, done = self.env.step(sentence, action, step_num)
			state_word.append(state[0])
			state_pos.append(state[1])
			state_dep.append(state[2])
			actions.append(action)
			rewards.append(reward)
			state = next_state
		states.append(state_word)
		states.append(state_pos)
		states.append(state_dep)
		if train:
			state = self.env.reset(sentence)

		return states, actions, rewards


	def train(self, sess, min_actor_lr=3e-5, min_critic_lr=1e-4, epsilon=1e-18, reward_scale=1e-2, gamma=0.95):
		# Trains the model on a single episode using A2C.
		sess.run(tf.global_variables_initializer())

		var_list = [v for v in tf.global_variables() if v.name.startswith('model/layer_connections') or v.name.startswith('model/feature_lookup')]
		saver = tf.train.Saver(var_list)
		saver.restore(sess, os.path.join("./data", "params_2018-05-02", "parser.weights"))

		for i in range(self.num_epoch):
			# random shuffle training data
			a = np.arange(len(self.dataset.train_data))
			np.random.shuffle(a)
			for k in range(a.shape[0]):
				idx = a[k]
				sentence = self.dataset.train_data[idx]
				states, actions, rewards = self.generate_episode(sess, sentence)

				values = sess.run(self.critic_model.output,
								  feed_dict=self.critic_model.create_feed_dict(states))
				T = len(rewards)
				R_t = np.zeros((T, 1))
				for t in reversed(range(T)):
					V_end = 0 if t + self.n >= T else values[t + self.n]
					R_t[t] = (gamma ** self.n) * V_end
					for j in range(min(T - t, self.n)):
						R_t[t] = R_t[t] + (gamma ** j) * rewards[t+j] * reward_scale
				adv = R_t - values

				# create one-hot label for actions
				word_len = len(actions)
				actions = np.array(actions)
				actions_onehot = np.zeros((word_len, self.action_num))
				actions_onehot[np.arange(word_len), actions] = 1

				# print(actions)
				# print(rewards)
				# update actor model
				feed_actor = self.actor_model.create_feed_dict(inputs_batch=states, labels_batch=actions_onehot)
				feed_actor[self.actions] = actions_onehot
				feed_actor[self.advantage] = adv
				feed_actor[self.actor_learning_rate] = self.actor_lr
				_, loss_actor, acc_actor, pred = sess.run([self.train_op_actor, self.loss_a, self.actor_model.accuracy, self.actor_pred], feed_dict=feed_actor)

				# update critic model
				feed_critic = self.critic_model.create_feed_dict(inputs_batch=states)
				feed_critic[self.Rt] = R_t
				feed_critic[self.critic_learning_rate] = self.critic_lr
				_, loss_critic = sess.run([self.train_op_critic, self.loss_c], feed_dict=feed_critic)

				if k % 50 == 0:
					sum_reward = np.sum(rewards)
					avg_reward = np.mean(rewards)
					if acc_actor == 0.5:
						print(actions)
						print(rewards)
						print(pred)
					print("{}, loss_a: {}, loss_c: {}, total reward: {}, avg reward: {}, actor acc: {}".format(k, loss_actor, loss_critic, sum_reward, avg_reward, acc_actor))
				if k % 200 == 0:
					self.test(sess)
					
			# test for every epoch
			print("finish {} epoch, start testing...".format(i))
			self.test(sess)


	def test(self, sess):

		reward_list = []
		for i, test_sent in enumerate(self.dataset.test_data):
			states, actions, rewards = self.generate_episode(sess, test_sent, False)
			rewards = np.sum(np.array(rewards))
			reward_list.append(rewards)
		r_mean = np.mean(reward_list)
		r_std = np.std(reward_list)
		print("test mean reward: {}, mean std: {}".format(r_mean, r_std))
		test_UAS, test_LAS = self.actor_model.get_UAS_LAS(self.dataset.test_data, self.dataset.dep2idx)
		print("test UAS: {}, test LAS: {}".format(test_UAS * 100, test_LAS * 100))
	


class Reinforce():

	def __init__(self, dataset, model, lr, num_episodes, env, e_sm=1e-8, name = None):

		self.dataset = dataset
		self.model = model
		self.lr = lr
		self.num_episodes = num_episodes
		self.e_sm = e_sm
		self.name = name

		self.env = env
		self.action_num = dataset.model_config.num_classes

		self.actions = tf.placeholder(tf.int32, shape=[None, ], name='actions')
		self.Gt = tf.placeholder(tf.float32, shape=[None, ], name='Gt')
		self.epsilon = tf.placeholder(tf.float32, shape=[], name='epsilon')
		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.train_op = self.build()

		self.training_episode = []
		self.reward_mean = []
		self.reward_error = []


	def build(self):
		self.model_pred = tf.nn.softmax(self.model.pred, axis=1)
		self.neg_log_prob = -tf.log(tf.reduce_sum(self.model_pred * tf.one_hot(self.actions, self.action_num), axis=1) + self.e_sm)

		self.loss = tf.reduce_mean(self.neg_log_prob * self.Gt)

		# train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		gvs = optimizer.compute_gradients(self.loss)
		self.capped_gvs = [(tf.clip_by_norm(grad, 1), var) for grad, var in gvs]
		train_op = optimizer.apply_gradients(self.capped_gvs)
		return train_op


	def generate_episode(self, sess, sentence, train=True):

		states = []
		state_word = []
		state_pos = []
		state_dep = []
		actions = []
		rewards = []

		state = self.env.reset(sentence)
		done = False
		step_num = 0
		while not done:
			word_inputs_batch = [state[0]]
			pos_inputs_batch = [state[1]]
			dep_inputs_batch = [state[2]]
			action_prob, logits = sess.run([self.model_pred, self.model.pred], \
											feed_dict=self.model.create_feed_dict([word_inputs_batch, pos_inputs_batch, dep_inputs_batch]))
			action_prob = action_prob[0]
			if np.math.isnan(action_prob[0]):
				print("action_prob: {}".format(action_prob))
				print("logits: {}".format(logits))

			# action_prob_e = self._epsilon_greedy_policy(action_prob)
			legal_labels = sentence.get_legal_labels(len(self.dataset.dep2idx), self.dataset.model_config.arc_only)
			prob_temp = action_prob * legal_labels  # remove illegal label
			# print("action_prob: {}".format(action_prob))
			# print("legal_labels: {}".format(legal_labels))
			if np.sum(prob_temp) == 0:
				prob_rescale = legal_labels / np.sum(legal_labels)
			else:
				prob_rescale = prob_temp / np.sum(prob_temp)
			action = np.random.choice(self.action_num, 1, p=prob_rescale)[0]

			step_num += 1
			next_state, reward, done = self.env.step(sentence, action, step_num)
			state_word.append(state[0])
			state_pos.append(state[1])
			state_dep.append(state[2])
			actions.append(action)
			rewards.append(reward)
			state = next_state
		states.append(state_word)
		states.append(state_pos)
		states.append(state_dep)
		if train:
			state = self.env.reset(sentence)

		return states, actions, rewards


	def train(self, sess, min_lr = 8e-4, epsilon = 1e-18, reward_scale=1e-2, gamma=0.99, k = 500):
		# Trains the model
		sess.run(tf.global_variables_initializer())

		for i in range(self.num_episodes):

			a = np.arange(len(self.dataset.train_data))
			np.random.shuffle(a)
			for k in range(a.shape[0]):
				idx = a[k]
				sentence = self.dataset.train_data[idx]

				states, actions, rewards = self.generate_episode(sess, sentence)

				discounted_reward = np.zeros_like(rewards)
				discounted_reward[-1] = rewards[-1]
				for t in reversed(range(len(rewards) - 1)):
					discounted_reward[t] += rewards[t] + gamma * discounted_reward[t + 1]

				discounted_reward = np.array(discounted_reward) * reward_scale

				feed_model = self.model.create_feed_dict(inputs_batch=states)
				feed_model[self.actions] = np.array(actions)
				feed_model[self.Gt] = np.array(discounted_reward)
				feed_model[self.learning_rate] = self.lr
				l, _ = sess.run([self.loss, self.train_op], feed_dict=feed_model)

				print("k: {}: loss: {}".format(k, l))

				# if k % 50 == 0:
				# 	sum_reward = np.sum(rewards)
				# 	avg_reward = np.mean(rewards)
				# 	print("{}, loss: {}, total reward: {}, avg reward: {}".format(k, l, sum_reward, avg_reward))
				
				# if k % 200 == 0:
				# 	self.test(sess)


	def test(self, sess):

		reward_list = []
		for i, test_sent in enumerate(self.dataset.test_data):
			states, actions, rewards = self.generate_episode(sess, test_sent, False)
			rewards = np.sum(np.array(rewards))
			reward_list.append(rewards)
		r_mean = np.mean(reward_list)
		r_std = np.std(reward_list)
		print("test mean reward: {}, mean std: {}".format(r_mean, r_std))
		test_UAS, test_LAS = self.model.get_UAS_LAS(self.dataset.test_data, self.dataset.dep2idx)
		print("test UAS: {}, test LAS: {}".format(test_UAS * 100, test_LAS * 100))
	


def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-epoch', dest='num_epoch', type=int,
						default=20, help="Number of episodes to train on.")
	parser.add_argument('--actor-lr', dest='actor_lr', type=float,
						default=1e-3, help="The actor's learning rate.") # 3e-3
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-2, help="The critic's learning rate.") # 1e-2
	parser.add_argument('--n', dest='n', type=int,
						default=20, help="The value of N in N-step A2C.")
	parser.add_argument('--epsilon', dest='epsilon', type=float,
						default=0.5, help="The epsilon used in epsilon greedy policy.")
	parser.add_argument('--name', dest='name', type=str,
						default='', help="name.")
	parser.add_argument('--arc-only', dest='arc_only', type=bool, 
						default=True, help="if action number is 3")
	return parser.parse_args()



def main_A2C():

	args = parse_arguments()
	num_epoch = args.num_epoch
	actor_lr = args.actor_lr
	critic_lr = args.critic_lr
	n = args.n
	epsilon = args.epsilon
	name = args.name
	arc_only = args.arc_only

	dataset = load_datasets(arc_only, True)
	config = dataset.model_config
	env = ParserEnv(dataset, arc_only)
	sess = tf.Session()
	with tf.variable_scope("model") as model_scope:
		actor_model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix, dataset.dep_embedding_matrix)
		critic_model = Critic(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix, dataset.dep_embedding_matrix)
		model = A2C(dataset, env, dataset.model_config.num_classes, actor_model, actor_lr, critic_model, critic_lr, num_epoch, epsilon, 1e-8, n, name)
		model.train(sess)



def main_reinforce():

	args = parse_arguments()
	num_epoch = args.num_epoch
	actor_lr = args.actor_lr
	critic_lr = args.critic_lr
	n = args.n
	epsilon = args.epsilon
	name = args.name
	arc_only = args.arc_only

	dataset = load_datasets(arc_only, True)
	config = dataset.model_config
	env = ParserEnv(dataset, arc_only)
	sess = tf.Session()
	with tf.variable_scope("model") as model_scope:
		model_r = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix, dataset.dep_embedding_matrix)
		model = Reinforce(dataset, model_r, actor_lr, num_epoch, env)
		model.train(sess)


if __name__ == "__main__":
	main_A2C()
	# main_reinforce()

