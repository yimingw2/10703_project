import numpy as np
from utils.feature_extraction import *
from parser_model import *


class ParserEnv():

	def __init__(self, dataset):

		self.dataset = dataset
		self.num_classes = self.dataset.model_config.num_classes

	
	def step(self, sentence, action):

		# update state inside sentence
		if action == 0:
			real_action = 0
			arc_label = -1
		else:
			action_temp = action - 1
			real_action = action_temp % 2 + 1 
			arc_label = action_temp / 2

		if real_action != 0:
			sentence.update_child_dependencies(real_action, arc_label)
		
		sentence.update_state_by_transition(real_action, arc_label, gold=False)
		# extract new state from sentence
		new_state = self.dataset.feature_extractor.extract_for_current_state(sentence, \
        																	 self.dataset.word2idx, \
        																	 self.dataset.pos2idx, \
        																	 self.dataset.dep2idx)
       	# compute reward
		if real_action == 0:
			reward = 0
		else:
			h, t, l = sentence.predicted_dependencies[-1] # [h, t]
			if t.head_id != h.token_id:
				reward = -1
			else:
				reward = 1
				if l == self.dataset.dep2idx[t.dep]:
					reward += 4
		# done
		if len(sentence.stack) == 1 and len(sentence.buff) == 0:
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


# class Critic():

# 	def __init__():
		


def test():

	dataset = load_datasets(False)
	config = dataset.model_config
	model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix, dataset.dep_embedding_matrix)

	env = ParserEnv(dataset)
	sent_test = dataset.train_data[1]

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	done = False
	cur_input = env.reset(sent_test)
	while not done:

		word_inputs_batch = [cur_input[0]]
		pos_inputs_batch = [cur_input[1]]
		dep_inputs_batch = [cur_input[2]]
		predictions = sess.run(model.pred, feed_dict=model.create_feed_dict([word_inputs_batch, pos_inputs_batch,dep_inputs_batch]))
		legal_labels = np.asarray([sent_test.get_legal_labels(dataset.model_config.dep_vocab_size)], dtype=np.float32)
		legal_transitions_whole = np.argmax(predictions + 1000 * legal_labels, axis=1)
		new_state, reward, done = env.step(sent_test, legal_transitions_whole)
		cur_input = new_state
		print("reward: {}".format(reward))


if __name__ == "__main__":
	test()

