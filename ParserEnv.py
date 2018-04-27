import numpy as np
from utils.feature_extraction import *
from parser_model import *


class ParserEnv():

	def __init__(self, dataset):

		self.LEFT = 0
		self.RIGHT = 1
		self.SHIFT = 2
		self.dataset = dataset

	
	def step(self, sentence, action):

		# update state inside sentence
		if action != self.SHIFT:
			sentence.update_child_dependencies(action)
		sentence.update_state_by_transition(action, gold=False)
		# extract new state from sentence
		new_state = self.dataset.feature_extractor.extract_for_current_state(sentence, \
        																	 self.dataset.word2idx, \
        																	 self.dataset.pos2idx, \
        																	 self.dataset.dep2idx)
       	# compute reward
		if action == self.SHIFT:
			reward = 0
		else:
			h, t = sentence.predicted_dependencies[-1] # [h, t]
			if t.head_id != h.token_id:
				reward = -1
			else:
				reward = 1
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


def test():

	dataset = load_datasets(False)
	config = dataset.model_config
	model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix, dataset.dep_embedding_matrix)

	env = ParserEnv(dataset)
	sent_test = dataset.train_data[0]

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	done = False
	cur_input = env.reset(sent_test)
	while not done:

		word_inputs_batch = [cur_input[0]]
		pos_inputs_batch = [cur_input[1]]
		dep_inputs_batch = [cur_input[2]]
		predictions = sess.run(model.pred, feed_dict=model.create_feed_dict([word_inputs_batch, pos_inputs_batch,dep_inputs_batch]))
		legal_labels = np.asarray([sent_test.get_legal_labels()], dtype=np.float32)
		legal_transitions = np.argmax(predictions + 1000 * legal_labels, axis=1)

		new_state, reward, done = env.step(sent_test, legal_transitions)
		cur_input = new_state
		print("reward: {}".format(reward))


if __name__ == "__main__":
	test()


