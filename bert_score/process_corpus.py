import argparse
import os
import pickle

import random
import numpy as np
from numpy.random import choice
import nltk
from nltk.corpus import stopwords
import spacy

sp = spacy.load('en_core_web_sm')


# example run: python3 bert_score/process_corpus.py -i bert_score/topic_labelling/data/wiki_sent/ -o summarization/dataset_fairseq --only-train --clean_labels

nltk.download('stopwords')  # run once
stop_words = stopwords.words('english')
stop_words = stop_words + ['nbsp', 'would', 'whilst', 'whereas', 'although', 'often', 'one']

names = []
with open(os.path.join(os.path.dirname(__file__), 'names', 'names', 'dist.male.first')) as f:
	for line in f:
		names.append(line.split(' ')[0].lower())
with open(os.path.join(os.path.dirname(__file__), 'names', 'names', 'dist.female.first')) as f:
	for line in f:
		names.append(line.split(' ')[0].lower())
with open(os.path.join(os.path.dirname(__file__), 'names', 'names', 'dist.all.last')) as f:
	for line in f:
		names.append(line.split(' ')[0].lower())

useless_labels = set(stop_words + names)

CLEAN_LABELS = False
MINIMUM_LABEL_LEN = 3

perfect_pos_distr = {
	'NOUN': 0.47,
	'PROPN': 0.42,
	'ADJ': 0.06,
	'VERB': 0.04
}

ngram_distr = {
	2: 0.1,
	3: 0.6,
	4: 0.3
}


# def preprocess_raw(input, tokenizer, are_labels=False, ngram_prob=0.0):
# 	processed_input = np.array(list(map(lambda s: s.replace('sostok', '').replace('eostok', '').strip(), tokenizer.sequences_to_texts(input))))
# 	remaining_indices = np.array(range(len(processed_input)))
#
# 	if CLEAN_LABELS and are_labels:
# 		processed_input = [' '.join([token for token in label.split(' ') if token not in useless_labels]).strip() for label in processed_input]
#
# 		# remove empty labels and store indices
# 		remaining_indices = np.array([idx for idx, label in enumerate(processed_input) if len(label) > MINIMUM_LABEL_LEN])
# 		processed_input = np.array([label for label in processed_input if len(label) > MINIMUM_LABEL_LEN])
#
# 	# if ngram_prob > 0.0 and are_labels:
# 	# 	new_samples_count = len(remaining_indices) *
# 	# 	random.sample()
#
# 	# source lines cannot be empty, so set them to a dummy character
# 	if not are_labels:
# 		processed_input[processed_input == ''] = '@'
#
# 	return processed_input, remaining_indices


def filter_pos(input, aggressive=True):
	filtered_topics = []
	for idx, topic in enumerate(input):
		sen = sp(str(topic))
		num_useless_tokens = 0

		topic_pos_count = {
			'NOUN': 0,
			'PROPN': 0,
			'ADJ': 0,
			'VERB': 0
		}

		for token in sen:
			if token.pos_ not in perfect_pos_distr:
				num_useless_tokens += 1
			else:
				topic_pos_count[token.pos_] += 1

		# compute the absolute maximum number of PoS that should be included following the perfect distribution
		max_pos_distr = {}
		for pos, percentage in perfect_pos_distr.items():
			max_pos_distr[pos] = int(percentage * (len(sen) - num_useless_tokens)) + 1

		if aggressive:
			# compute the PoS that appears the least
			min_percentage = 1.0
			for pos, max_count in max_pos_distr.items():
				if (topic_pos_count[pos] / max_count) < min_percentage:
					min_percentage = topic_pos_count[pos] / max_count

			# cap all the other PoS according to the least appearing
			for pos in max_pos_distr:
				max_pos_distr[pos] = np.ceil(max_pos_distr[pos] * min_percentage)

		# iterate through the topic terms and decide to include them or not based on remaining spots
		filtered_topic = ''
		for token in sen:
			if token.pos_ in max_pos_distr and max_pos_distr[token.pos_] > 0:
				filtered_topic += str(token) + ' '
				max_pos_distr[token.pos_] -= 1

		filtered_topics.append(filtered_topic.strip())

		print('Progress: ' + str(idx) + '/' + str(len(input)), end='\r')
	print()

	return filtered_topics


def sample_topic_terms(input, num_samples):
	random.seed(0)
	processed_topics = []
	for topic in input:
		topic_terms = topic.split(' ')
		if num_samples < len(topic_terms):
			topic_terms = random.sample(topic_terms, num_samples)
		processed_topics.append(' '.join(topic_terms))

	return processed_topics


def process(input, output, input_tokenizer, output_tokenizer, aggressive_pos_filtering=False, pos_filtering=False, sample=False, ngram_prob=0.0):
	input = np.array(list(map(lambda s: s.replace('sostok', '').replace('eostok', '').strip(), input_tokenizer.sequences_to_texts(input))))
	output = np.array(list(map(lambda s: s.replace('sostok', '').replace('eostok', '').strip(), output_tokenizer.sequences_to_texts(output))))

	# label filtering
	if CLEAN_LABELS:
		output = np.array([' '.join([token for token in label.split(' ') if token not in useless_labels]).strip() for label in output])

		# remove empty labels and store indices
		remaining_indices = [idx for idx, label in enumerate(output) if len(label) > MINIMUM_LABEL_LEN]
		output = output[remaining_indices]
		input = input[remaining_indices]

	# topic processing
	if int(aggressive_pos_filtering) + int(pos_filtering) + int(sample) > 1:
		raise RuntimeError('You can only select one of: aggressive POS filtering, regular POS filtering and sampling')

	if aggressive_pos_filtering:
		input = filter_pos(input, aggressive=True)
	elif pos_filtering:
		input = filter_pos(input, aggressive=False)
	elif sample:
		input = sample_topic_terms(input, sample)

	# ngram labels augmentation
	if ngram_prob > 0.0:
		new_samples_count = int(len(output) * ngram_prob)
		selected_indices = random.sample(range(len(output)), new_samples_count)

		new_labels = []
		for idx in selected_indices:
			ngram_length = choice(list(ngram_distr.keys()), 1, p=list(ngram_distr.values()))
			if ngram_length > len(input[idx].split(' ')):
				ngram_length = len(input[idx].split(' '))
			ngram = ' '.join(choice(input[idx].split(' '), ngram_length, replace=False))
			new_labels.append(ngram)

		input = np.concatenate((input, input[selected_indices]))
		output = np.concatenate((output, new_labels))

	return input, output


def main():
	parser = argparse.ArgumentParser(
		description='Scoring of model based on BERTScore against gold NETL labels'
	)
	parser.add_argument('--input-dir', '-i', required=True, type=str,
						help='Directory containing the corpus files used in the experiments of Aliokaili et al. (2020)')
	parser.add_argument('--output-dir', '-o', required=True, type=str,
						help='Name of the output file')
	parser.add_argument('--only-train', required=False, action='store_true',
						help='Whether to dump all the available data into the train file and none into the test file')
	parser.add_argument('--clean-labels', required=False, action='store_true',
						help='Whether to remove stopwords from the target labels')
	parser.add_argument('--pos-filtering', required=False, action='store_true', default=False,
						help='Whether to filter the topic terms to respect the original NETL distribution')
	parser.add_argument('--aggressive-pos-filtering', required=False, action='store_true', default=False,
						help='Whether to AGGRESSIVELY (distribution closer to perfect, but may dramatically decrease average size of topics) filter the topic terms to respect the original NETL distribution')
	parser.add_argument('--sample', required=False, type=int, default=False,
						help='Sample size for the topic terms. If this is not set, all the topic terms are included (unless filtering is set)')
	parser.add_argument('--ngram-label-prob', required=False, type=float, default=0.0,
						help='Chance that an additional sample with the same terms, but an n-gram label will be generated')
	args = parser.parse_args()

	global CLEAN_LABELS
	if args.clean_labels:
		CLEAN_LABELS = True

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# input text (x) loading
	input_tok = pickle.load(open(os.path.join(args.input_dir, 'x_tokenizer.pickle'), 'rb'))
	train_input = np.load(os.path.join(args.input_dir, 'x_tr.npy'))
	val_input = np.load(os.path.join(args.input_dir, 'x_val.npy'))
	test_input = np.load(os.path.join(args.input_dir, 'x_test.npy'))

	train_input = np.concatenate((train_input, val_input))
	if args.only_train:
		train_input = np.concatenate((train_input, test_input))

	# output text (y) loading
	output_tok = pickle.load(open(os.path.join(args.input_dir, 'y_tokenizer.pickle'), 'rb'))
	train_output = np.load(os.path.join(args.input_dir, 'y_tr.npy'))
	val_output = np.load(os.path.join(args.input_dir, 'y_val.npy'))
	test_output = np.load(os.path.join(args.input_dir, 'y_test.npy'))

	train_output = np.concatenate((train_output, val_output))
	if args.only_train:
		train_output = np.concatenate((train_output, test_output))

	# actual processing
	train_input, train_output = process(train_input, train_output, input_tok, output_tok,
										aggressive_pos_filtering=args.aggressive_pos_filtering,
										pos_filtering=args.pos_filtering, sample=args.sample,
										ngram_prob=args.ngram_label_prob)

	with open(os.path.join(args.output_dir, 'train.target'), 'w') as out_file:
		out_file.write('\n'.join(train_output))

	with open(os.path.join(args.output_dir, 'train.source'), 'w') as out_file:
		out_file.write('\n'.join(train_input))

	if not args.only_train:
		test_input, test_output = process(test_input, test_output, input_tok, output_tok,
										  aggressive_pos_filtering=args.aggressive_pos_filtering,
										  pos_filtering=args.pos_filtering, sample=args.sample,
										  ngram_prob=0.0)

		with open(os.path.join(args.output_dir, 'test.target'), 'w') as out_file:
			out_file.write('\n'.join(test_output))

		with open(os.path.join(args.output_dir, 'test.source'), 'w') as out_file:
			out_file.write('\n'.join(test_input))

	else:
		with open(os.path.join(args.output_dir, 'test.target'), 'w') as out_file:
			out_file.write('\n')

		with open(os.path.join(args.output_dir, 'test.source'), 'w') as out_file:
			out_file.write('\n')

	# # output text (y)
	# tok = pickle.load(open(os.path.join(args.input_dir, 'y_tokenizer.pickle'), 'rb'))
	# with open(os.path.join(args.output_dir, 'train.target'), 'w') as out_file:
	# 	train_output = np.load(os.path.join(args.input_dir, 'y_tr.npy'))
	# 	train_output, train_remaining_indices = preprocess_raw(train_output, tok, are_labels=True)
	# 	out_file.write('\n'.join(train_output))
	# 	out_file.write('\n')
	# 	val_output = np.load(os.path.join(args.input_dir, 'y_val.npy'))
	# 	val_output, val_remaining_indices = preprocess_raw(val_output, tok, are_labels=True)
	# 	out_file.write('\n'.join(val_output))
	#
	# 	if args.only_train:
	# 		test_output = np.load(os.path.join(args.input_dir, 'y_test.npy'))
	# 		test_output, test_remaining_indices = preprocess_raw(test_output, tok, are_labels=True)
	# 		out_file.write('\n'.join(test_output))
	#
	# with open(os.path.join(args.output_dir, 'test.target'), 'w') as out_file:
	# 	if not args.only_train:
	# 		test_output = np.load(os.path.join(args.input_dir, 'y_test.npy'))
	# 		test_output, test_remaining_indices = preprocess_raw(test_output, tok, are_labels=True)
	# 		out_file.write('\n'.join(test_output))
	# 	else:
	# 		# newline, so the file is not totally empty
	# 		out_file.write('\n')
	#
	# # input text (x)
	# tok = pickle.load(open(os.path.join(args.input_dir, 'x_tokenizer.pickle'), 'rb'))
	# with open(os.path.join(args.output_dir, 'train.source'), 'w') as out_file:
	# 	train_input = np.load(os.path.join(args.input_dir, 'x_tr.npy'))[train_remaining_indices]
	# 	train_input, _ = preprocess_raw(train_input, tok)
	# 	if args.aggressive_pos_filtering:
	# 		train_input = pos_filtering(train_input, aggressive=True)
	# 	elif args.pos_filtering:
	# 		train_input = pos_filtering(train_input, aggressive=False)
	# 	elif args.sample:
	# 		train_input = sample_topic_terms(train_input, args.sample)
	# 	out_file.write('\n'.join(train_input))
	# 	out_file.write('\n')
	# 	val_input = np.load(os.path.join(args.input_dir, 'x_val.npy'))[val_remaining_indices]
	# 	val_input, _ = preprocess_raw(val_input, tok)
	# 	if args.aggressive_pos_filtering:
	# 		val_input = pos_filtering(val_input, aggressive=True)
	# 	elif args.pos_filtering:
	# 		val_input = pos_filtering(val_input, aggressive=False)
	# 	elif args.sample:
	# 		val_input = sample_topic_terms(val_input, args.sample)
	# 	out_file.write('\n'.join(val_input))
	#
	# 	if args.only_train:
	# 		test_input = np.load(os.path.join(args.input_dir, 'x_test.npy'))[test_remaining_indices]
	# 		test_input, _ = preprocess_raw(test_input, tok)
	# 		if args.aggressive_pos_filtering:
	# 			test_input = pos_filtering(test_input, aggressive=True)
	# 		elif args.pos_filtering:
	# 			test_input = pos_filtering(test_input, aggressive=False)
	# 		elif args.sample:
	# 			test_input = sample_topic_terms(test_input, args.sample)
	# 		out_file.write('\n'.join(test_input))
	#
	# with open(os.path.join(args.output_dir, 'test.source'), 'w') as out_file:
	# 	if not args.only_train:
	# 		test_input = np.load(os.path.join(args.input_dir, 'x_test.npy'))[test_remaining_indices]
	# 		test_input, _ = preprocess_raw(test_input, tok)
	# 		if args.aggressive_pos_filtering:
	# 			test_input = pos_filtering(test_input, aggressive=True)
	# 		elif args.pos_filtering:
	# 			test_input = pos_filtering(test_input, aggressive=False)
	# 		elif args.sample:
	# 			test_input = sample_topic_terms(test_input, args.sample)
	# 		out_file.write('\n'.join(test_input))
	# 	else:
	# 		# newline, so the file is not totally empty
	# 		out_file.write('\n')


if __name__ == '__main__':
	main()
