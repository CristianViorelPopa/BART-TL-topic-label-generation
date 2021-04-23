import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from survey_utils import get_model_scores_with_topics


def preprocess_label(label):
	return label.split('(')[0].replace('_', ' ').strip()


def get_num_new_labels_by_topic(model_dir, indices, max_labels_per_topic):
	topics = []
	train_labels_by_topic = []
	all_train_labels = []
	prev_topic = ''
	with open(os.path.join(model_dir, 'dataset_fairseq', 'train.source')) as source_dataset, \
			open(os.path.join(model_dir, 'dataset_fairseq', 'train.target')) as target_dataset:

		while True:
			source_line = source_dataset.readline()
			target_line = target_dataset.readline()

			if source_line.strip() == '':
				break

			if prev_topic != source_line:
				train_labels_by_topic.append([])
				topics.append(source_line)
				prev_topic = source_line

			label = target_line.strip()

			train_labels_by_topic[-1].append(label)
			all_train_labels.append(label)

	# add the labels from NETL
	with open('summarization/output_unsupervised_netl_all_before_v2') as f:
		count = 0
		for line in f:
			train_labels_by_topic[count].extend(map(lambda label: preprocess_label(label), line.strip().split(' ')))
			train_labels_by_topic[count] = list(set(train_labels_by_topic[count]))
			count += 1
	with open('summarization/output_supervised_netl_all_before_v2') as f:
		count = 0
		for line in f:
			train_labels_by_topic[count].extend(map(lambda label: preprocess_label(label), line.strip().split(' ')))
			train_labels_by_topic[count] = list(set(train_labels_by_topic[count]))
			count += 1

	pred_labels_by_topic = [[preprocess_label(label) for label in line.strip().split(' ')][:max_labels_per_topic]
							for line in open(os.path.join(model_dir, 'summaries', 'train2_25.hypo'))]

	num_new_labels_by_topic = []
	new_labels_dict = {}
	for topic_idx in indices:
		count = 0
		for label in pred_labels_by_topic[topic_idx]:
			if label not in train_labels_by_topic[topic_idx]:
				if topics[topic_idx] not in new_labels_dict:
					new_labels_dict[topics[topic_idx]] = []
				new_labels_dict[topics[topic_idx]].append(label)
				count += 1
		num_new_labels_by_topic.append(count)
	num_new_labels_by_topic = np.array(num_new_labels_by_topic)

	return num_new_labels_by_topic, new_labels_dict


def get_rating_new_labels_by_topic(model_dir, indices, max_labels_per_topic, model_scores):
	topics = []
	train_labels_by_topic = []
	all_train_labels = []
	prev_topic = ''
	with open(os.path.join(model_dir, 'dataset_fairseq', 'train.source')) as source_dataset, \
			open(os.path.join(model_dir, 'dataset_fairseq', 'train.target')) as target_dataset:

		while True:
			source_line = source_dataset.readline()
			target_line = target_dataset.readline()

			if source_line.strip() == '':
				break

			if prev_topic != source_line:
				train_labels_by_topic.append([])
				topics.append(source_line)
				prev_topic = source_line

			label = target_line.strip()

			train_labels_by_topic[-1].append(label)
			all_train_labels.append(label)

	# add the labels from NETL
	with open('summarization/output_unsupervised_netl_all_before_v2') as f:
		count = 0
		for line in f:
			train_labels_by_topic[count].extend(map(lambda label: preprocess_label(label), line.strip().split(' ')))
			train_labels_by_topic[count] = list(set(train_labels_by_topic[count]))
			count += 1
	with open('summarization/output_supervised_netl_all_before_v2') as f:
		count = 0
		for line in f:
			train_labels_by_topic[count].extend(map(lambda label: preprocess_label(label), line.strip().split(' ')))
			train_labels_by_topic[count] = list(set(train_labels_by_topic[count]))
			count += 1

	pred_labels_by_topic = [[preprocess_label(label) for label in line.strip().split(' ')][:max_labels_per_topic]
							for line in open(os.path.join(model_dir, 'summaries', 'train2_25.hypo'))]

	num_new_labels_by_topic = []
	new_labels_dict = {}
	new_labels_ratings = {}
	# for topic_idx in range(len(pred_labels_by_topic)):
	for topic_idx in indices:
		topic_scores = {label: score for label, score in model_scores[' '.join(topics[topic_idx].split(' ')[:10])]}
		count = 0
		for label in pred_labels_by_topic[topic_idx]:
			if label not in train_labels_by_topic[topic_idx]:
				if topics[topic_idx] not in new_labels_dict:
					new_labels_dict[topics[topic_idx]] = []
				new_labels_dict[topics[topic_idx]].append(label)
				if topics[topic_idx] not in new_labels_ratings:
					new_labels_ratings[topics[topic_idx]] = []
				if label in topic_scores:
					new_labels_ratings[topics[topic_idx]].append(topic_scores[label])

				count += 1
		num_new_labels_by_topic.append(count)
	num_new_labels_by_topic = np.array(num_new_labels_by_topic)

	return num_new_labels_by_topic, new_labels_dict, new_labels_ratings


def main():
	if len(sys.argv) < 4:
		print("Usage: " + sys.argv[0] + " <indices file> <output image> [<model directory>]+")
		exit(0)

	limit = 10

	num_topics_per_subject = 6
	indices = []
	with open(sys.argv[1]) as indices_file:
		indices.append([])
		for line in indices_file:
			line = line.strip()
			if line == '---':
				indices[-1] = indices[-1][:num_topics_per_subject]
				indices.append([])
			else:
				indices[-1].append(int(line))
	indices[-1] = indices[-1][:num_topics_per_subject]
	indices = np.array(indices).flatten()







	#================= QUALITY ==================#

	survey_csv_responses = [
		"summarization/survey/surveys/june_11/Topic Labeling Survey.csv"
	]
	topics_csv = "bart-tl/topics_all_new_v2.csv"
	model_hypos_file = "bart-tl/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/train2_25.hypo"

	model_scores, sufficiently_annotated_labels, insufficiently_annotated_labels, total_annotated_labels = get_model_scores_with_topics(
		survey_csv_responses, topics_csv, model_hypos_file, 2, None, False)

	plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	font = {'size': 15}
	matplotlib.rc('font', **font)
	plt.ylim([2.0, 3.0])
	plt.ylabel('Average rating of new labels')
	plt.xlabel('Number of top labels')
	plt.xticks(list(range(1, limit + 1)))
	plt.yticks([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])

	model_novelty_quality = []
	idx = 0

	num_new_labels_by_topic = []
	new_labels_dict = {}
	for model_dir in sys.argv[3:]:
		model_novelty_quality.append([])
		num_new_labels_by_topic.append([])
		for max_labels_per_topic in range(1, limit + 1):
			model_num_new_labels_by_topic, model_new_labels_dict, model_new_labels_ratings = get_rating_new_labels_by_topic(model_dir, indices, max_labels_per_topic, model_scores)
			num_new_labels_by_topic[-1].append(np.mean(model_num_new_labels_by_topic) / max_labels_per_topic)
			for topic, labels_ in model_new_labels_dict.items():
				if topic not in new_labels_dict:
					new_labels_dict[topic] = []
				new_labels_dict[topic] += model_new_labels_dict[topic]
			model_novelty_quality[idx].append(np.mean([np.mean(x) for x in np.array(list(model_new_labels_ratings.values())) if x]))

		idx += 1

	plt.plot(list(range(1, limit + 1)), model_novelty_quality[1], '.-.', linewidth=2,
			 color='black')
	plt.plot(list(range(1, limit + 1)), model_novelty_quality[0], 'x:', linewidth=2,
			 color='dimgrey')
	plt.legend(['BART-TL-ng', 'BART-TL-all'])
	plt.savefig(sys.argv[2])

	new_labels_dict = {' '.join(topic.split(' ')[:10]): new_labels for topic, new_labels in new_labels_dict.items()}

	total_score_sum = 0
	total_labels_scored = 0

	for topic, new_labels in new_labels_dict.items():
		topic_scores = {label: score for label, score in model_scores[topic]}

		for label in set(new_labels):
			if label in topic_scores:
				total_score_sum += topic_scores[label]
				total_labels_scored += 1

	# ================= QUALITY ==================#


	print('Average score of new labels: ' + str(total_score_sum / total_labels_scored))


if __name__ == '__main__':
	main()
