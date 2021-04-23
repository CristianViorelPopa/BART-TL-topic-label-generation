import os
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')  # run once
stop_words = stopwords.words('english')

invalid_tokens = stop_words + \
				 ['we', 'once', 'that', 'you', 'i', 'an', 'yet', 'whom', 'was', 'being',
				  'my', 'our', 'their', 'he', 'she', 'us', 'them', 'him', 'her', 'twice',
				  'had', 'the', 'your', 'his', 'would', 'mr.', 'sir', 'not', 'ok', 'also',
				  'mr', 'one', 'no', 'while', 'as']
invalid_tokens = list(set(invalid_tokens))


def preprocess_label(label):
	return label.split('(')[0].replace('_', ' ').strip()


def dcg(scores):
	result = 0
	for idx, score in enumerate(scores):
		result += (2 ** score - 1) / np.log2(idx + 2)
	return result


def ndcg(scores, n):
	target_scores = scores[:n]
	perfect_scores = sorted(scores, reverse=True)[:n]

	return dcg(target_scores) / dcg(perfect_scores)


def get_new_labels_scores(model_dir, topic_labels_scores, old=False):
	train_labels_by_topic = []
	all_train_labels = []
	prev_topic = ''
	topic_terms = []
	with open(os.path.join(model_dir, 'dataset_fairseq', 'train.source')) as source_dataset, \
			open(os.path.join(model_dir, 'dataset_fairseq', 'train.target')) as target_dataset:

		while True:
			source_line = source_dataset.readline()
			target_line = target_dataset.readline()

			if source_line.strip() == '':
				break

			if prev_topic != source_line:
				train_labels_by_topic.append([])
				prev_topic = source_line
				topic_terms.append(' '.join(source_line.strip().split(' ')[:10]))

			label = target_line.strip()

			train_labels_by_topic[-1].append(label)
			all_train_labels.append(label)

	pred_labels_by_topic = [
		[(topic_terms, preprocess_label(label)) for label in line.strip().split(' ')]
		for topic_terms, line in
		zip(topic_terms, [l for l in open(os.path.join(model_dir, 'summaries', 'train2_25.hypo'))])]

	new_labels_scores = []
	for topic_idx in range(len(pred_labels_by_topic)):
		new_labels_scores.append([])
		for topic_terms, label in pred_labels_by_topic[topic_idx]:
			if topic_terms not in topic_labels_scores:
				continue
			if not old:
				if label not in train_labels_by_topic[topic_idx]:
					try:
						if topic_labels_scores[topic_terms][label] != -1.0:
							new_labels_scores[-1].append(topic_labels_scores[topic_terms][label])
					except:
						# bugged question
						continue
			else:
				if label in train_labels_by_topic[topic_idx]:
					try:
						if topic_labels_scores[topic_terms][label] != -1.0:
							new_labels_scores[-1].append(topic_labels_scores[topic_terms][label])
					except:
						# bugged question
						continue

	return new_labels_scores


def main():
	if len(sys.argv) < 2:
		print("Usage: " + sys.argv[0] + " [<survey responses csv>]+ <model directory>")
		exit(0)

	answer_values = {
		'Not relevant at all': 0,
		'A bit relevant': 1,
		# 'Somehow relevant': 2,
		# 'Relevant': 3,
		'Relevant': 2,
		# 'Very relevant': 4
		'Very relevant': 3
	}

	topic_labels_scores = {}
	target_topic_terms = []
	for csv_file in sys.argv[1:-1]:
		responses = pd.read_csv(csv_file)

		invalid_tokens_scores = []
		for _ in range(responses.shape[0]):
			invalid_tokens_scores.append([])

		# skip timestamp and nickname columns
		for question in responses.columns[2:]:
			# bugged question
			if len(question.split('\n')) == 1:
				continue

			else:
				topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:].split(', '))

			label = question.split('[')[1].split(']')[0]
			if topic_terms not in topic_labels_scores:
				topic_labels_scores[topic_terms] = {}

			answers = responses[question].values

			if label in invalid_tokens:
				for idx in range(len(answers)):
					if not pd.isnull(answers[idx]):
						invalid_tokens_scores[idx].append(answer_values[answers[idx]])

			answers = answers[~pd.isnull(answers)]
			answers = [answer_values[answer] for answer in answers]

			if len(answers) > 0:
				topic_labels_scores[topic_terms][label] = np.mean(answers)
			else:
				topic_labels_scores[topic_terms][label] = -1.0

			# only account for the topics in the last survey
			if csv_file == sys.argv[-3]:
				target_topic_terms.append(topic_terms)

		valid_answers = [sum(np.array(user_answers) <= 1) >= 0.75 * len(user_answers)
						 for user_answers in invalid_tokens_scores]
		if not np.all(valid_answers):
			print('Not all answers passed the validation!')

	model_dir = sys.argv[-1]
	new_labels_scores = np.array(list(filter(bool, get_new_labels_scores(model_dir,
																		 topic_labels_scores))))
	old_labels_scores = np.array(list(filter(bool, get_new_labels_scores(model_dir,
																		 topic_labels_scores,
																		 old=True))))

	print("NEW")
	print("-----")
	print("Top-1 Average Rating: " + str(np.mean([scores[0] for scores in new_labels_scores])))
	print("Top-3 Average Rating: " + str(np.mean([scores[:3] for scores in new_labels_scores])))
	print("Top-5 Average Rating: " + str(np.mean([scores[:5] for scores in new_labels_scores])))
	print("All-labels Average Rating: " + str(
		np.mean([np.mean(scores) for scores in new_labels_scores])))
	print("nDCG-1: " + str(np.mean([ndcg(scores, 1) for scores in new_labels_scores])))
	print("nDCG-3: " + str(np.mean([ndcg(scores, 3) for scores in new_labels_scores])))
	print("nDCG-5: " + str(np.mean([ndcg(scores, 5) for scores in new_labels_scores])))
	print("")
	print("OLD")
	print("-----")
	print("Top-1 Average Rating: " + str(np.mean([scores[0] for scores in old_labels_scores])))
	print("Top-3 Average Rating: " + str(np.mean([scores[:3] for scores in old_labels_scores])))
	print("Top-5 Average Rating: " + str(np.mean([scores[:5] for scores in old_labels_scores])))
	print("All-labels Average Rating: " + str(
		np.mean([np.mean(scores) for scores in old_labels_scores])))
	print("nDCG-1: " + str(np.mean([ndcg(scores, 1) for scores in old_labels_scores])))
	print("nDCG-3: " + str(np.mean([ndcg(scores, 3) for scores in old_labels_scores])))
	print("nDCG-5: " + str(np.mean([ndcg(scores, 5) for scores in old_labels_scores])))


if __name__ == '__main__':
	main()
