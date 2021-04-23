import sys
import os
import json
import numpy as np
from survey_utils import get_model_scores

import matplotlib
from matplotlib import pyplot as plt


def dcg(scores):
	result = 0
	for idx, score in enumerate(scores):
		result += (2 ** score - 1) / np.log2(idx + 2)
	return result


def ndcg(scores, n):
	target_scores = scores[:n]
	perfect_scores = sorted(scores, reverse=True)[:n]

	return dcg(target_scores) / dcg(perfect_scores)


def main():
	if len(sys.argv) < 6:
		print("Usage: " + sys.argv[0] + " [<survey csv responses>]+ <topics csv> <model hypos> <indices file> <model scores> <output image>")
		exit(0)

	# subject_partitioning = [
	# 	('all', 2, None, False),
	# 	('english', 0 * 10 + 2, 18 * 10 + 2, True),
	# 	('biology', 18 * 10 + 2, 37 * 10 + 2, True),
	# 	('economics', 37 * 10 + 2, 55 * 10 + 2, True),
	# 	('law', 55 * 10 + 2, 73 * 10 + 2, True),
	# 	('photo', 73 * 10 + 2, 91 * 10 + 2, True)
	# ]

	# for subject, start, end, ignore_validation in subject_partitioning:
	model_scores, _, _, _ = get_model_scores(sys.argv[1:-5], sys.argv[-5], sys.argv[-4])

	flattened_scores = [a for x in [[score[1] for score in scores] for scores in model_scores] for a in x]
	# model_scores = np.array(list(filter(bool, [[float(score[1]) for score in scores] for scores in model_scores])))

	num_topics_per_subject = 6
	indices = []
	with open(sys.argv[-3]) as indices_file:
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

	netl_scores = np.array(json.load(open(sys.argv[-2])))[indices]
	netl_scores = [{netl_label_score[0].replace('_', ' ').split('(')[0].strip(): float(netl_label_score[1]) for netl_label_score in netl_topic_scores} for netl_topic_scores in netl_scores]

	model_scores = [[(label_score[0], label_score[1], netl_scores[topic_idx][label_score[0]]) for label_score in model_scores[topic_idx]] for topic_idx in range(len(model_scores))]
	netl_scores = sorted(list(set(list(map(lambda x: x[2], [(label_score[0], label_score[1], netl_scores[topic_idx][label_score[0]]) for topic_idx in range(len(model_scores)) for label_score in model_scores[topic_idx]])))))

	avg_ratings = []
	for netl_score in netl_scores:
		filtered_scores = [[label_score[1] for label_score in model_scores[topic_idx] if label_score[2] >= netl_score] for topic_idx in range(len(model_scores))]
		filtered_scores = np.array(list(filter(bool, filtered_scores)))
		num_remaining_labels = len([score for scores in filtered_scores for score in scores])
		avg_ratings.append((netl_score, np.mean([np.mean(scores) for scores in filtered_scores]), num_remaining_labels))

		# np.mean([np.mean(scores) for scores in np.array(list(filter(bool, [[label_score[1] for label_score in model_scores[topic_idx] if label_score[2] >= netl_score] for topic_idx in range(len(model_scores))])))])

	plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
	font = {'size': 16}
	matplotlib.rc('font', **font)
	plt.xlabel('NETL score threshold')
	plt.ylabel('All-Average Rating')
	plt.hlines(2.417, np.min(netl_scores), np.max(netl_scores), label='NETL (unsupervised)')
	plt.hlines(2.391, np.min(netl_scores), np.max(netl_scores), label='NETL (supervised)')
	plt.plot(list(map(lambda x: x[0], avg_ratings)), list(map(lambda x: x[1], avg_ratings)))
	plt.savefig(sys.argv[-1])


if __name__ == '__main__':
	main()
