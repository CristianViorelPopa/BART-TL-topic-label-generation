import sys
import os
import numpy as np
from survey_utils import get_model_scores

import matplotlib
import matplotlib.pyplot as plt


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
	if len(sys.argv) < 4:
		print("Usage: " + sys.argv[0] + " [<survey csv responses>]+ <topics csv> <output file>")
		exit(0)

	models = [
		('NETL unsupervised', 'summarization/output_unsupervised_netl_all_before_v2', 'o-', 'black'),
		('NETL supervised', 'summarization/output_supervised_netl_all_before_v2', 'o-', 'dimgray'),
		# ('BART-TL-all', 'summarization/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/train2_25.hypo'),
		('BART-TL-all (unsupervised)', 'summarization/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/netl_unsupervised_train2_25.hypo', '.-.', 'black'),
		('BART-TL-all (supervised)', 'summarization/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/netl_supervised_train2_25.hypo', '.-.', 'dimgray'),
		# ('BART-TL-ngram', 'summarization/models/bart_finetuned_terms_labels_ngrams_v5/summaries/train2_25.hypo'),
		('BART-TL-ngram (unsupervised)', 'summarization/models/bart_finetuned_terms_labels_ngrams_v5/summaries/netl_unsupervised_train2_25.hypo', 'x:', 'black'),
		('BART-TL-ngram (supervised)', 'summarization/models/bart_finetuned_terms_labels_ngrams_v5/summaries/netl_supervised_train2_25.hypo', 'x:', 'dimgray')
	]

	subject_partitioning = [
		('all', 2, None, False),
		# ('english', 0 * 10 + 2, 18 * 10 + 2, True),
		# ('biology', 18 * 10 + 2, 37 * 10 + 2, True),
		# ('economics', 37 * 10 + 2, 55 * 10 + 2, True),
		# ('law', 55 * 10 + 2, 73 * 10 + 2, True),
		# ('photo', 73 * 10 + 2, 91 * 10 + 2, True)
	]

	limit = 10

	plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	font = {'size': 15}
	matplotlib.rc('font', **font)

	plt.ylabel('Average Rating')
	plt.xlabel('Number of top labels')
	plt.xticks(list(range(1, limit + 1)))
	plt.xlim(1, limit)
	plt.ylim(2.0, 3.0)

	for model_name, model_file, linestyle, color in models:
		for subject, start, end, ignore_validation in subject_partitioning:
			model_scores, _, _, _ = get_model_scores(sys.argv[1:-2], sys.argv[-2], model_file, start, end, ignore_validation)

			model_scores = np.array(list(filter(bool, [[float(score[1]) for score in scores] for scores in model_scores])))

			ratings = []
			for i in range(1, limit + 1):
				ratings.append(np.mean([scores[:i] for scores in model_scores if len(scores) >= i]))

			plt.plot(list(range(1, limit + 1)), ratings, linestyle, linewidth=2, color=color)

	plt.legend(list(map(lambda x: x[0], models)))
	plt.savefig(sys.argv[-1])


if __name__ == '__main__':
	main()
