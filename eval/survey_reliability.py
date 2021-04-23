import sys
import numpy as np
from survey_utils import get_model_stddev


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
	if len(sys.argv) < 3:
		print("Usage: " + sys.argv[0] + " [<survey csv responses>]+ <topics csv>")
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

	for model_name, model_file, linestyle, color in models:
		for subject, start, end, ignore_validation in subject_partitioning:
			model_scores, _, _, _ = get_model_stddev(sys.argv[1:-1], sys.argv[-1], model_file, start, end, ignore_validation)

			flattened_scores = [a for x in [[score[1] for score in scores] for scores in model_scores] for a in x]
			print(model_name + ": " + str(np.mean(flattened_scores)))

	# NETL unsupervised: 0.43027801572488733
	# NETL supervised: 0.435986272208545
	# BART-TL-all (unsupervised): 0.4399068568012093
	# BART-TL-all (supervised): 0.4399068568012093
	# BART-TL-ngram (unsupervised): 0.421469803459316
	# BART-TL-ngram (supervised): 0.42146980345931606


if __name__ == '__main__':
	main()
