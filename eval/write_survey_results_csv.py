import sys
import numpy as np
from survey_utils import get_model_scores


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
		print("Usage: " + sys.argv[0] + " [<survey csv responses>]+ <topics csv> <output csv>")
		exit(0)

	models = [
		('NETL unsupervised', 'summarization/output_unsupervised_netl_all_before_v2'),
		('NETL supervised', 'summarization/output_supervised_netl_all_before_v2'),
		('BART-TL-all', 'summarization/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/train2_25.hypo'),
		('BART-TL-all (unsupervised)', 'summarization/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/netl_unsupervised_train2_25.hypo'),
		('BART-TL-all (supervised)', 'summarization/models/bart_finetuned_terms_labels_sentences_ngrams_nps_v2/summaries/netl_supervised_train2_25.hypo'),
		('BART-TL-ngram', 'summarization/models/bart_finetuned_terms_labels_ngrams_v5/summaries/train2_25.hypo'),
		('BART-TL-ngram (unsupervised)', 'summarization/models/bart_finetuned_terms_labels_ngrams_v5/summaries/netl_unsupervised_train2_25.hypo'),
		('BART-TL-ngram (supervised)', 'summarization/models/bart_finetuned_terms_labels_ngrams_v5/summaries/netl_supervised_train2_25.hypo')
	]

	subject_partitioning = [
		('all', 2, None, False),
		('english', 0 * 10 + 2, 18 * 10 + 2, True),
		('biology', 18 * 10 + 2, 37 * 10 + 2, True),
		('economics', 37 * 10 + 2, 55 * 10 + 2, True),
		('law', 55 * 10 + 2, 73 * 10 + 2, True),
		('photo', 73 * 10 + 2, 91 * 10 + 2, True)
	]

	with open(sys.argv[-1], 'w') as output_csv:
		# write the header
		output_csv.write('Model;Top-1 Avg.;Top-3 Avg.;Top-5 Avg.;All Avg.;nDCG-1;nDCG-3;nDCG-5;Top-1 Avg.;Top-3 Avg.;Top-5 Avg.;All Avg.;nDCG-1;nDCG-3;nDCG-5;Top-1 Avg.;Top-3 Avg.;Top-5 Avg.;All Avg.;nDCG-1;nDCG-3;nDCG-5;Top-1 Avg.;Top-3 Avg.;Top-5 Avg.;All Avg.;nDCG-1;nDCG-3;nDCG-5;Top-1 Avg.;Top-3 Avg.;Top-5 Avg.;All Avg.;nDCG-1;nDCG-3;nDCG-5;Top-1 Avg.;Top-3 Avg.;Top-5 Avg.;All Avg.;nDCG-1;nDCG-3;nDCG-5\n')
		for model_name, model_file in models:
			output_csv.write(model_name)
			for subject, start, end, ignore_validation in subject_partitioning:
				model_scores, _, _, _ = get_model_scores(sys.argv[1:-2], sys.argv[-2], model_file, start, end, ignore_validation)

				model_scores = np.array(list(filter(bool, [[float(score[1]) for score in scores] for scores in model_scores])))

				output_csv.write(";%.3f" % np.mean([scores[:1] for scores in model_scores if len(scores) >= 1]))
				output_csv.write(";%.3f" % np.mean([scores[:3] for scores in model_scores if len(scores) >= 3]))
				output_csv.write(";%.3f" % np.mean([scores[:5] for scores in model_scores if len(scores) >= 5]))
				output_csv.write(";%.3f" % np.mean([np.mean(scores) for scores in model_scores]))
				output_csv.write(";%.3f" % np.mean([ndcg(scores, 1) for scores in model_scores]))
				output_csv.write(";%.3f" % np.mean([ndcg(scores, 3) for scores in model_scores]))
				output_csv.write(";%.3f" % np.mean([ndcg(scores, 5) for scores in model_scores]))

			output_csv.write('\n')


if __name__ == '__main__':
	main()
