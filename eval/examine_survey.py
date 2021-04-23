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
		print("Usage: " + sys.argv[0] + " [<survey csv responses>]+ <topics csv> <model hypos>")
		exit(0)

	subject_partitioning = [
		('all', 2, None, False),
		('english', 0 * 10 + 2, 18 * 10 + 2, True),
		('biology', 18 * 10 + 2, 37 * 10 + 2, True),
		('economics', 37 * 10 + 2, 55 * 10 + 2, True),
		('law', 55 * 10 + 2, 73 * 10 + 2, True),
		('photo', 73 * 10 + 2, 91 * 10 + 2, True)
	]

	for subject, start, end, ignore_validation in subject_partitioning:
		model_scores, sufficiently_annotated_labels, insufficiently_annotated_labels, total_annotated_labels = get_model_scores(sys.argv[1:-2], sys.argv[-2], sys.argv[-1], start, end, ignore_validation)

		model_scores = np.array(list(filter(bool, [[float(score[1]) for score in scores] for scores in model_scores])))

		print(subject.upper())
		print("--------")
		print("Top-1 Average Rating: " + str(np.mean([scores[:1] for scores in model_scores if len(scores) >= 1])))
		print("Top-3 Average Rating: " + str(np.mean([scores[:3] for scores in model_scores if len(scores) >= 3])))
		print("Top-5 Average Rating: " + str(np.mean([scores[:5] for scores in model_scores if len(scores) >= 5])))
		print("All-labels Average Rating: " + str(np.mean([np.mean(scores) for scores in model_scores])))
		print("nDCG-1: " + str(np.mean([ndcg(scores, 1) for scores in model_scores])))
		print("nDCG-3: " + str(np.mean([ndcg(scores, 3) for scores in model_scores])))
		print("nDCG-5: " + str(np.mean([ndcg(scores, 5) for scores in model_scores])))
		print("")

		total_labels = sum([len(scores) for scores in model_scores])
		print("Total number of labels: " + str(total_labels))

		print("============")
		print("Sufficiently annotated labels: " + str(sufficiently_annotated_labels))
		print("Insufficiently annotated labels: " + str(insufficiently_annotated_labels))
		print("Total annotated labels: " + str(total_annotated_labels))
		print("")


if __name__ == '__main__':
	main()
