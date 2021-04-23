import argparse

import numpy as np
from bert_score import BERTScorer


def main():
	parser = argparse.ArgumentParser(
		description='Scoring of model based on BERTScore against gold NETL labels'
	)
	parser.add_argument('--preds', '-p', required=True, type=str,
						help='File containing predictions on the NETL topics')
	parser.add_argument('--golds', '-g', required=True, type=str,
						help='File containing gold-standard labels on the NETL topics')
	parser.add_argument('--ignored', '-i', required=False, type=str,
						help='Ignored NETL topics in the gold standards')
	parser.add_argument('--num-labels', '-n', required=False, type=int,
						help='How many labels to consider when computing metrics')

	args = parser.parse_args()

	print('WARNING: This script does not compute the same results as the ones in the Automatic Generation of Topic Labels, 2020 paper. For that, use bert_score/compute_bertscore.py')

	if args.ignored:
		ignored_indices = [int(line.strip()) for line in open(args.ignored)]
	else:
		ignored_indices = []
	preds = [list(map(lambda label: label.replace('_', ' '), line.strip().split(' '))) for idx, line
			 in enumerate(open(args.preds).read().split('\n')) if ((idx + 1) not in ignored_indices and line.strip() != '')]
	golds = [line.strip().split(',') for line in open(args.golds)]
	scorer = BERTScorer(lang='en')

	topic_recalls = []
	topic_precisions = []
	topic_f1s = []

	for topic_idx in range(len(preds)):
		topic_labels_recalls = []
		topic_labels_precisions = []
		topic_labels_f1s = []

		for pred_label in preds[topic_idx]:
			recalls = []
			precisions = []
			f1s = []

			for gold_label in golds[topic_idx]:
				recall, precision, f1 = scorer.score([pred_label], [gold_label])
				recalls.append(recall.item())
				precisions.append(precision.item())
				f1s.append(f1.item())

			topic_labels_recalls.append(np.max(recalls))
			topic_labels_precisions.append(np.max(precisions))
			topic_labels_f1s.append(np.max(f1s))

		topic_recalls.append(topic_labels_recalls)
		topic_precisions.append(topic_labels_precisions)
		topic_f1s.append(topic_labels_f1s)

		print('Progress: ' + str(topic_idx + 1) + '/' + str(len(preds)) + ' ' * 10, end='\r')

	print()
	print()

	if args.num_labels:
		print('Average recall: ' + str(np.mean([np.mean(l[:args.num_labels]) for l in topic_recalls])))
		print('Average precision: ' + str(np.mean([np.mean(l[:args.num_labels]) for l in topic_precisions])))
		print('Average F1: ' + str(np.mean([np.mean(l[:args.num_labels]) for l in topic_f1s])))

	else:
		for i in [1, 3, 5]:
			print('Top-' + str(i))
			print('Average recall: ' + str(np.mean([np.mean(l[:i]) for l in topic_recalls])))
			print('Average precision: ' + str(np.mean([np.mean(l[:i]) for l in topic_precisions])))
			print('Average F1: ' + str(np.mean([np.mean(l[:i]) for l in topic_f1s])))
			print()


if __name__ == '__main__':
	main()
