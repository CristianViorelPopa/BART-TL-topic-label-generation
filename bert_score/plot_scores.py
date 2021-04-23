import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
	parser = argparse.ArgumentParser(
		description='Plot BERTScore evolution from README'
	)
	parser.add_argument('--output', '-o', required=True, type=str,
						help='Name of the output directory')
	args = parser.parse_args()

	included_models = {
		# 'Baseline-2': 'Baseline 2',
		# 'Baseline-3': 'B',
		# 'ds_wiki_tfidf-topics_bhatia': '',
		# 'ds_wiki_tfidf-topics_bhatia_tfidf': '',
		# 'ds_wiki_sent-topics_bhatia': '',
		'ds_wiki_sent-topics_bhatia_tfidf': '',
		# 'bart_finetuned_terms_labels_ngrams_v5': '',
		# 'bart_finetuned_terms_labels_nps_v3': '',
		# 'bart_finetuned_terms_labels_sentences_v5': '',
		# 'bart_finetuned_terms_labels_v16': '',
		# 'bart_finetuned_areej_wiki_sent_v1': '',
		'bart_finetuned_areej_wiki_sent_v2': 'Large BART',
		# 'bart_finetuned_areej_wiki_tfidf_v1': '',
		# 'bart_base_areej_wiki_sent_v1': 'Base BART',
		# 'bart_base_areej_wiki_sent_v2': 'Beam Search',
		# 'bart_base_areej_wiki_sent_v3': 'Base BART',
		'prophetnet_large_uncased_areej_wiki_sent_v1': 'ProphetNet',
		# 'bart_base_areej_wiki_sent_v4': '',
		'bart_base_areej_wiki_sent_v5': 'Final base BART',
		'bart_base_areej_wiki_sent_v6': 'Random sampling n=20',
		# 'bart_base_areej_wiki_sent_v7': 'Random sampling n=10',
		'bart_base_areej_wiki_sent_v8': 'Added 20% n-grams',
		'bart_base_areej_wiki_sent_v9': 'Added 50% n-grams',
	}

	scores = {}
	with open(os.path.join(os.path.dirname(__file__), 'results.csv')) as scores_csv:
		# header
		scores_csv.readline()

		for line in scores_csv:
			tokens = line.strip().split(',')
			recalls = np.array([[1, float(tokens[1])], [3, float(tokens[4])], [5, float(tokens[7])]])
			precisions = np.array([[1, float(tokens[2])], [3, float(tokens[5])], [5, float(tokens[8])]])
			f1s = np.array([[1, float(tokens[3])], [3, float(tokens[6])], [5, float(tokens[9])]])

			if tokens[0] not in included_models:
				continue

			scores[tokens[0]] = {}
			scores[tokens[0]]['Recall'] = recalls
			scores[tokens[0]]['Precision'] = precisions
			scores[tokens[0]]['F1'] = f1s

	for metric in ['Recall', 'Precision', 'F1']:
		plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
		font = {'size': 12}
		matplotlib.rc('font', **font)

		plt.ylabel('BERTScore ' + metric)
		plt.xlabel('Number of top labels')

		for model_name, stats in scores.items():
			if included_models[model_name] == '':
				plt.plot(stats[metric][:, 0], stats[metric][:, 1], '-', label=model_name)
			else:
				plt.plot(stats[metric][:, 0], stats[metric][:, 1], '-', label=included_models[model_name])

		plt.legend(loc="upper left", framealpha=0.4)
		plt.savefig(os.path.join(args.output, metric + '.png'))


if __name__ == '__main__':
	main()
