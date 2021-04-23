import spacy

sp = spacy.load('en_core_web_sm')


def main():
	topics_file = '../bart-tl/dataset_fairseq/train.source'
	topics = [line.strip() for line in open(topics_file)]

	all_pos_dict = {'ADJ': 0, 'ADP': 0, 'PUNCT': 0, 'ADV': 0, 'AUX': 0, 'SYM': 0, 'INTJ': 0,
					'CONJ': 0, 'X': 0, 'NOUN': 0, 'DET': 0, 'PROPN': 0, 'NUM': 0, 'VERB': 0,
					'PART': 0, 'PRON': 0, 'SCONJ': 0, 'CCONJ': 0}
	total_tokens = 0
	total_topics_len = 0

	for idx, topic in enumerate(topics):
		sen = sp(topic)
		for token in sen:
			all_pos_dict[token.pos_] += 1
			total_tokens += 1
		total_topics_len += len(sen)

		print('Progress: ' + str(idx) + '/' + str(len(topics)), end='\r')
	print()
	print()

	for pos, count in all_pos_dict.items():
		if count < 0.01:
			continue

		print(pos + ' - %.2f' % (count / total_tokens * 100))

	print(total_topics_len)

	# NETL distribution
	# NOUN - 47%
	# PROPER NOUN - 42%
	# ADJECTIVE - 6%
	# VERB - 4%
	# OTHERS - <1%

	# original cleaned ds_wiki_sent distribution:
	# NOUN - 26%
	# PROPER NOUN - 43%
	# ADJECTIVE - 10%
	# VERB - 15%
	# ADVERB - 3%
	# OTHERS - <3%
	# avg. tokens per topic: 29.06

	# redistributed cleaned ds_wiki_sent distribution (not aggressive):
	# NOUN - 30%
	# PROPER NOUN - 52%
	# ADJECTIVE - 7%
	# VERB - 10%
	# OTHERS - <1%
	# avg. tokens per topic: 20.95

	# redistributed cleaned ds_wiki_sent distribution (aggressive):
	# NOUN - 36%
	# PROPER NOUN - 44%
	# ADJECTIVE - 9%
	# VERB - 10%
	# OTHERS - <1%
	# avg. tokens per topic: 12.75


if __name__ == '__main__':
	main()
