import sys
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

num_candidates = 46


invalid_tokens = stop_words + \
				 ['we', 'once', 'that', 'you', 'i', 'an', 'yet', 'whom', 'was', 'being',
				  'my', 'our', 'their', 'he', 'she', 'us', 'them', 'him', 'her', 'twice',
				  'had', 'the', 'your', 'his', 'would', 'mr.', 'sir', 'not', 'ok', 'also',
				  'mr', 'one', 'no', 'while', 'as']
invalid_tokens = list(set(invalid_tokens))

forbidden_tokens = ['pussy', 'dicks', 'cunt', 'fuck', 'hentai', 'bdsm', 'masturbation',
					'nigger', 'faggot_(slang)', 'threesome', 'penis', 'dildo', 'slut',
					'shemale', 'lesbian', 'bukkake', 'vagina', 'playboy', 'fart', 'orgasm',
					'bullshit']


def main():
	if len(sys.argv) != 5:
		print("Usage: " + sys.argv[0] + " <topics csv file> <candidates file 1> <candidates file 2>"
										" <output survey file>")
		exit(0)

	with open(sys.argv[4], 'w') as out_file:
		with open(sys.argv[1]) as topics_file:
			# skip header line
			topics_file.readline()
			topics = []
			for line in topics_file:
				tokens = line.strip().split(',')
				topics.append(tokens[2:])

		topic_index = 0
		with open(sys.argv[2]) as cand_file1, open(sys.argv[3]) as cand_file2:
			while topic_index < len(topics):
				line1 = cand_file1.readline()
				line2 = cand_file2.readline()

				# done
				if line1.strip() == "":
					break

				cands1 = line1.strip().split()
				cands2 = line2.strip().split()

				topic_cands = []
				idx1 = -1
				idx2 = -1
				while len(topic_cands) < num_candidates:
					while True:
						idx1 += 1
						if cands1[idx1] in topic_cands or cands1[idx1] in invalid_tokens \
								or cands1[idx1] in forbidden_tokens:
							continue
						topic_cands.append(cands1[idx1])
						break
					while True:
						idx2 += 1
						if cands2[idx2] in topic_cands or cands2[idx2] in invalid_tokens \
								or cands2[idx2] in forbidden_tokens:
							continue
						topic_cands.append(cands2[idx2])
						break

				# format of the output file:
				# <TOP 10 TERMS FOR TOPIC 1 (space-separated)>
				# <COMBINED CANDIDATES FOR TOPIC 1 (space-separated)>
				# ...
				# <TOP 10 TERMS FOR TOPIC n>
				# <COMBINED CANDIDATES FOR TOPIC n>
				# (no empty lines)
				out_file.write(' '.join(topics[topic_index]) + '\n')
				out_file.write(' '.join(topic_cands) + '\n')

				topic_index += 1


if __name__ == '__main__':
	main()
