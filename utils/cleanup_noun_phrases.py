import sys
import re
import string


def process_np(np):
	exclude = string.punctuation
	include = ['\'', ',', '-', '_']
	exclude = list(filter(lambda c: c not in include, exclude))
	for c in exclude:
		np = np.replace(c, '')
	np = np.replace('_', ' ')
	np = re.sub(' +', ' ', np)
	np = np.strip().replace(' ', '_')
	return np


def main():
	if len(sys.argv) != 2:
		print("Usage: " + sys.argv[0] + " <noun phrases txt file>")
		exit(-1)

	noun_phrases = []
	with open(sys.argv[1]) as in_file:
		for line in in_file:
			nps = line.strip().split(' ')
			nps = [process_np(np) for np in nps]
			noun_phrases.append(' '.join(nps))

	with open(sys.argv[1], 'w') as out_file:
		out_file.write('\n'.join(noun_phrases))


if __name__ == '__main__':
	main()
