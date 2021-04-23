import sys

from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


def main():
	if len(sys.argv) != 3:
		print("Usage: " + sys.argv[0] + " <glove input file> <word2vec output file>")
		exit(0)

	#glove_file = datapath(sys.argv[1])
	glove_file = sys.argv[1]
	#word2vec_file = get_tmpfile(sys.argv[2])
	word2vec_file = sys.argv[2]

	glove2word2vec(glove_file, word2vec_file)


if __name__ == '__main__':
	main()
