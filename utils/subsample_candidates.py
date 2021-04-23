import sys


def main():
	if len(sys.argv) != 4:
		print("Usage: " + sys.argv[0] + " <candidates input file> <candidates out file>"
										" <# of remaining labels>")
		exit(0)

	num_labels = int(sys.argv[3])

	with open(sys.argv[1]) as in_file, open(sys.argv[2], 'w') as out_file:
		for line in in_file:
			if line.strip() == '':
				continue

			labels = line.strip().split(' ')
			if num_labels >= len(labels):
				print("Warning: There are not enough candidates in the input file")

			out_file.write(' '.join(labels[:num_labels]) + '\n')


if __name__ == '__main__':
	main()
