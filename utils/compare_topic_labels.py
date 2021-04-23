import sys


def main():
	if len(sys.argv) < 3 or len(sys.argv) > 5:
		print("Usage: " + sys.argv[0] + " <topic labels file #1> <topic labels file #2> [<topics file> <num common labels>]")
		exit(-1)

	topic_index = 0
	count = [0] * 11

	topics = None
	if len(sys.argv) >= 4:
		topics = []
		with open(sys.argv[3]) as topics_file:
			# skip header line
			topics_file.readline()
			for line in topics_file:
				tokens = line.strip().split(',')
				topics.append(tokens[2:])

	with open(sys.argv[1]) as file1, open(sys.argv[2]) as file2:

		while True:
			labels1 = []
			labels2 = []
			line1 = file1.readline()
			line2 = file2.readline()

			# done
			if line1.strip() == "":
				break

			# 3 topic labels for every topic
			for i in range(10):
				labels1.append(file1.readline().strip())
				labels2.append(file2.readline().strip())

			# process topic labels
			num_common_labels = 0
			for label1 in labels1:
				for label2 in labels2:
					if label1 == label2:
						num_common_labels += 1

			count[num_common_labels] += 1

			if num_common_labels == int(sys.argv[4]):
				print(topics[topic_index])
				print(labels1)
				print(labels2)
				print()
			#	count += 1

			topic_index += 1
			#print(count)

			# one blank line left after each topic
			file1.readline()
			file2.readline()

	print("Total number of cases: " + str(count))
	print(sum(count))


if __name__ == '__main__':
	main()
