import sys


def main():
	if len(sys.argv) < 3 or len(sys.argv) > 5:
		print("Usage: " + sys.argv[0] + " <topic candidates file #1> <topic candidates file #2> [<topics file> <num common candidates>]")
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

		for topic_index in range(len(topics)):
			line1 = file1.readline()
			line2 = file2.readline()

			# done
			if line1.strip() == "":
				break

			candidates1 = line1.strip().split(' ')[:10]
			candidates2 = line2.strip().split(' ')[:10]

			# process topic candidates
			num_common_candidates = 0
			for candidate1 in candidates1:
				for candidate2 in candidates2:
					if candidate1 == candidate2:
						num_common_candidates += 1

			count[num_common_candidates] += 1

			if num_common_candidates == int(sys.argv[4]):
				print(topics[topic_index])
				print(candidates1)
				print(candidates2)
				print()
			#	count += 1

	print("Total number of cases: " + str(count))
	print(sum(count))


if __name__ == '__main__':
	main()
