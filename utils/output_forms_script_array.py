import sys
import json


def main():
	if len(sys.argv) != 2:
		print("Usage: " + sys.argv[0] + " <survey file (output of prepare_survey_questions.py)")
		exit(0)

	survey_array = []

	with open(sys.argv[1]) as survey_file:
		for line in survey_file:
			survey_array.extend(json.loads(line))

	print(survey_array)
	import pdb
	pdb.set_trace()


if __name__ == '__main__':
	main()
