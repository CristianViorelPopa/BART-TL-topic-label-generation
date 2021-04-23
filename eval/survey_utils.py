import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')  # run once
stop_words = stopwords.words('english')

invalid_tokens = stop_words + \
				 ['we', 'once', 'that', 'you', 'i', 'an', 'yet', 'whom', 'was', 'being',
				  'my', 'our', 'their', 'he', 'she', 'us', 'them', 'him', 'her', 'twice',
				  'had', 'the', 'your', 'his', 'would', 'mr.', 'sir', 'not', 'ok', 'also',
				  'mr', 'one', 'no', 'while', 'as']
invalid_tokens = list(set(invalid_tokens))


num_minimum_annotators = 2
total_annotated_labels = 0
insufficiently_annotated_labels = 0
sufficiently_annotated_labels = 0


def preprocess_label(label):
	return label.split('(')[0].replace('_', ' ').strip()


def get_model_scores(survey_csv_files, topics_csv_file, model_hypos_file, start=2, end=None, ignore_validation=True):
	answer_values = {
		'Not relevant at all': 0,
		'A bit relevant': 1,
		# 'Somehow relevant': 2,
		# 'Relevant': 3,
		'Relevant': 2,
		# 'Very relevant': 4
		'Very relevant': 3
	}

	topic_labels_scores = {}
	target_topic_terms = []
	for csv_file in survey_csv_files:
		# if csv_file == sys.argv[2]:
		#     import pdb
		#     pdb.set_trace()

		responses = pd.read_csv(csv_file)

		invalid_tokens_scores = []
		for _ in range(responses.shape[0]):
			invalid_tokens_scores.append([])

		# skip timestamp and nickname columns
		if end is None:
			end = len(responses.columns)

		for question in responses.columns[start:end]:
			# topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:-1].split(', '))

			# bugged question
			if len(question.split('\n')) == 1:
				continue
				topic_terms = ' '.join(question
									   .split('Select how relevant the terms below are for describing the following word sequence:')[1]
									   .split('Here are two sample paragraphs')[0]
									   .split(', '))

			else:
				topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:].split(', '))

			label = question.split('[')[1].split(']')[0]
			if topic_terms not in topic_labels_scores:
				topic_labels_scores[topic_terms] = {}

			answers = responses[question].values

			if label in invalid_tokens:
				for idx in range(len(answers)):
					if not pd.isnull(answers[idx]):
						invalid_tokens_scores[idx].append(answer_values[answers[idx]])

			answers = answers[~pd.isnull(answers)]
			answers = [answer_values[answer] for answer in answers]

			if len(answers) > 0:
				# topic_labels_scores[topic_terms][label] = np.mean(answers)
				if label not in invalid_tokens:
					# if len(answers) < num_minimum_annotators:
					# 	insufficiently_annotated_labels += 1
					# else:
					if label not in topic_labels_scores[topic_terms]:
						topic_labels_scores[topic_terms][label] = []
						# sufficiently_annotated_labels += 1
					topic_labels_scores[topic_terms][label].extend(answers)
				# sufficiently_annotated_labels += 1

			else:
				if label not in topic_labels_scores[topic_terms] and label not in invalid_tokens:
					topic_labels_scores[topic_terms][label] = []
				# topic_labels_scores[topic_terms][label] = -1.0
				# print(topic_terms)

			# only account for the topics in the last survey
			if csv_file == survey_csv_files[-1]:
				target_topic_terms.append(topic_terms)

		valid_answers = [sum(np.array(user_answers) <= 1) >= 0.25 * len(user_answers)
						 for user_answers in invalid_tokens_scores]
		if not ignore_validation and not np.all(valid_answers):
			print('WARNING: Not all answers passed the validation!')
			import pdb
			pdb.set_trace()

	# topic_labels_scores['voice direction speed signal button format technique light command procedure']['accelerometer'] = 3.0
	# topic_labels_scores['voice direction speed signal button format technique light command procedure']['addition'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['observation'] = 2.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['metabolism'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['circadian rhythm'] = 3.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['contradiction'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['proteus'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['hypnotism'] = 3.5
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['subjective experience'] = 4.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['vibration'] = 1.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['unconsciousness'] = 4.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['signal processing'] = 1.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['principle'] = 4.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['formalism'] = 4.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['argument ad fallacy'] = 2.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['interrogation'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['confidentiality'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['prosecutor'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['apprehension'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['public enquiry'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['crime and punishment'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['observation'] = 1.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['propos inquiry'] = 1.0
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['head of state'] = 3.5
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['freedom of speech'] = 4.0
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['imperialism'] = 4.0
	# topic_labels_scores['cloud virus amazon trail frankenstein addicted deployed crumb maze othello']['viruses'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['passenger vehicle'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['passenger car'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['point of sale'] = 1.0

	global total_annotated_labels
	global insufficiently_annotated_labels
	global sufficiently_annotated_labels

	for _, labels_scores in topic_labels_scores.items():
		for label in labels_scores.keys():
			if len(labels_scores[label]) >= num_minimum_annotators:
				labels_scores[label] = np.mean(labels_scores[label])
				sufficiently_annotated_labels += 1
				total_annotated_labels += 1
			elif labels_scores[label]:
				labels_scores[label] = -1.0
				insufficiently_annotated_labels += 1
				total_annotated_labels += 1
			else:
				labels_scores[label] = -1.0
				total_annotated_labels += 1

	model_scores = []
	with open(topics_csv_file) as topics_csv, open(model_hypos_file) as model_hypos:
		# ignore header
		topics_csv.readline()

		while True:
			topic_terms = topics_csv.readline()
			if topic_terms.strip() == '':
				break
			topic_terms = ' '.join(topic_terms.strip().split(',')[2:])
			labels = [preprocess_label(label)
					  for label in model_hypos.readline().strip().split(' ')]

			# the topic was not included in the survey
			# if topic_terms not in topic_labels_scores:
			#     continue

			# the topic is not in the last survey
			if topic_terms not in target_topic_terms:
				continue

			label_scores = []

			for label in labels:
				if label not in topic_labels_scores[topic_terms]:
					# this is no longer applicable, since every label that doesn't exist has a score
					# of -1 and is treated correctly
					print('Not all labels of the model were included in the survey!')
					continue
				# label_scores.append(topic_labels_scores[topic_terms][label])
				label_scores.append((label, topic_labels_scores[topic_terms][label]))

			# label_scores = list(filter(lambda x: x != -1.0, label_scores))
			label_scores = list(filter(lambda x: x[1] != -1.0, label_scores))

			for label in labels:
				if label in invalid_tokens:
					label_scores.append((label, 0.0))

			if len(label_scores) > 0:
				model_scores.append(label_scores)
			else:
				model_scores.append([])

	model_scores = np.array(model_scores)
	return model_scores, sufficiently_annotated_labels, insufficiently_annotated_labels, total_annotated_labels


def get_model_scores_with_topics(survey_csv_files, topics_csv_file, model_hypos_file, start=2, end=None, ignore_validation=True):
	answer_values = {
		'Not relevant at all': 0,
		'A bit relevant': 1,
		# 'Somehow relevant': 2,
		# 'Relevant': 3,
		'Relevant': 2,
		# 'Very relevant': 4
		'Very relevant': 3
	}

	topic_labels_scores = {}
	target_topic_terms = []
	for csv_file in survey_csv_files:
		# if csv_file == sys.argv[2]:
		#     import pdb
		#     pdb.set_trace()

		responses = pd.read_csv(csv_file)

		invalid_tokens_scores = []
		for _ in range(responses.shape[0]):
			invalid_tokens_scores.append([])

		# skip timestamp and nickname columns
		if end is None:
			end = len(responses.columns)

		for question in responses.columns[start:end]:
			# topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:-1].split(', '))

			# bugged question
			if len(question.split('\n')) == 1:
				continue
				topic_terms = ' '.join(question
									   .split('Select how relevant the terms below are for describing the following word sequence:')[1]
									   .split('Here are two sample paragraphs')[0]
									   .split(', '))

			else:
				topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:].split(', '))

			label = question.split('[')[1].split(']')[0]
			if topic_terms not in topic_labels_scores:
				topic_labels_scores[topic_terms] = {}

			answers = responses[question].values

			if label in invalid_tokens:
				for idx in range(len(answers)):
					if not pd.isnull(answers[idx]):
						invalid_tokens_scores[idx].append(answer_values[answers[idx]])

			answers = answers[~pd.isnull(answers)]
			answers = [answer_values[answer] for answer in answers]

			if len(answers) > 0:
				# topic_labels_scores[topic_terms][label] = np.mean(answers)
				if label not in invalid_tokens:
					# if len(answers) < num_minimum_annotators:
					# 	insufficiently_annotated_labels += 1
					# else:
					if label not in topic_labels_scores[topic_terms]:
						topic_labels_scores[topic_terms][label] = []
						# sufficiently_annotated_labels += 1
					topic_labels_scores[topic_terms][label].extend(answers)
				# sufficiently_annotated_labels += 1

			else:
				if label not in topic_labels_scores[topic_terms] and label not in invalid_tokens:
					topic_labels_scores[topic_terms][label] = []
				# topic_labels_scores[topic_terms][label] = -1.0
				# print(topic_terms)

			# only account for the topics in the last survey
			if csv_file == survey_csv_files[-1]:
				target_topic_terms.append(topic_terms)

		valid_answers = [sum(np.array(user_answers) <= 1) >= 0.25 * len(user_answers)
						 for user_answers in invalid_tokens_scores]
		if not ignore_validation and not np.all(valid_answers):
			print('WARNING: Not all answers passed the validation!')
			import pdb
			pdb.set_trace()

	# topic_labels_scores['voice direction speed signal button format technique light command procedure']['accelerometer'] = 3.0
	# topic_labels_scores['voice direction speed signal button format technique light command procedure']['addition'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['observation'] = 2.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['metabolism'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['circadian rhythm'] = 3.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['contradiction'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['proteus'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['hypnotism'] = 3.5
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['subjective experience'] = 4.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['vibration'] = 1.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['unconsciousness'] = 4.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['signal processing'] = 1.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['principle'] = 4.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['formalism'] = 4.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['argument ad fallacy'] = 2.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['interrogation'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['confidentiality'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['prosecutor'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['apprehension'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['public enquiry'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['crime and punishment'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['observation'] = 1.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['propos inquiry'] = 1.0
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['head of state'] = 3.5
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['freedom of speech'] = 4.0
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['imperialism'] = 4.0
	# topic_labels_scores['cloud virus amazon trail frankenstein addicted deployed crumb maze othello']['viruses'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['passenger vehicle'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['passenger car'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['point of sale'] = 1.0

	global total_annotated_labels
	global insufficiently_annotated_labels
	global sufficiently_annotated_labels

	for _, labels_scores in topic_labels_scores.items():
		for label in labels_scores.keys():
			if len(labels_scores[label]) >= num_minimum_annotators:
				labels_scores[label] = np.mean(labels_scores[label])
				sufficiently_annotated_labels += 1
				total_annotated_labels += 1
			elif labels_scores[label]:
				labels_scores[label] = -1.0
				insufficiently_annotated_labels += 1
				total_annotated_labels += 1
			else:
				labels_scores[label] = -1.0
				total_annotated_labels += 1

	model_scores = {}
	with open(topics_csv_file) as topics_csv, open(model_hypos_file) as model_hypos:
		# ignore header
		topics_csv.readline()

		while True:
			topic_terms = topics_csv.readline()
			if topic_terms.strip() == '':
				break
			topic_terms = ' '.join(topic_terms.strip().split(',')[2:])
			labels = [preprocess_label(label)
					  for label in model_hypos.readline().strip().split(' ')]

			# the topic was not included in the survey
			# if topic_terms not in topic_labels_scores:
			#     continue

			# the topic is not in the last survey
			if topic_terms not in target_topic_terms:
				continue

			label_scores = []

			for label in labels:
				if label not in topic_labels_scores[topic_terms]:
					# this is no longer applicable, since every label that doesn't exist has a score
					# of -1 and is treated correctly
					print('Not all labels of the model were included in the survey!')
					continue
				# label_scores.append(topic_labels_scores[topic_terms][label])
				label_scores.append((label, topic_labels_scores[topic_terms][label]))

			# label_scores = list(filter(lambda x: x != -1.0, label_scores))
			label_scores = list(filter(lambda x: x[1] != -1.0, label_scores))

			for label in labels:
				if label in invalid_tokens:
					label_scores.append((label, 0.0))

			if len(label_scores) > 0:
				model_scores[topic_terms] = label_scores
			else:
				model_scores[topic_terms] = []

	# model_scores = np.array(model_scores)
	return model_scores, sufficiently_annotated_labels, insufficiently_annotated_labels, total_annotated_labels


def get_model_stddev(survey_csv_files, topics_csv_file, model_hypos_file, start=2, end=None, ignore_validation=True):
	answer_values = {
		'Not relevant at all': 0,
		'A bit relevant': 1,
		# 'Somehow relevant': 2,
		# 'Relevant': 3,
		'Relevant': 2,
		# 'Very relevant': 4
		'Very relevant': 3
	}

	topic_labels_scores = {}
	target_topic_terms = []
	for csv_file in survey_csv_files:
		# if csv_file == sys.argv[2]:
		#     import pdb
		#     pdb.set_trace()

		responses = pd.read_csv(csv_file)

		invalid_tokens_scores = []
		for _ in range(responses.shape[0]):
			invalid_tokens_scores.append([])

		# skip timestamp and nickname columns
		if end is None:
			end = len(responses.columns)

		for question in responses.columns[start:end]:
			# topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:-1].split(', '))

			# bugged question
			if len(question.split('\n')) == 1:
				continue
				topic_terms = ' '.join(question
									   .split('Select how relevant the terms below are for describing the following word sequence:')[1]
									   .split('Here are two sample paragraphs')[0]
									   .split(', '))

			else:
				topic_terms = ' '.join(question.split('\n')[1].split('[')[0][:].split(', '))

			label = question.split('[')[1].split(']')[0]
			if topic_terms not in topic_labels_scores:
				topic_labels_scores[topic_terms] = {}

			answers = responses[question].values

			if label in invalid_tokens:
				for idx in range(len(answers)):
					if not pd.isnull(answers[idx]):
						invalid_tokens_scores[idx].append(answer_values[answers[idx]])

			answers = answers[~pd.isnull(answers)]
			answers = [answer_values[answer] for answer in answers]

			if len(answers) > 0:
				# topic_labels_scores[topic_terms][label] = np.mean(answers)
				if label not in invalid_tokens:
					# if len(answers) < num_minimum_annotators:
					# 	insufficiently_annotated_labels += 1
					# else:
					if label not in topic_labels_scores[topic_terms]:
						topic_labels_scores[topic_terms][label] = []
						# sufficiently_annotated_labels += 1
					topic_labels_scores[topic_terms][label].extend(answers)
				# sufficiently_annotated_labels += 1

			else:
				if label not in topic_labels_scores[topic_terms] and label not in invalid_tokens:
					topic_labels_scores[topic_terms][label] = []
				# topic_labels_scores[topic_terms][label] = -1.0
				# print(topic_terms)

			# only account for the topics in the last survey
			if csv_file == survey_csv_files[-1]:
				target_topic_terms.append(topic_terms)

		valid_answers = [sum(np.array(user_answers) <= 1) >= 0.25 * len(user_answers)
						 for user_answers in invalid_tokens_scores]
		if not ignore_validation and not np.all(valid_answers):
			print('WARNING: Not all answers passed the validation!')
			import pdb
			pdb.set_trace()

	# topic_labels_scores['voice direction speed signal button format technique light command procedure']['accelerometer'] = 3.0
	# topic_labels_scores['voice direction speed signal button format technique light command procedure']['addition'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['observation'] = 2.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['metabolism'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['circadian rhythm'] = 3.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['contradiction'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['proteus'] = 0.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['hypnotism'] = 3.5
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['subjective experience'] = 4.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['vibration'] = 1.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['unconsciousness'] = 4.0
	# topic_labels_scores['suffix ending brain eye perception visual experienced breath ic historic']['signal processing'] = 1.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['principle'] = 4.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['formalism'] = 4.0
	# topic_labels_scores['argument ad logical attack fallacy toilet beer visited delivery accusative']['argument ad fallacy'] = 2.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['interrogation'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['confidentiality'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['prosecutor'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['apprehension'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['public enquiry'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['crime and punishment'] = 4.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['observation'] = 1.0
	# topic_labels_scores['mid sudden inquiry newly beard ford flavour investigate apropos inquire']['propos inquiry'] = 1.0
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['head of state'] = 3.5
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['freedom of speech'] = 4.0
	# topic_labels_scores['power government kill approved ruler democracy dictator chauvinism unlimited authoritarian']['imperialism'] = 4.0
	# topic_labels_scores['cloud virus amazon trail frankenstein addicted deployed crumb maze othello']['viruses'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['passenger vehicle'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['passenger car'] = 3.0
	# topic_labels_scores['plate vehicle state license motor shall registration law apostille issued']['point of sale'] = 1.0

	global total_annotated_labels
	global insufficiently_annotated_labels
	global sufficiently_annotated_labels

	for _, labels_scores in topic_labels_scores.items():
		for label in labels_scores.keys():
			if len(labels_scores[label]) >= num_minimum_annotators:
				labels_scores[label] = np.std(labels_scores[label])
				sufficiently_annotated_labels += 1
				total_annotated_labels += 1
			elif labels_scores[label]:
				labels_scores[label] = -1.0
				insufficiently_annotated_labels += 1
				total_annotated_labels += 1
			else:
				labels_scores[label] = -1.0
				total_annotated_labels += 1

	model_scores = []
	with open(topics_csv_file) as topics_csv, open(model_hypos_file) as model_hypos:
		# ignore header
		topics_csv.readline()

		while True:
			topic_terms = topics_csv.readline()
			if topic_terms.strip() == '':
				break
			topic_terms = ' '.join(topic_terms.strip().split(',')[2:])
			labels = [preprocess_label(label)
					  for label in model_hypos.readline().strip().split(' ')]

			# the topic was not included in the survey
			# if topic_terms not in topic_labels_scores:
			#     continue

			# the topic is not in the last survey
			if topic_terms not in target_topic_terms:
				continue

			label_scores = []

			for label in labels:
				if label not in topic_labels_scores[topic_terms]:
					# this is no longer applicable, since every label that doesn't exist has a score
					# of -1 and is treated correctly
					print('Not all labels of the model were included in the survey!')
					continue
				# label_scores.append(topic_labels_scores[topic_terms][label])
				label_scores.append((label, topic_labels_scores[topic_terms][label]))

			# label_scores = list(filter(lambda x: x != -1.0, label_scores))
			label_scores = list(filter(lambda x: x[1] != -1.0, label_scores))

			for label in labels:
				if label in invalid_tokens:
					label_scores.append((label, 0.0))

			if len(label_scores) > 0:
				model_scores.append(label_scores)
			else:
				model_scores.append([])

	model_scores = np.array(model_scores)
	return model_scores, sufficiently_annotated_labels, insufficiently_annotated_labels, total_annotated_labels