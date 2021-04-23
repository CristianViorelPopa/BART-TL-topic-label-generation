import argparse
import json
import ast
from copy import deepcopy
import random
from functools import reduce
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')  # run once
stop_words = stopwords.words('english')

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

num_answers_per_question = 10
num_invalid_answers_per_question = 1


def prepare_questions(topic_terms, candidates):
    res = []
    res.append(topic_terms)
    res.append([])

    # shuffle clean the candidate list of invalid tokens
    random.shuffle(candidates)
    candidates = [[candidate for candidate in cand_group if candidate not in invalid_tokens]
                  for cand_group in candidates]

    for candidate in candidates:
        if candidate in forbidden_tokens:
            print('Forbidden word found as label in the topic!')

    # modify candidate strings to be fit for answers
    candidates = [[candidate.split('(')[0].strip() for candidate in cand_group]
                  for cand_group in candidates]
    candidates_copy = deepcopy(candidates)

    num_candidate_answers_per_question = num_answers_per_question - num_invalid_answers_per_question
    unique_cands = list(set([cand for cand_group in candidates for cand in cand_group]))

    for idx in range(len(unique_cands) // (num_answers_per_question - 1)):
        candidate_answers = []
        # keep track of which group of candidates had a label extracted
        taken_group = [False] * len(candidates)

        while len(candidate_answers) != num_candidate_answers_per_question:
            if np.all(taken_group):
                for idx in range(len(candidates)):
                    taken_group[idx] = len(candidates[idx]) == 0

            try:
                new_cand = candidates[taken_group.index(False)][0]
            except:
                # empty group
                taken_group[taken_group.index(False)] = True
                continue

            candidate_answers.append(new_cand)

            for idx in range(len(candidates)):
                try:
                    candidates[idx].remove(new_cand)
                    taken_group[idx] = True
                except:
                    pass

        answers = candidate_answers + random.sample(invalid_tokens,
                                                    num_invalid_answers_per_question)
        # one more shuffle, so the invalid answer will have a random position
        random.shuffle(answers)

        res[1].append(answers)

    # start with the flattened list of remaining candidates
    last_candidate_answer = list(set([cand for cand_group in candidates for cand in cand_group]))
    # keep track of which group of candidates had a label extracted
    taken_group = [False] * len(candidates_copy)

    while len(last_candidate_answer) != num_candidate_answers_per_question:
        if np.all(taken_group):
            taken_group = [False] * len(candidates_copy)

        try:
            new_cand = candidates_copy[taken_group.index(False)][0]
        except:
            # empty group
            taken_group[taken_group.index(False)] = True
            continue

        last_candidate_answer.append(new_cand)

        for idx in range(len(candidates_copy)):
            try:
                candidates_copy[idx].remove(new_cand)
                taken_group[idx] = True
            except:
                pass

    last_answer = last_candidate_answer + random.sample(invalid_tokens,
                                                        num_invalid_answers_per_question)
    random.shuffle(last_answer)
    res[1].append(last_answer)

    return res


def main():
    parser = argparse.ArgumentParser(
        description='Script for preparing the survey questions used in Google Forms'
    )
    parser.add_argument('--survey-file', '-v', required=True, type=str,
                        help='Path to the survey file (output of create_survey_array.py)')
    parser.add_argument('--sentences-file', '-s', required=True, type=str,
                        help='Path to the file containing the helping sentences file')
    parser.add_argument('--num-sentences', '-n', required=True, type=int,
                        help='Number of helping sentences per question')
    parser.add_argument('--indices-file', '-i', required=True, type=str,
                        help='File that contains the indices of the topics that will appear in the '
                             'survey')
    parser.add_argument('--output-path', '-o', required=True, type=str,
                        help='Path to the output file that will contain the survey questions')

    args = parser.parse_args()

    num_sentences = args.num_sentences

    NUM_TOPICS_PER_SUBJECT = 6

    real_indices = []
    with open(args.indices_file) as indices_file:
        real_indices.append([])
        for line in indices_file:
            line = line.strip()
            if line == '---':
                real_indices[-1] = real_indices[-1][:NUM_TOPICS_PER_SUBJECT]
                real_indices.append([])
            else:
                real_indices[-1].append(int(line))
        real_indices[-1] = real_indices[-1][:NUM_TOPICS_PER_SUBJECT]

    with open(args.survey_file) as survey_file, open(args.sentences_file) as sentences_file:
        curr_index = 0
        topic_questions = []
        topic_questions.append([])

        current_subject_idx = 0
        sentence_file_idx = 0
        while True:
            # ~~~ PROCESS THE SURVEY FILE ~~~
            line = survey_file.readline()
            # done
            if line == '':
                break

            topic_info = ast.literal_eval(line.strip())
            topic_terms = topic_info[0]
            # candidates = survey_file.readline().strip().split(' ')
            candidates = topic_info[1][0]
            # candidates = [cand.replace(' ', '_') for cand in candidates]
            curr_topic_questions = prepare_questions(topic_terms, candidates)

            curr_topic_questions.append([])
            for _ in range(num_sentences):
                sentence = sentences_file.readline().strip()
                sentence_file_idx += 1
                curr_topic_questions[2].append(sentence)

            # read empty line
            _ = sentences_file.readline()
            sentence_file_idx += 1

            if curr_index not in real_indices[current_subject_idx]:
                curr_index += 1
                continue
            curr_index += 1

            print(sentence_file_idx)

            # ~~~ PROCESS THE SENTENCES_FILE
            # topic terms, useless
            # _ = sentences_file.readline()

            topic_questions[-1].append(curr_topic_questions)
            if len(topic_questions[-1]) == NUM_TOPICS_PER_SUBJECT:
                current_subject_idx += 1
                if current_subject_idx == 5:
                    break
                topic_questions.append([])

    # shape of `topic_questions`: #topics x #Qs_per_topic x (#terms + #Qs x #As_per_Q + #sentences)

    with open(args.output_path, 'a') as out_file:
        out_file.write(json.dumps(topic_questions) + '\n')


if __name__ == '__main__':
    main()
