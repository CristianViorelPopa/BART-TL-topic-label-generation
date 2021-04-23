import sys
import numpy as np


def preprocess_label(label):
    return label.split('(')[0].replace('_', ' ').strip()


def dcg(scores):
    result = 0
    for idx, score in enumerate(scores):
        result += (2 ** score - 1) / np.log2(idx + 2)
    return result


def ndcg(scores, n):
    target_scores = scores[:n]
    perfect_scores = sorted(scores, reverse=True)[:n]

    return dcg(target_scores) / dcg(perfect_scores)


def main():
    if len(sys.argv) != 3:
        print("Usage: " + sys.argv[0] + " <annotated labels csv file> <target labels file>")
        exit(0)

    with open(sys.argv[1]) as in_file:
        # ignore the header
        in_file.readline()

        label_scores = [{}]
        current_topic = 0
        while True:
            line = in_file.readline()
            if line.strip() == '':
                break

            # format of the annotated dataset: label, topic id, annotator scores...
            tokens = line.strip().split(',')
            topic_id = int(float(tokens[1]))
            if topic_id > current_topic:
                label_scores.append({})
                current_topic = topic_id

            scores = [int(float(score)) for score in tokens[2:]]
            final_score = sum(scores) / len(scores)
            label_scores[topic_id][preprocess_label(tokens[0])] = final_score

    target_labels = [[preprocess_label(label) for label in line.strip().split(' ')]
                     for line in open(sys.argv[2])]

    target_scores = []
    for topic_idx in range(len(target_labels)):
        target_scores.append([])
        added_labels = []
        for label in target_labels[topic_idx]:
            if label in label_scores[topic_idx] and label not in added_labels:
                target_scores[-1].append(label_scores[topic_idx][label])
                added_labels.append(label)

    target_scores = np.array(target_scores)

    print(np.min([len(scores) for scores in target_scores]))

    print("Top-1 Average Rating: " + str(np.mean([scores[0] for scores in target_scores])))
    print("nDCG-1: " + str(np.mean([ndcg(scores, 1) for scores in target_scores])))
    print("nDCG-3: " + str(np.mean([ndcg(scores, 3) for scores in target_scores])))
    print("nDCG-5: " + str(np.mean([ndcg(scores, 5) for scores in target_scores])))


if __name__ == '__main__':
    main()
