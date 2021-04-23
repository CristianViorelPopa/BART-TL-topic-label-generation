import sys
import os
import numpy as np


def preprocess_label(label):
    return label.split('(')[0].replace('_', ' ').strip()


def main():
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <model directory>")
        exit(0)

    train_labels_by_topic = []
    all_train_labels = []
    prev_topic = ''
    with open(os.path.join(sys.argv[1], 'dataset_fairseq', 'train.source')) as source_dataset,\
         open(os.path.join(sys.argv[1], 'dataset_fairseq', 'train.target')) as target_dataset:

        while True:
            source_line = source_dataset.readline()
            target_line = target_dataset.readline()

            if source_line.strip() == '':
                break

            if prev_topic != source_line:
                train_labels_by_topic.append([])
                prev_topic = source_line

            label = target_line.strip()

            train_labels_by_topic[-1].append(label)
            all_train_labels.append(label)

    pred_labels_by_topic = [[preprocess_label(label) for label in line.strip().split(' ')]
                            for line in open(os.path.join(sys.argv[1], 'summaries', 'train.hypo'))]

    all_pred_labels = list(set(np.array(pred_labels_by_topic).flatten()))

    num_new_labels_by_topic = []
    for topic_idx in range(len(pred_labels_by_topic)):
        count = 0
        for label in pred_labels_by_topic[topic_idx]:
            if label not in train_labels_by_topic[topic_idx]:
                count += 1
        num_new_labels_by_topic.append(count)
    num_new_labels_by_topic = np.array(num_new_labels_by_topic)

    all_new_labels = 0
    for label in all_pred_labels:
        if label not in all_train_labels:
            all_new_labels += 1

    print('Mean # of new in-topic labels: ' + str(np.mean(num_new_labels_by_topic)))
    print('Stddev # of new in-topic labels: ' + str(np.std(num_new_labels_by_topic)))
    print('# of all-new labels: ' + str(all_new_labels))


if __name__ == '__main__':
    main()
