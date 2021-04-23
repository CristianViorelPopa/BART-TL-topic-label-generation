import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def preprocess_label(label):
    return label.split('(')[0].replace('_', ' ').strip()


def get_num_new_labels_by_topic(model_dir, indices, max_labels_per_topic):
    topics = []
    train_labels_by_topic = []
    all_train_labels = []
    prev_topic = ''
    with open(os.path.join(model_dir, 'dataset_fairseq', 'train.source')) as source_dataset, \
            open(os.path.join(model_dir, 'dataset_fairseq', 'train.target')) as target_dataset:

        while True:
            source_line = source_dataset.readline()
            target_line = target_dataset.readline()

            if source_line.strip() == '':
                break

            if prev_topic != source_line:
                train_labels_by_topic.append([])
                topics.append(source_line)
                prev_topic = source_line

            label = target_line.strip()

            train_labels_by_topic[-1].append(label)
            all_train_labels.append(label)

    # add the labels from NETL
    with open('summarization/output_unsupervised_netl_all_before_v2') as f:
        count = 0
        for line in f:
            train_labels_by_topic[count].extend(map(lambda label: preprocess_label(label), line.strip().split(' ')))
            train_labels_by_topic[count] = list(set(train_labels_by_topic[count]))
            count += 1
    with open('summarization/output_supervised_netl_all_before_v2') as f:
        count = 0
        for line in f:
            train_labels_by_topic[count].extend(map(lambda label: preprocess_label(label), line.strip().split(' ')))
            train_labels_by_topic[count] = list(set(train_labels_by_topic[count]))
            count += 1

    pred_labels_by_topic = [[preprocess_label(label) for label in line.strip().split(' ')][:max_labels_per_topic]
                            for line in open(os.path.join(model_dir, 'summaries', 'train2_25.hypo'))]

    num_new_labels_by_topic = []
    new_labels_dict = {}
    # for topic_idx in range(len(pred_labels_by_topic)):
    for topic_idx in indices:
        count = 0
        for label in pred_labels_by_topic[topic_idx]:
            if label not in train_labels_by_topic[topic_idx]:
                if topics[topic_idx] not in new_labels_dict:
                    new_labels_dict[topics[topic_idx]] = []
                new_labels_dict[topics[topic_idx]].append(label)
                count += 1
        num_new_labels_by_topic.append(count)
    num_new_labels_by_topic = np.array(num_new_labels_by_topic)

    return num_new_labels_by_topic, new_labels_dict


def main():
    if len(sys.argv) < 4:
        print("Usage: " + sys.argv[0] + " <indices file> <output image> [<model directory>]+")
        exit(0)

    labels = ['BART-TL-ng', 'BART-TL-all']

    limit = 10

    num_topics_per_subject = 6
    indices = []
    with open(sys.argv[1]) as indices_file:
        indices.append([])
        for line in indices_file:
            line = line.strip()
            if line == '---':
                indices[-1] = indices[-1][:num_topics_per_subject]
                indices.append([])
            else:
                indices[-1].append(int(line))
    indices[-1] = indices[-1][:num_topics_per_subject]
    indices = np.array(indices).flatten()

    num_new_labels_by_topic = []
    new_labels_dict = {}
    for model_dir in sys.argv[3:]:
        num_new_labels_by_topic.append([])
        for max_labels_per_topic in range(1, limit + 1):
            model_num_new_labels_by_topic, model_new_labels_dict = get_num_new_labels_by_topic(model_dir, indices, max_labels_per_topic)
            # num_new_labels_by_topic[-1].append(model_num_new_labels_by_topic)
            num_new_labels_by_topic[-1].append(np.mean(model_num_new_labels_by_topic) / max_labels_per_topic)
            for topic, labels_ in model_new_labels_dict.items():
                if topic not in new_labels_dict:
                    new_labels_dict[topic] = []
                new_labels_dict[topic] += model_new_labels_dict[topic]

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    font = {'size': 15}
    matplotlib.rc('font', **font)
    plt.ylim([0, 1])
    plt.ylabel('Proportion of new labels')
    plt.xlabel('Number of top labels')
    plt.xticks(list(range(1, limit + 1)))
    plt.yticks([0.0, 0.25, 0.40, 0.50, 0.75, 1.0])
    plt.hlines(0.40, 1, limit, color='black')

    plt.plot(list(range(1, limit + 1)), num_new_labels_by_topic[0], '.-.', linewidth=2, color='black')
    plt.plot(list(range(1, limit + 1)), num_new_labels_by_topic[1], 'x:', linewidth=2, color='dimgrey')
    plt.legend(labels)
    plt.savefig(sys.argv[2])


if __name__ == '__main__':
    main()
