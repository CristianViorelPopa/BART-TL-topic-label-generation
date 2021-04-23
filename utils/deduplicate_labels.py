import sys


def preprocess_label(label):
    return label.split('(')[0].replace('_', ' ').strip()


def main():
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <target labels file>")
        exit(0)

    labels = [line.strip().split(' ') for line in open(sys.argv[1])]

    unique_labels = []
    for topic_labels in labels:
        unique_labels.append(list(dict.fromkeys(topic_labels)))

    with open(sys.argv[1] + '.unique', 'w') as out_file:
        for labels in unique_labels:
            out_file.write(' '.join(labels) + '\n')


if __name__ == '__main__':
    main()
