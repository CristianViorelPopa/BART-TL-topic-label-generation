import sys


def preprocess_label(label):
    return label.split('(')[0].replace('_', ' ').strip()


def main():
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <target labels file>")
        exit(0)

    labels = [[preprocess_label(label) for label in line.strip().split(' ')]
              for line in open(sys.argv[1])]

    duplicates = []
    for idx, topic_labels in enumerate(labels):
        if len(list(set(topic_labels))) != len(topic_labels):
            duplicates.append(str(idx))

    if not duplicates:
        print('OK! No duplicate labels in the file')
    else:
        print('Duplicate labels in topics: ' + ','.join(duplicates))


if __name__ == '__main__':
    main()
