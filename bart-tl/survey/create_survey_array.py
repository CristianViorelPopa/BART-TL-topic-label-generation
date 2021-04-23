import sys
from collections import OrderedDict


def preprocess_label(label):
    return label.split('(')[0].replace('_', ' ').strip()


def main():
    if len(sys.argv) < 4:
        print("Usage: " + sys.argv[0] + " <source topic terms file> <output file> [<labels file>]+")
        exit(0)

    terms_as_csv = True
    if terms_as_csv:
        with open(sys.argv[1]) as topics_file:
            # ignore the header
            topics_file.readline()
            topics = [(line.strip().split(',')[2:]) for line in topics_file]
    else:
        topics = list(OrderedDict.fromkeys([line.strip().split(' ') for line in open(sys.argv[1])]))

    labels = []
    new_labels = [line.strip().split(' ') for line in open(sys.argv[3])]
    new_labels = [[preprocess_label(label) for label in topic_labels]
                  for topic_labels in new_labels]
    assert len(topics) == len(new_labels)

    for idx in range(len(topics)):
        labels.append([])
        labels[-1].append(new_labels[idx])

    for idx in range(len(sys.argv[4:])):
        new_labels = [line.strip().split(' ') for line in open(sys.argv[idx + 4])]
        new_labels = [[preprocess_label(label) for label in topic_labels] for topic_labels in
                      new_labels]
        assert len(topics) == len(new_labels)

        for idx2 in range(len(topics)):
            labels[idx2].append(new_labels[idx2])

    survey = []
    for idx in range(len(topics)):
        survey.append([topics[idx], [labels[idx]]])

    with open(sys.argv[2], 'w') as out_file:
        out_file.write('\n'.join([str(survey_item) for survey_item in survey]))


if __name__ == '__main__':
    main()
