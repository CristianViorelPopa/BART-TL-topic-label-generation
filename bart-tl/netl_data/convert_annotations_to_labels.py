import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Script for converting NETL annotations to labels'
    )
    parser.add_argument('--annotated-dataset', '-d', required=True, type=str,
                        help='Where the annotated dataset is stored')
    parser.add_argument('--output-file', '-o', required=True, type=str,
                        help='Where the extracted labels will be stored')

    args = parser.parse_args()

    with open(args.annotated_dataset) as in_file, open(args.output_file, 'w') as out_file:
        # ignore the header
        in_file.readline()

        labels = [[]]
        current_topic = 0
        while True:
            line = in_file.readline()
            if line.strip() == '':
                break

            # format of the annotated dataset: label, topic id, annotator scores...
            tokens = line.strip().split(',')
            topic_id = int(float(tokens[1]))
            if topic_id > current_topic:
                labels.append([])
                current_topic = topic_id

            scores = [int(float(score)) for score in tokens[2:]]
            final_score = sum(scores) / len(scores)
            labels[topic_id].append((tokens[0], final_score))

        # sort labels (descending, best one first) by their annotator scores
        for idx in range(len(labels)):
            labels[idx] = sorted(labels[idx], key=lambda x: x[1], reverse=True)

        # remove the scores, no longer useful
        # replace spaces in label with '_'
        # join labels for each topic into a single string
        labels = [' '.join(['_'.join(label[0].split(' ')) for label in topic_labels])
                  for topic_labels in labels]

        out_file.write('\n'.join(labels))


if __name__ == '__main__':
    main()
