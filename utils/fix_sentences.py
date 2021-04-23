import sys


def main():
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <sentences file> <topics csv file> <num sentences>")
        exit(0)

    num_sentences = int(sys.argv[3])
    sentences = []
    with open(sys.argv[1]) as sentences_file:
        idx = 0
        for line in sentences_file:
            sentence = line.strip()
            if idx % (num_sentences + 1) == 0:
                sentences.append([sentence])

            elif (idx + 1) % (num_sentences + 1) == 0:
                pass

            else:
                sentences[-1].append(sentence)

            idx += 1

    import pdb
    pdb.set_trace()

    with open(sys.argv[2]) as topics_file, open(sys.argv[1], 'w') as sentences_file:
        topics_file.readline()
        idx = 0
        for line in topics_file:
            sentences_file.write(', '.join(line.strip().split(',')[2:]) + '\n')
            sentences_file.write('\n'.join(sentences[idx]) + '\n\n')
            idx += 1


if __name__ == '__main__':
    main()
