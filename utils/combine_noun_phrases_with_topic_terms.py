import sys
from collections import OrderedDict


def main():
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <source topic terms file> <summary file>"
                                        " <output merged file>")
        exit(0)

    terms_as_csv = True
    if terms_as_csv:
        with open(sys.argv[1]) as topics_file:
            # ignore the header
            topics_file.readline()
            topics = [' '.join(line.strip().split(',')[2:]) for line in topics_file]
    else:
        topics = list(OrderedDict.fromkeys([line.strip() for line in open(sys.argv[1])]))
    nps = [line.strip() for line in open(sys.argv[2])]

    assert len(topics) == len(nps)

    with open(sys.argv[3], 'a') as out_file:
        for idx in range(len(topics)):
            out_file.write(topics[idx] + '\n' + nps[idx] + '\n\n')


if __name__ == '__main__':
    main()
