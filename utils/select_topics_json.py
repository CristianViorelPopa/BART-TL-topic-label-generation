import sys
import json


def main():
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <topics json file> <ids input file> <output json file>")
        exit(0)

    ids = [int(line) for line in open(sys.argv[2]).read().split()]

    topics = json.load(open(sys.argv[1]))
    topics = [topics[idx] for idx in (set(range(len(topics))) - set(ids))]
    json.dump(topics, open(sys.argv[3], 'w'))


if __name__ == '__main__':
    main()
