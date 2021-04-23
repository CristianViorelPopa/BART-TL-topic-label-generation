import sys


def main():
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <candidates input file> <ids input file>"
                                        " <candidates output file>")
        exit(0)

    ids = [int(line) for line in open(sys.argv[2]).read().split()]

    with open(sys.argv[1]) as in_file, open(sys.argv[3], 'w') as out_file:
        idx = 0
        for line in in_file:
            if idx in ids:
                idx += 1
                continue

            out_file.write(line)
            idx += 1


if __name__ == '__main__':
    main()
