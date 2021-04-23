import sys


def main():
    if len(sys.argv) != 5:
        print("Usage: " + sys.argv[0] + " <input sentences file> <ids input file>"
                                        " <output sentences file> <num sentences>")
        exit(0)

    ids = [int(line) for line in open(sys.argv[2]).read().split()]
    num_sentences = int(sys.argv[4])

    with open(sys.argv[1]) as in_file, open(sys.argv[3], 'w') as out_file:
        # start from -1 because of header line, which is always written
        idx = 0

        while True:
            line = in_file.readline()

            if line == '':
                break

            if idx in ids:
                for _ in range(num_sentences + 1):
                    in_file.readline()

            else:
                out_file.write(line)
                for _ in range(num_sentences + 1):
                    out_file.write(in_file.readline())

            idx += 1


if __name__ == '__main__':
    main()
