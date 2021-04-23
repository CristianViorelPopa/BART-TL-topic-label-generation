import sys


def main():
    if len(sys.argv) != 4:
        print("Usage: " + sys.argv[0] + " <topics csv file> <ids input file> <output csv file>")
        exit(0)

    ids = [int(line) for line in open(sys.argv[2]).read().split()]

    with open(sys.argv[1]) as in_file, open(sys.argv[3], 'w') as out_file:
        # start from -1 because of header line, which is always written
        idx = -1
        real_idx = -1
        for line in in_file:
            if idx == -1:
                idx += 1
                real_idx += 1
                out_file.write(line)
                continue

            if idx in ids:
                idx += 1
                continue

            tokens = line.strip().split(',')
            out_file.write(str(real_idx) + ',' + ','.join(tokens[1:]) + '\n')

            idx += 1
            real_idx += 1


if __name__ == '__main__':
    main()
