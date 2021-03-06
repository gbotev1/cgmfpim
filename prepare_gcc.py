# Keep imports lightweight
from csv import reader as csv_reader
from csv import writer as csv_writer
from os import path
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

detokenizer = TreebankWordDetokenizer()


def process_gcc_split(data_dir: str, tsvname: str) -> List[List[str]]:
    lines = []
    with open(path.join(data_dir, tsvname), newline='') as tsvfile:
        tsv_reader = csv_reader(tsvfile, delimiter='\t')
        for line in tsv_reader:
            lines.append([detokenizer.detokenize(line[0].split()), line[1]])
    return lines


def main(args: Namespace) -> None:
    lines = process_gcc_split(args.data_dir, args.train)
    lines.extend(process_gcc_split(args.data_dir, args.val))
    with open(path.join(args.data_dir, args.output), 'w', newline='') as tsvfile:
        tsv_writer = csv_writer(tsvfile, delimiter='\t')
        tsv_writer.writerows(lines)


if __name__ == '__main__':
    parser = ArgumentParser(description="Combines the already downloaded train and validation files from Google's Conceptual Captions dataset into a single TSV file, detokenizing its captions using NLTK's TreebankWordDetokenizer.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-t', '--train', type=str, default='gcc_train.tsv',
                        help='TSV input filename in local data directory of training split of GCC dataset')
    parser.add_argument('-v', '--val', type=str, default='gcc_val.tsv',
                        help='TSV input filename in local data directory of validation split of GCC dataset')
    parser.add_argument('-o', '--output', type=str, default='gcc_full.tsv',
                        help='TSV output filename to save in local data directory of combined, detokenized GCC dataset')
    main(parser.parse_args())
