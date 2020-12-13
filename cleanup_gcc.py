# Keep imports lightweight
from glob import glob
from os import path
from csv import reader as csv_reader
from pickle import load
from numpy import stack, save
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def main(args: Namespace) -> None:
    caption_indices = set()  # Safe to do b/c embeddings inserted in right order
    embeddings = []
    filenames = glob(path.join(args.data_dir, args.embed_dir, '*.pickle'))
    # Sort filenames in-place by numerical value of file (not lexicographically)
    filenames.sort(key=lambda filename: int(filename.split('/')[-1][:-7]))
    print('Sorted partial embedding files')
    for filename in filenames:
        with open(filename, 'rb') as partial_embed:
            # Zip iterator of (caption index, 2048-dim NumPy image embedding)
            for index, embed in load(partial_embed):
                caption_indices.add(index)
                embeddings.append(embed)
    print('Started stacking embeddings after loading them into memory')
    # Stack embeddings together into single matrix before saving
    embeddings = stack(embeddings)
    print('Finished stacking embeddings')
    save(path.join(args.data_dir, args.outfile), embeddings)
    print('Finished saving embeddings')
    # Save pruned captions as simple text file (no need for TSV anymore)
    with open(path.join(args.data_dir, args.infile), newline='') as tsvfile:
        tsv_reader = csv_reader(tsvfile, delimiter='\t')
        with open(path.join(args.data_dir, args.pruned_captions), 'w') as outfile:
            for i, row in enumerate(tsv_reader):
                if i in caption_indices:
                    outfile.write(f'{row[0]}\n')
    print('Finished saving pruned captions')


if __name__ == "__main__":
    parser = ArgumentParser(description='Executes clean-up tasks on partial zip iterators of caption indices to corresponding embeddings generated after running img2vec.py. Prints helpful progress logs to stdout.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-e', '--embed_dir', type=str, default='embeddings',
                        help='sub-directory of local data directory containing partial GCC embedding dumps from img2vec.py')
    parser.add_argument('-i', '--infile', type=str,
                        default='gcc_full.tsv', help='TSV input filename in local data directory of combined, detokenized GCC dataset')
    parser.add_argument('-p', '--pruned_captions', type=str,
                        default='gcc_captions.txt', help='output text filename in local data directory for pruned GCC dataset captions corresponding to rows of embeddings matrix')
    parser.add_argument('-o', '--outfile', type=str,
                        default='embeddings.npy', help='filename of combined NumPy embeddings matrix to save in local data directory')
    main(parser.parse_args())
