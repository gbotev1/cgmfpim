# Keep imports lightweight
from glob import glob
from os import path
from csv import reader as csv_reader
from pickle import load, dump, HIGHEST_PROTOCOL
from numpy import stack
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(data_dir: str, embed_dir: str, infile: str, pruned_captions: str, outfile: str) -> None:
    caption_indices = set()  # Safe to do b/c embeddings inserted in right order
    embeddings = []
    filenames = glob(path.join(data_dir, embed_dir, '*.pickle'))
    # Sort filenames in-place by numerical value of file (not lexicographically)
    filenames.sort(key=lambda filename: int(filename.split('/')[-1][:-7]))
    for filename in filenames:
        with open(filename, 'rb') as partial_embed:
            # Zip iterator of (caption index, 2048-dim NumPy image embedding)
            for index, embed in load(partial_embed):
                caption_indices.add(index)
                embeddings.append(embed)
    # Stack embeddings together into single matrix before saving
    embeddings = stack(embeddings)
    with open(path.join(data_dir, outfile), 'wb') as outfile:
        dump(caption_indices, outfile, protocol=HIGHEST_PROTOCOL)
    # Save pruned captions as simple text file (no need for TSV anymore)
    with open(path.join(data_dir, infile), newline='') as tsvfile:
        tsv_reader = csv_reader(tsvfile, delimiter='\t')
        with open(path.join(data_dir, pruned_captions), 'w') as outfile:
            for i, row in enumerate(tsv_reader):
                if i in caption_indices:
                    outfile.write(f'{row[0]}\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="Performs clean-up tasks after processing the GCC dataset in img2vec.py.",
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
                        default='embeddings.pickle', help='filename of combined NumPy embeddings matrix to save in local data directory')
    args = parser.parse_args()
    main(args.data_dir,
         args.embed_dir,
         args.infile,
         args.pruned_captions,
         args.outfile)
