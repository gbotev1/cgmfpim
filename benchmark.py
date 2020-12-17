'''
Usage:
	$ python benchmark.py --meme_data data/meme_data.tsv --model_out data/model_output.txt -o benchmark.txt
	
'''
#%%
import os
import argparse
import numpy as np

from csv import reader as csv_reader
import editdistance
#%%
def read_meme_data(file_path: str):
    captions = []
    with open(os.path.join(os.getcwd(), file_path), newline='') as tsvfile:
        tsv_reader = csv_reader(tsvfile, delimiter='\t')
        _ = next(tsv_reader)  # Consume header
        for meme in tsv_reader:
            # Associate meme type with its caption by separating with special control sequence (do not add new token!)
            type = meme[1].lower() # parse type
            caption = meme[2].lower()  # Handle ALL-CAPS captions
            # Strip extraneous whitespace, accounting for multiline text too!
            caption = '\n'.join([line.strip() for line in caption.split('\n')])
            captions.append((caption, type))
    
    return captions
#%%
def read_model_output(file_path: str):
    captions = []
    with open(os.path.join(os.getcwd(), file_path)) as fh:
        _ = fh.readline() # consume header
        _ = fh.readline()
        while True:
            line = fh.readline()
            if 'DATA' in line: # skip model header
                line = fh.readline()
                continue
            if len(line) == 0:
                break
            if line.startswith('Tags:'):
                # tag= line[6:].rstrip()
                caption = ''
            else:
                caption += line.strip()+'\n'
            if line == '\n':
                captions.append(caption.strip())
			
    return captions
#%%
def nearest_neighbors(meme_train: list, model_out: list, out_file: str):
    with open(os.path.join(os.getcwd(), out_file), 'w') as fh:
        for capt1 in model_out:
            min_dist = np.inf
            neighbors = []
            for capt2, type in meme_train:
                dist = editdistance.eval(capt1, capt2)
                if dist < min_dist:
                    min_dist = dist
                    neighbors = []
                    neighbors.append((capt2, type, dist))
                elif dist == min_dist:
                    neighbors.append((capt2, type, dist))

            print('model output:', capt1, file=fh)
            for i in range(len(neighbors)):
                print(file=fh)
                print('candidate'+str(i+1)+':', neighbors[i][0]+';', 'type:', neighbors[i][1]+';', \
                    'edit dist:', neighbors[i][2], file=fh)
            print(file=fh)
            print(file=fh)
    
    return

#%%
def main():

    parser = argparse.ArgumentParser(prog='benchmark', \
        description='benchmark the performance of our model-generated camptions against the original meme captions')
    parser.add_argument('--meme_data', required = True, help='meme training data')
    parser.add_argument('--model_out', required = True, help='model output captions')
    parser.add_argument('-o', required = True, help='benchmark output')
    
    args = parser.parse_args()
    meme_train = read_meme_data(args.meme_data)
    model_out = read_model_output(args.model_out)
    nearest_neighbors(meme_train, model_out, args.o)
    
    return
#%%
if __name__ == '__main__':
	main()
