EXPERIMENT #1: time python3 train.py -g gpt2 --max_epochs 1 --gpus 1 --batch_size 8 --deterministic --checkpoint_callback
    TIME: real    6m50.099s
          user    7m21.193s
          sys     0m24.078s


EXPERIMENT #2: time python3 train.py -g gpt2 --max_epochs 1 --gpus 1 --batch_size 8 --deterministic --precision 16 --checkpoint_callback
    TIME: real    6m47.504s
	  user    7m19.059s
	  sys     0m23.452s

The result of these two experiments is that they looked about the same. Moving forward without specifying --precision.

- epochs
- accumulating gradients [currently 1; 2, 4, 8, 16]
- 
