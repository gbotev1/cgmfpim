EXPERIMENT #1: `time python3 train.py -g gpt2 --max_epochs 1 --gpus 1 --batch_size 8 --deterministic --checkpoint_callback`

    TIME: real    6m50.099s
          user    7m21.193s
          sys     0m24.078s


EXPERIMENT #2: `time python3 train.py -g gpt2 --max_epochs 1 --gpus 1 --batch_size 8 --deterministic --precision 16 --checkpoint_callback`

    TIME: real    6m47.504s
          user    7m19.059s
          sys     0m23.452s

The result of these two experiments is that they looked about the same. Moving forward without specifying --precision.

EXPERIMENT #3: `time python3 train.py -g gpt2 --max_epochs 2 --gpus 1 --batch_size 8 --deterministic --checkpoint_callback`

    TIME: real    12m59.397s
          user    13m52.984s
          sys     0m45.049s

EXPERIMENT #4: `time python3 train.py -g gpt2 --max_epochs 3 --gpus 1 --batch_size 8 --deterministic --checkpoint_callback`

    TIME: real    20m24.709s
          user    21m37.525s
          sys     0m46.100s

Looking at the training losses, the loss seems to plateau at 2 epochs. Moving forward using 2 epochs.

EXPERIMENT #5: `time python3 train.py -g gpt2 --max_epochs 2 --gpus 1 --batch_size 8 --deterministic --checkpoint_callback --accumulate_grad_batches 2`

    TIME: real    12m17.767s
          user    13m8.365s
          sys     0m30.746s

EXPERIMENT #6: `time python3 train.py -g gpt2 --max_epochs 2 --gpus 1 --batch_size 8 --deterministic --checkpoint_callback --accumulate_grad_batches 4`

    TIME: real    11m34.241s
          user    12m25.078s
          sys     0m30.706s

- accumulating gradients [currently 1; 2, 4, 8, 16]
