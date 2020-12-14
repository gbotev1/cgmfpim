EXPERIMENT #1: `time python3 train.py -g gpt2 --gpus 8 --batch_size 2 --progress_bar_refresh_rate 20 --checkpoint_callback --accelerator ddp --plugins=ddp_sharded --benchmark --seed_everything --max_epochs 3`
```
TIME: real    9m59.163s
      user    9m54.623s
      sys     0m28.000s
```

EXPERIMENT #2: `time horovodrun -np 8 python3 train.py -g gpt2 --gpus 1 --batch_size 2 --progress_bar_refresh_rate 20 --checkpoint_callback --accelerator horovod --benchmark --seed_everything --max_epochs 3`
```
TIME: real    9m52.191s
      user    36m24.832s
      sys     8m3.980s
```
**Note:**
>The following message was returned during this run: `Primary job terminated normally, but 1 process returned a non-zero exit code. Per user-direction, the job has been aborted.`

EXPERIMENT #3: `time horovodrun -np 8 --autotune --autotune-log-file notes/a100_experiment_3.csv python3 train.py -g gpt2 --gpus 1 --batch_size 2 --progress_bar_refresh_rate 20 --checkpoint_callback --accelerator horovod --benchmark --seed_everything --max_epochs 3`
```
TIME: real    21m40.327s
      user    33m12.908s
      sys     17m4.426s
```
**Note:**
>The following message was returned during this run: `Primary job terminated normally, but 1 process returned a non-zero exit code. Per user-direction, the job has been aborted.`

EXPERIMENT #4: `time horovodrun -np 8 --cache-capacity 8192 --no-hierarchical-allreduce --cycle-time-ms 1.00406 --fusion-threshold-mb 13 python3 train.py -g gpt2 --gpus 1 --batch_size 2 --progress_bar_refresh_rate 20 --checkpoint_callback --accelerator horovod --benchmark --seed_everything --max_epochs 3`
```
TIME: real    10m38.522s
      user    15m0.401s
      sys     2m20.530s
```
**Note:**
>The following message was returned during this run: `Primary job terminated normally, but 1 process returned a non-zero exit code. Per user-direction, the job has been aborted.`
