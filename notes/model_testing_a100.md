EXPERIMENT #1: `time python3 train.py -g gpt2 --gpus 8 --batch_size 2 --progress_bar_refresh_rate 20 --checkpoint_callback --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 3`
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

EXPERIMENT #5: `time python3 train.py -g gpt2-xl --gpus 8 --batch_size 3 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    10m5.590s
      user    9m15.976s
      sys     0m31.536s
```

EXPERIMENT #6: `time python3 train.py -g gpt2-xl --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 1e-5 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    8m55.496s
      user    7m59.026s
      sys     0m39.297s
```

EXPERIMENT #7: `time python3 train.py -g gpt2-xl --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 1e-6 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    8m51.952s
      user    8m3.699s
      sys     0m30.607s
```

EXPERIMENT #7: `time python3 train.py -g gpt2-xl --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 4e-4 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    9m6.785s
      user    8m19.032s
      sys     0m32.454s
```

EXPERIMENT #8: `time python3 train.py -g gpt2 --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 5e-5 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    2m24.347s
      user    2m14.117s
      sys     0m11.649s
```

EXPERIMENT #9: `time python3 train.py -g gpt2 --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 1.6e-3 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    2m22.668s
      user    2m12.454s
      sys     0m12.316s
```

EXPERIMENT #10: `time python3 train.py -g gpt2 --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 1e-8 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    2m22.421s
      user    2m14.815s
      sys     0m26.938s
```

EXPERIMENT #11: `time python3 train.py -g gpt2 --gpus 8 --batch_size 3 --accumulate_grad_batches 4 --learning_rate 1.5e-6 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    2m22.311s
      user    2m11.766s
      sys     0m12.617s
```

EXPERIMENT #12: `time python3 train.py -g gpt2 --gpus 8 --batch_size 3 --accumulate_grad_batches 16 --learning_rate 1.5e-6 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    2m18.144s
      user    2m6.858s
      sys     0m11.431s
```

EXPERIMENT #13: `time python3 train.py -g gpt2 --gpus 8 --batch_size 8 --learning_rate 4e-4 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    1m29.995s
      user    1m23.088s
      sys     0m13.171s
```

EXPERIMENT #14: `time python3 train.py -g gpt2 --gpus 8 --batch_size 16 --accumulate_grad_batches 2 --learning_rate 4e-4 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1`
```
TIME: real    1m8.382s
      user    1m2.333s
      sys     0m12.515s
```

EXPERIMENT #15: `time python3 train.py -g gpt2 --gpus 8 --batch_size 8 --accumulate_grad_batches 4 --learning_rate 4e-4 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1 -i meme_data.tsv`
```
TIME: real    10m43.583s
      user    14m27.900s
      sys     1m15.371s
```

The following experiments were run using the new dataset format.

EXPERIMENT #16: `time python3 train.py -g gpt2 --gpus 8 --batch_size 8 --accumulate_grad_batches 64 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1 -i meme_data.tsv`
```
TIME: real    9m45.052s
      user    13m57.288s
      sys     0m56.061s
```

EXPERIMENT #17: `time python3 train.py -g gpt2 --gpus 8 --learning_rate 1.5e-6 --batch_size 8 --accumulate_grad_batches 4 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1 -i meme_data.tsv`
```
TIME: real    10m3.108s
      user    13m40.333s
      sys     0m56.538s
```

EXPERIMENT #18: `time python3 train.py -g gpt2 --gpus 8 --batch_size 8 --accumulate_grad_batches 4 --accelerator ddp --plugins ddp_sharded --benchmark --seed_everything --max_epochs 1 -i meme_data.tsv`
```
TIME: real    10m15.354s
      user    14m11.430s
      sys     0m55.549s
```
