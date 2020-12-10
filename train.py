from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, Union, List, Dict


def calculate_training_steps(dataset,
                             gpus: Optional[Union[int, str, List[int]]],
                             batch_size: int,
                             accumulate_grad_batches: Union[int, Dict[int, int], List[list]],
                             num_epochs: int) -> int:
    num_devices = max(1, gpus)
    effective_batch_size = batch_size * accumulate_grad_batches * num_devices
    return (len(dataset.train_dataloader()) / effective_batch_size) * num_epochs

def main(gpus: Optional[Union[int, str, List[int]]],
         accelerator: Optional[str],
         train_sharded: bool,
         amp_backend: str,
         amp_level: str,
         precision: int,
         accumulate_grad_batches: Union[int, Dict[int, int], List[list]],
         autoscale_batch_size: str,
         learning_rate: float,
         num_warmup_steps: int,
         num_epochs: int,
         weight_decay: float) -> None:

    img_flip = MemesDataModule()

    model = GPT2(lr=learning_rate, num_warmup_steps=num_warmup_steps, weight_decay=weight_decay)
    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      plugins='ddp_sharded' if use_sharded and accelerator == 'ddp' else None,
                      amp_backend=amp_backend,
                      amp_level=amp_level,
                      precision=precision,
                      accumulate_grad_batches=accumulate_grad_batches,
                      auto_scale_batch_size=autoscale_batch_size)
    tuner = Tuner(trainer)

    if autoscale_batch_size is not None:
        batch_size = tuner.scale_batch_size(model)  # extra parameters here?
    else:
        batch_size = img_flip.batch_size

    model.hparams.num_training_steps = calculate_training_steps(img_flip,
                                                                gpus,
                                                                batch_size,
                                                                accumulate_grad_batches,
                                                                num_epochs)

    trainer.fit(model, img_flip)


if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'gpus', type=Optional[Union[int, str, List[int]]], help="PyTorch Lightning Trainer's class gpus keyword argument")
    parser.add_argument(
        'accelerator', type=Optional[str], help="PyTorch Lightning Trainer's class accelerator keyword argument")
    parser.add_argument('--train_sharded', action='store_true',
                        help='use sharded training powered by FairScale')
    parser.add_argument('--amp_backend', type=str, default='native',
                        choices=['native', 'apex'], help='which mixed precision backend to use ("native" or "apex")')
    parser.add_argument('--amp_level', type=str, default='O1',
                        choices=['O0', 'O1', 'O2', 'O3'], help='which optimization level to use for APEX backend')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32],
                        help='whether to use full precision (32) or half precision (16) for CPU, GPU, or TPU')
    parser.add_argument('--accumulate_grad_batches', type=Union[int, Dict[int, int], List[list]],
                        default=1, help="Accumulates gradients every k batches or as set up in the dict (in line with PyTorch Lightning's API")
    parser.add_argument('--autoscale_batch_size', type=str, choices=[None, 'power', 'binsearch'],
                        default=None, help='auto scale batch size: (None (no scaling), "power" scaling, or "binsearch" scaling)')
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-w', '--num_warmup_steps', type=int,
                        default=0, help='number of warmup steps')
    parser.add_argument('-e', '--num_epochs', type=int,
                        default=3, help='number of epochs to run fine-tuning')
    parser.add_argument('-d', '--weight_decay', type=float,
                        default=0.0, help="weight decay")
    args = parser.parse_args()
    main(args.gpus,
         args.accelerator,
         args.train_sharded,
         args.amp_backend,
         args.amp_level,
         args.precision,
         args.accumulate_grad_batches,
         args.autoscale_batch_size,
         args.learning_rate,
         args.num_warmup_steps,
         args.num_epochs,
         args.weight_decay)
