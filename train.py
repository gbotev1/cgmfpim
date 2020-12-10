from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, Union, List


def main(gpus: Optional[Union[int, str, List[int]]],
         accelerator: Optional[str],
         amp_backend: str,
         learning_rate: float,
         num_warmup_steps: int,
         num_epochs: int,
         weight_decay: float) -> None:
         
    img_flip = MemesDataModule()
    model = GPT2(lr=learning_rate, num_warmup_steps=num_warmup_steps,
                 num_training_steps=num_training_steps, weight_decay=weight_decay)
    trainer = Trainer(gpus=gpus,
                      accelerator=accelerator,
                      amp_backend=amp_backend)
    trainer.fit(model, img_flip)


if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'gpus', type=Optional[Union[int, str, List[int]]], help="PyTorch Lightning Trainer's class gpus keyword argument")
    parser.add_argument(
        'accelerator', type=Optional[str], help="PyTorch Lightning Trainer's class accelerator keyword argument")
    parser.add_argument('-a', '--amp_backend', type=str,
                        default='native', help='which mixed precision backend to use ("native" or "apex")')
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
         args.amp_backend,
         args.learning_rate,
         args.num_warmup_steps,
         args.num_epochs,
         args.weight_decay)
