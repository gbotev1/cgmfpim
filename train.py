from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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

def main(args) -> None:
    img_flip = MemesDataModule()

    model = GPT2(lr=args.learning_rate, num_warmup_steps=args.num_warmup_steps, weight_decay=args.weight_decay)
    trainer = Trainer.from_argparse_args(args)
    tuner = Tuner(trainer)

    if autoscale_batch_size is not None:
        batch_size = tuner.scale_batch_size(model)  # extra parameters here?
    else:
        batch_size = img_flip.batch_size

    model.hparams.num_training_steps = calculate_training_steps(img_flip,
                                                                args.gpus,
                                                                batch_size,
                                                                args.accumulate_grad_batches,
                                                                args.max_epochs)

    trainer.fit(model, img_flip)


if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--train_sharded', action='store_true',
                        help='use sharded training powered by FairScale')
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-w', '--num_warmup_steps', type=int,
                        default=0, help='number of warmup steps')
    parser.add_argument('-d', '--weight_decay', type=float,
                        default=0.0, help="weight decay")
    args = parser.parse_args()
    main(args)
 