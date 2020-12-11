from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, ProgressBar
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, Union, List, Dict


def main(args) -> None:
    datamodule = MemesDataModule()
    model = GPT2(args, datamodule.tokenizer)
    trainer = Trainer.from_argparse_args(
        args, progress_bar_refresh_rate=20, callbacks=[GPUStatsMonitor(), ProgressBar(), ModelCheckpoint()])  # Use 20 here to prevent Google Colab warning
    trainer.tune(model, datamodule=datamodule)
    model.set_num_train_steps(datamodule.get_train_len())  # jank
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-w', '--num_warmup_steps', type=int,
                        default=0, help='number of warmup steps')
    parser.add_argument('-d', '--weight_decay', type=float,
                        default=0.0, help="weight decay")
    parser.add_argument('-c', '--gradient_checkpointing', action='store_true',
                        help='use gradient checkpointing to save memory at expense of slower backward pass')
    parser.add_argument('-g', '--gpt2_model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='pre-trained model ID string for GPT-2')
    args = parser.parse_args()
    main(args)
