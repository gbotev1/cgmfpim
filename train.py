from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBar
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(args) -> None:
    datamodule = MemesDataModule(args)
    model = GPT2(args, datamodule.tokenizer)
    trainer = Trainer.from_argparse_args(args, callbacks=[ProgressBar()])
    trainer.tune(model, datamodule=datamodule)
    model.set_num_train_steps(datamodule.splits[0])
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1, help='initial batch size')
    parser.add_argument('-s', '--num_warmup_steps', type=int,
                        default=0, help='number of warmup steps')
    parser.add_argument('-w', '--weight_decay', type=float,
                        default=0.0, help='weight decay')
    parser.add_argument('-c', '--gradient_checkpointing', action='store_true',
                        help='use gradient checkpointing to save memory at expense of slower backward pass')
    parser.add_argument('-g', '--gpt2_model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='pre-trained model ID string for GPT-2')
    parser.add_argument('-f', '--freeze_encoder', action='store_true',
                        help='freeze pre-trained weights in encoder "base_model" part of GPT-2')
    parser.add_argument('-d', '--data_dir', type=str, default='data',
                        help='local data directory')
    parser.add_argument('-i', '--infile', type=str, default='meme_data_top.tsv',
                        help='infile TSV name of meme data to initialize MemesDataModule')
    parser.add_argument('-o', '--outfile', type=str, default='data.pickle',
                        help='outfile TSV name of pickle file when MemesDataModule\'s "prepare_data" is called')
    parser.add_argument('--num_training_steps', type=int, default=0,
                        help='DO NOT CHANGE: will be automatically set but added here for code readability')
    args = parser.parse_args()
    main(args)
