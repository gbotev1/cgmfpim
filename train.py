from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ProgressBar, ModelCheckpoint
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def main(args: Namespace) -> None:
    if args.seed_everything:
        seed_everything(0)  # For reproducibility
    datamodule = MemesDataModule(args)
    model = GPT2(args=args, tokenizer=datamodule.tokenizer)
    trainer = Trainer.from_argparse_args(args, callbacks=[ProgressBar(), ModelCheckpoint(
        monitor='train_loss', save_top_k=args.max_epochs, save_weights_only=True)])  # Save checkpoint after every epoch
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1, help='initial batch size')
    parser.add_argument('-c', '--gradient_checkpointing', action='store_true',
                        help='use gradient checkpointing to save memory at expense of slower backward pass')
    parser.add_argument('-s', '--seed_everything', action='store_true',
                        help="whether to call PyTorch Lightning's \"seed_everything\" method with argument 0 for reproducability")
    parser.add_argument('-g', '--gpt2_model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='pre-trained model ID string for GPT-2')
    parser.add_argument('-d', '--data_dir', type=str, default='data',
                        help='local data directory')
    parser.add_argument('-i', '--infile', type=str, default='meme_data_top.tsv',
                        help='infile TSV name of meme data to initialize MemesDataModule')
    parser.add_argument('-o', '--outfile', type=str, default='data.pickle',
                        help='outfile TSV name of pickle file when MemesDataModule\'s "prepare_data" is called')
    parser.add_argument('-r', '--split_ratios', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                        help='ratios of train, validation, and test dataloaders to use for training')
    main(parser.parse_args())
