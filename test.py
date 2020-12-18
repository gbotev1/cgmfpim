from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from transformers import GPT2TokenizerFast
from pytorch_lightning import Trainer, seed_everything
from data import MemesDataModule
from model import GPT2


def main(args: Namespace) -> None:
    if args.seed_everything:
        seed_everything(0)  # For reproducibility
    # Initialize tokenizer the same way we did when training (in MemesDataModule)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model_type)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # Validate
    memes_module=MemesDataModule(args)
    model = GPT2(args=args, tokenizer=memes_module.tokenizer)
    # memes_module = MemesDataModule(args)
    # print(memes_module.train_dataloader)
    # print("testing")
    Trainer().test(ckpt_path=args.checkpoint, datamodule=memes_module)


if __name__ == '__main__':
    parser = ArgumentParser(description='Generates meme captions from a GPT-2 model checkpoint with the given parameters for each line of starter text in the specified text file both printing and saving the results.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint filepath from which to load GPT-2 model weights')
    parser.add_argument('-s', '--seed_everything', action='store_true',
                        help="whether to call PyTorch Lightning's \"seed_everything\" method with argument 0 for reproducibility")
    parser.add_argument('-r', '--split_ratios', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                        help='ratios of train, validation, and test dataloaders to use for training')
    parser.add_argument('-g', '--gpt2_model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='pre-trained model ID string for GPT-2')
    main(parser.parse_args())