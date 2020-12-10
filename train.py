from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(learning_rate: float, num_warmup_steps: int, num_training_steps: int, weight_decay: float) -> None:
    img_flip = MemesDataModule()
    model = GPT2(lr=learning_rate, num_warmup_steps=num_warmup_steps,
                 num_training_steps=num_training_steps, weight_decay=weight_decay)
    train(model, img_flip)


if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-w', '--num_warmup_steps', type=int,
                        default=0, help='number of warmup steps')
    parser.add_argument('-e', '--num_epochs', type=int,
                        default=3, help='number of epochs to run fine-tuning')
    parser.add_argument('-d', '--weight_decay', type=float,
                        default=0.0, help="weight decay")
    args = parser.parse_args()
    main(args.learning_rate, num_warmup_steps, num_training_steps, weight_decay)
