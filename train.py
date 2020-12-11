from data import MemesDataModule
from model import GPT2
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.tuner.batch_size_scaling import scale_batch_size
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, ProgressBar
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, Union, List, Dict


def calculate_training_steps(data_module: LightningDataModule,
                             gpus: Optional[Union[int, str, List[int]]],
                             batch_size: int,
                             accumulate_grad_batches: Union[int, Dict[int, int], List[list]],
                             num_epochs: int) -> int:
    if gpus is None:
        num_devices = 1
    elif type(gpus) == int:
        num_devices = max(1, gpus)  # -1 bug here!
    elif type(gpus) == list:
        num_devices = len(gpus)
    elif type(gpus) == str:
        num_devices = len(gpus.split(','))  # -1 bug here!
    else:
        raise ValueError(
            'Unexpected type encountered for "gpus" keyword argument. Type should be one of Optional[Union[int, str, List[int]]].')
    effective_batch_size = batch_size * accumulate_grad_batches * num_devices
    return data_module.train_len / effective_batch_size * num_epochs


def main(args) -> None:
    model = GPT2(lr=args.learning_rate, num_warmup_steps=args.num_warmup_steps,
                 weight_decay=args.weight_decay)
    trainer = Trainer.from_argparse_args(
        args, progress_bar_refresh_rate=20, callbacks=[GPUStatsMonitor(), ProgressBar(), ModelCheckpoint()])  # Use 20 here to prevent Google Colab warning
    # if args.auto_scale_batch_size:
    #     if type(args.auto_scale_batch_size) == str:
    #         batch_size = scale_batch_size(
    #             trainer, model, mode=args.auto_scale_batch_size)
    #     else:
    #         batch_size = scale_batch_size(trainer, model)
    # else:
    #     # Use default batch size specified in model
    #     batch_size = model.batch_size
    data = MemesDataModule()
    trainer.tune(model, datamodule=data)
    model.hparams.num_training_steps = calculate_training_steps(img_flip,
                                                                args.gpus,
                                                                data.batch_size,
                                                                args.accumulate_grad_batches,
                                                                args.max_epochs)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tunes pre-trained GPT-2 model with weights from HuggingFace on MemesDataModule using PyTorch Lightning',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=5e-5, help='initial learning rate for AdamW optimizer')
    parser.add_argument('-w', '--num_warmup_steps', type=int,
                        default=0, help='number of warmup steps')
    parser.add_argument('-d', '--weight_decay', type=float,
                        default=0.0, help="weight decay")
    args = parser.parse_args()
    main(args)
