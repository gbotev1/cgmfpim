from transformers import GPT2TokenizerFast
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from os import path, cpu_count
from csv import reader as csv_reader
from typing import Optional, List
from argparse import Namespace
from pickle import load, dump, HIGHEST_PROTOCOL
import torch


class MemesDataset(Dataset):

    def __init__(self, path: str) -> None:
        with open(path, 'rb') as handle:
            self.data = load(handle)
        self.num_memes = len(self.data)  # Save for efficiency

    def __len__(self) -> int:
        return self.num_memes

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


class MemesDataModule(LightningDataModule):

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.gpu_boole = torch.cuda.is_available()
        if args.accelerator == 'ddp_spawn':
            self.num_cpus = 0
        elif args.accelerator is None:
            self.num_cpus = cpu_count()
        else:
            self.num_cpus = 1
        # There should be no parallelism: stop warnings
        # environ['TOKENIZERS_PARALLELISM'] = 'false' (maybe it should actually be 'true'?)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            args.gpt2_model_type)
        # Make sure pad token is also <|endoftext|>
        self.tokenizer.add_special_tokens(
            {'pad_token': self.tokenizer.eos_token})
        # Define custom collate function for data loader to tokenize batch properly
        self.collate_fn = lambda batch: self.tokenizer(
            batch, return_tensors='pt', padding=True, truncation=True)
        # Save hyperparameters using hack b/c this is a data module
        self.hparams = args

    # prepare_data(): called first on MemesDataModule() object
    # produces pickle object at location data_dir/outfile
    # do not assign to self in order to work for multi-GPU training!
    def prepare_data(self) -> None:
        captions = []
        with open(path.join(self.hparams.data_dir, self.hparams.infile), newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for meme in tsv_reader:
                # Associate meme type with its caption by separating with special control sequence (do not add new token!)
                caption = meme[2].lower()  # Handle ALL-CAPS captions
                # Strip extraneous whitespace, accounting for multiline text too!
                caption = '\n'.join([line.strip()
                                     for line in caption.split('\n')])
                captions.append(
                    f'Tags: {meme[3]}\n\n{caption}{self.tokenizer.eos_token}')
        with open(path.join(self.hparams.data_dir, self.hparams.outfile), 'wb') as handle:
            dump(captions, handle, HIGHEST_PROTOCOL)

    # get_splits(): called by setup() function
    # produces sizes of train/validation/test dataloaders for use later
    def get_splits(self, data_len: int) -> List[int]:
        # Error handling
        if sum(self.hparams.split_ratios) != 1.0:
            raise ValueError(
                f'Split ratios "{self.hparams.split_ratios}" given has sum {sum(self.hparams.split_ratios)} instead of 1.0.')
        if len(self.hparams.split_ratios) != 3:
            raise ValueError(
                f'Split ratios "{self.hparams.split_ratios}" given has {len(self.hparams.split_ratios)} splits specified instead of 3 (corresponding to train-validation-test).')
        # Make sure each split has an integral size
        splits = [int(self.hparams.split_ratios[0] * data_len),
                  int(self.hparams.split_ratios[1] * data_len)]
        splits.append(data_len - sum(splits))
        return splits

    # setup(): called second on MemesDataModule object
    # produces train, validation, and test dataloaders
    def setup(self, stage: Optional[str] = None) -> None:
        data = MemesDataset(
            path.join(self.hparams.data_dir, self.hparams.outfile))
        splits = self.get_splits(data.num_memes)
        self.data_train, self.data_val, self.data_test = random_split(
            data, splits)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=self.gpu_boole, collate_fn=self.collate_fn, num_workers=self.num_cpus)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size, pin_memory=self.gpu_boole, collate_fn=self.collate_fn, num_workers=self.num_cpus)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size, pin_memory=self.gpu_boole, collate_fn=self.collate_fn, num_workers=self.num_cpus)
