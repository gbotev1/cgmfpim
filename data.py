from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2TokenizerFast
from os import path, environ, cpu_count
from csv import reader as csv_reader
from typing import Optional, List
import torch
import pickle


class MemesDataset(Dataset):

    def __init__(self, path: str, num_memes: int) -> None:
        with open(path, 'rb') as handle:
            self.data = pickle.load(handle)
        self.num_memes = num_memes

    def __len__(self) -> int:
        return self.num_memes

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


class MemesDataModule(LightningDataModule):

    def __init__(self,
                 args,
                 split_ratios: List[float] = [0.8, 0.1, 0.1]) -> None:
        super().__init__()
        self.split_ratios = split_ratios
        self.gpu_boole = torch.cuda.is_available()
        self.num_cpus = cpu_count()
        # There should be no parallelism: stop warnings
        # environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            args.gpt2_model_type)
        # Make sure pad token is also <|endoftext|> and set special separater token
        self.tokenizer.add_special_tokens(
            {'pad_token': self.tokenizer.eos_token, 'sep_token': '<|SEP|>'})
        # Define custom collate function for data loader to tokenize batch properly
        self.collate_fn = lambda batch: self.tokenizer(
            batch, return_tensors='pt', padding=True, truncation=True)
        # Save hyperparameters
        self.save_hyperparameters(args)

    # prepare_data(): called first on MemesDataModule() object
    # produces pickle object at location data_dir/outfile
    def prepare_data(self) -> None:
        captions = []
        with open(path.join(self.hparams.data_dir, self.hparams.infile), newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for meme in tsv_reader:
                # Associate meme's tags to its caption by separating with sep_token
                captions.append(
                    f'{meme[3]}{self.tokenizer.sep_token}{meme[2]}{self.tokenizer.eos_token}')
        self.splits = self.get_splits(len(captions))
        with open(path.join(self.hparams.data_dir, self.hparams.outfile), 'wb') as handle:
            pickle.dump(captions, handle, pickle.HIGHEST_PROTOCOL)

    # get_splits(): called by setup() function
    # produces sizes of train/validation/test dataloaders for use later
    def get_splits(self, data_len: int) -> List[int]:
        # Error handling
        if sum(self.split_ratios) != 1.0:
            raise ValueError(
                f'Split ratios "{self.split_ratios}" given has sum {sum(self.split_ratios)} instead of 1.0.')
        if len(self.split_ratios) != 3:
            raise ValueError(
                f'Split ratios "{self.split_ratios}" given has {len(self.split_ratios)} splits specified instead of 3 (corresponding to train-validation-test).')
        # Make sure each split has an integral size
        splits = [int(self.split_ratios[0] * data_len),
                  int(self.split_ratios[1] * data_len)]
        splits.append(data_len - sum(splits))
        return splits

    # setup(): called second on MemesDataModule object
    # produces train, validation, and test dataloaders
    def setup(self, stage: Optional[str] = None) -> None:
        data = MemesDataset(
            path.join(self.hparams.data_dir, self.hparams.outfile), sum(self.splits))
        self.data_train, self.data_val, self.data_test = random_split(
            data, self.splits)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=self.gpu_boole, collate_fn=self.collate_fn, num_workers=self.num_cpus)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size, pin_memory=self.gpu_boole, collate_fn=self.collate_fn, num_workers=self.num_cpus)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size, pin_memory=self.gpu_boole, collate_fn=self.collate_fn, num_workers=self.num_cpus)
