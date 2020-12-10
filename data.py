from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2TokenizerFast
from csv import reader as csv_reader
from typing import Optional, List
import torch
from os import path
import pickle


class MemesDataset(Dataset):

    def __init__(self, path: str) -> None:
        with open(path, 'rb') as handle:
            self.data = pickle.load(handle)
        self.num_memes = len(self.data)  # Precompute length for efficiency

    def __len__(self) -> int:
        return self.num_memes

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.data[item]


class MemesDataModule(LightningDataModule):

    def __init__(self, data_dir: str = 'data', infile: str = 'meme_data.tsv', outfile: str = 'data.pickle', gpt2_model_type: str = 'gpt2', split_ratios: List[float] = [0.8, 0.1, 0.1], batch_size: int = 1) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.infile = infile
        self.outfile = outfile
        self.gpt2_model_type = gpt2_model_type
        self.split_ratios = split_ratios
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        tokenizer = GPT2TokenizerFast.from_pretrained(self.gpt2_model_type)
        # Make sure pad token is also <|endoftext|> and set special separater token
        tokenizer.add_special_tokens(
            {'pad_token': tokenizer.eos_token, 'sep_token': '<|SEP|>'})
        data = []
        with open(path.join(self.data_dir, self.infile), newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for meme in tsv_reader:
                # Associate meme's tags to its caption by separating with sep_token
                tokenizer_input = f'{meme[3]}{tokenizer.sep_token}{tokenizer.eos_token}'
                tokenized_caption = tokenizer(
                    tokenizer_input, return_tensors='pt', padding=True, truncation=True)
                data.append({'caption': tokenized_caption, 'views': int(
                    meme[4]), 'upvotes': int(meme[5])})
        with open(path.join(self.data_dir, self.outfile), 'wb') as handle:
            pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

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

    def setup(self, stage: Optional[str] = None) -> None:
        data = MemesDataset(path.join(self.data_dir, self.outfile))
        splits = self.get_splits(len(data))
        self.data_train, self.data_val, self.data_test = random_split(
            data, splits)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size)
