from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase
from csv import reader as csv_reader
from typing import Optional, List
from torch import tensor, Tensor, float16
from os import path
import pickle


class MemesDataset(Dataset):
    
    def __init__(self, path: str) -> None:
        with open(path, mode='rb') as handle:
            self.data = pickle.load(handle)
        self.num_memes = len(self.data)  # Precompute length for efficiency

    def __len__(self) -> int:
        return self.num_memes

    def __getitem__(self, item: int) -> Tensor:
        return self.data[item]


class MemesDataModule(LightningDataModule):

    def __init__(self, data_dir: str = 'data', infile: str = 'data.tsv', outfile: str = 'data.pickle', gpt2_model_type: str = 'gpt2', split_ratios: List[float] = [0.8, 0.1, 0.1], batch_size: int = 1, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.infile = infile
        self.outfile = outfile
        self.gpt2_model_type = gpt2_model_type
        self.split_ratios = split_ratios
        self.batch_size = batch_size
        self.tokenizer = tokenizer
    
    def prepare_data(self) -> None:
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Make sure pad token is also <|endoftext|>
        data = []
        with open(path.join(self.data_dir, self.infile), newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for meme in tsv_reader:
                meme_tags = f'<|{meme[3]}|>'  # Associate meme's tags to its caption by using custom control code; do not tokenize as special character for generalizability
                tokenizer_input = f'{meme_tags}{meme[2]}{self.tokenizer.eos_token}'
                data.append({'caption': self.tokenizer(tokenizer_input, return_tensors='pt', padding=True, truncation=True), dtype=float16), 'views': int(meme[4]), 'upvotes': int(meme[5])})
        with open(path.join(self.data_dir, self.outfile), mode='wb') as handle:
            pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
    
    def get_splits(self, data_len: int) -> List[int]:
        # Error handling
        if sum(self.split_ratios) != 1.0:
            raise ValueError(f'Split ratios "{self.split_ratios}" given has sum {sum(self.split_ratios)} instead of 1.0.')
        if len(self.split_ratios) != 3:
            raise ValueError(f'Split ratios "{self.split_ratios}" given has {len(self.split_ratios)} splits specified instead of 3 (corresponding to train-validation-test).')
        # Make sure each split has an integral size
        splits = [int(self.split_ratios[0] * data_len), int(self.split_ratios[1] * data_len)]
        splits.append(data_len - sum(splits))
        return splits

    def setup(self, stage: Optional[str] = None) -> None:
        data = MemesDataset(path.join(self.data_dir, self.infile))
        splits = self.get_splits(len(self.data))
        self.data_train, self.data_val, self.data_test = random_split(data, splits)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size)
