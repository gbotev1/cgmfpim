from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2TokenizerFast
from csv import reader as csv_reader
from typing import Optional, List
from torch import tensor, Tensor, float16
import pickle


class MemesDataset(Dataset):
    
    def __init__(self, infile: str) -> None:
        with open(infile, mode='rb') as handle:
            self.data = pickle.load(handle)
        self.num_memes = len(self.data)  # Precompute length for efficiency

    def __len__(self) -> int:
        return self.num_memes

    def __getitem__(self, item: int) -> Tensor:
        return self.data[item]


class MemesDataModule(LightningDataModule):

    def __init__(self, infile: str = 'data.tsv', outfile: str = 'data.pickle', gpt2_model_type: str = 'gpt2', split_ratios: List[float] = [0.8, 0.1, 0.1], batch_size: int = 1) -> None:
        super().__init__()
        self.infile = infile
        self.outfile = outfile
        self.gpt2_model_type = gpt2_model_type
        self.split_ratios = split_ratios
        self.batch_size = batch_size
    
    def prepare_data(self) -> None:
        tokenizer = GPT2TokenizerFast.from_pretrained(self.gpt2_model_type)
        data = []
        with open(self.infile, newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for meme in tsv_reader:
                meme_tags = f'<|{meme[3]}|>'  # Associate meme's tags to its caption by using custom control code; do not tokenize as special character for generalizability
                meme_caption = meme[2][:tokenizer.model_max_length]  # Trim to maximum length allowed by model
                tokenizer_input = f'{meme_tags}{meme_caption}{tokenizer.eos_token}'
                data.append({'caption': tensor(tokenizer.encode(tokenizer_input), dtype=float16), 'views': int(meme[4]), 'upvotes': int(meme[5])})
        with open(self.outfile, mode='wb') as handle:
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
        data = MemesDataset(self.infile)
        splits = self.get_splits(len(self.data))
        self.data_train, self.data_val, self.data_test = random_split(data, splits)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size)
