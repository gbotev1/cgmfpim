from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from csv import reader as csv_reader
from typing import Optional
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

    def __init__(self, infile: str = 'data.tsv', outfile: str = 'data.pickle', batch_size: int = 32) -> None:
        super().__init__()
        self.infile = infile
        self.outfile = outfile
        self.batch_size = batch_size
    
    def prepare_data(self, gpt2_model_type: str = 'gpt2') -> None:
        tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_type)
        data = []
        with open(self.infile, newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for meme in tsv_reader:
                meme_tags = f'<|{meme[3]}|>'
                meme_caption = meme[2][:tokenizer.model_max_length]  # Trim to maximum length allowed by model
                tokenizer_input = f'{meme_tags}{meme_caption}{tokenizer.eos_token}'
                data.append({'caption': tensor(tokenizer.encode(tokenizer_input), dtype=float16), 'views': int(meme[4]), 'upvotes': int(meme[5])})
        with open(self.outfile, mode='wb') as handle:
            pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            pass
        if stage == 'test' or stage is None:
            pass
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size)
