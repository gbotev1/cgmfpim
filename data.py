from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from csv import reader as csv_reader
from typing import Optional
from torch import Tensor


class MemeCaptionsDataset(Dataset):
    
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.data = []
        with open(filename, newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            _ = next(tsv_reader)  # Consume header
            for line in tsv_reader:
                self.data.append({'type': line[1], 'caption': line[2], 'tags': line[3].split(','), 'views': int(line[4]), 'upvotes': int(line[5])})
        self.num_captions = len(self.data)  # Precompute length for efficiency

    def __len__(self) -> int:
        return self.num_captions

    def __getitem__(self, item: int) -> Tensor:
        pass

    def __repr__(self) -> str:
        return f'MemeCaptionsDataset containing {self.num_captions} examples initialized from {self.filename}'


class ImgflipMemesDataModule(LightningDataModule):

    def __init__(self, data: str = 'data.tsv', batch_size: int = 32) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size
    
    def prepare_data(self) -> None:
        pass

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
