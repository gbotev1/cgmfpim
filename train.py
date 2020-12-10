import data
from model import GPT2

dataset = data.MemesDataModule() # data must be in data/meme_data.tsv (this is specified in data.py)
dataset.prepare_data() # this generates a pickle
dataset.setup() # generates dataloaders

train_dl = dataset.train_dataloader()
val_dl = dataset.val_dataloader()
test_dl = dataset.test_dataloader()

# specify model
GPT2_model = GPT2(lr=0.02, num_warmup_steps=1, num_training_steps=1, weight_decay=0.0)

### TODO: specify loop over which to train/validate/test