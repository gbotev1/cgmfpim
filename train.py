import data
from model import GPT2

def prepare_data():
    dataset = data.MemesDataModule() # data must be in data/meme_data.tsv (this is specified in data.py)
    dataset.prepare_data() # this generates a pickle
    dataset.setup() # generates dataloaders

    train_dl = dataset.train_dataloader()
    val_dl = dataset.val_dataloader()
    test_dl = dataset.test_dataloader()
    return train_dl, val_dl, test_dl

def define_model(learning_rate, num_warmup_steps, num_training_steps, weight_decay)
    GPT2_model = GPT2(lr=learning_rate, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, weight_decay=weight_decay)
    return GPT2_model

def train(train_dl, val_dl, test_dl, model):
    ### TODO: fill in this code

def main(learning_rate, num_warmup_steps, num_training_steps, weight_decay):
    train_dl, val_dl, test_dl = prepare_data()
    model = define_model(learning_rate, num_warmup_steps, num_training_steps, weight_decay)
    train(train_dl, val_dl, test_dl, model)

if __name__ == "__main__":
    parser = ArgumentParser(description="Train GPT-2 model with pre-defined weights from HuggingFace on dataset as specified in data.py.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, # TODO: tweak this default
                        help='learning rate for model')
    parser.add_argument('-w', '--num_warmup_steps', type=int, default=100, # TODO: tweak this default
                        help='number of warmup steps') # TODO: make more descriptive
    parser.add_argument('-t', '--num_training_steps', type=int, default=100, # TODO: tweak this default
                        help='number of training steps') # TODO: make more descriptive
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, # TODO: tweak this default
                        help="weight decay")
    args = parser.parse_args()

    main(learning_rate, num_warmup_steps, num_training_steps, weight_decay)