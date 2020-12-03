from pytorch_lightning import LightningModule
from transformers import GPT2TokenizerFast, GPT2DoubleHeadsModel

class GPT2(LightningModule):


    def __init__(self, gpt2_model_type: str = 'gpt2'):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Make sure pad token is also <|endoftext|>
        self.model = GPT2DoubleHeadsModel.from_pretrained(gpt2_model_type, pad_token_id=self.tokenizer.eos_token_id)  # Do not forget to update pad token ID too!

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
