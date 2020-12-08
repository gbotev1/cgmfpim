from pytorch_lightning import LightningModule
from transformers import GPT2TokenizerFast, GPT2DoubleHeadsModel, get_cosine_schedule_with_warmup
import torch
import torch.optim as optim
from typing import Dict, Union


class GPT2(LightningModule):

    def __init__(self, lr: float,
                       num_warmup_steps: int,
                       num_training_steps: int,
                       gpt2_model_type: str = 'gpt2'):
        super(GPT2, self).__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_type)
        # Make sure pad token is also <|endoftext|>
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2DoubleHeadsModel.from_pretrained(
            gpt2_model_type, pad_token_id=self.tokenizer.eos_token_id)  # Do not forget to update pad token ID too!
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_hyperparameters('lr')

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], _) -> None:
        # Try to predict input IDs by setting them as labels
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], _) -> None:
        # Try to predict input IDs by setting them as labels
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], _) -> None:
        # Try to predict input IDs by setting them as labels
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        if "weight_decay" in self.hparams:
            weight_decay = self.hparams
        else:
            weight_decay = 0.0

        optimizer_grouped_parameters = [{ "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                                         "weight_decay": weight_decay }]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.num_warmup_steps, self.num_training_steps)
        return [optimizer], [scheduler]
