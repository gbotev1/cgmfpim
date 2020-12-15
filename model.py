from transformers import PreTrainedTokenizerBase, GPT2LMHeadModel
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from typing import Dict, Union, Optional
from argparse import Namespace


class GPT2(LightningModule):

    def __init__(self,
                 args: Optional[Namespace] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 ):
        super(GPT2, self).__init__()
        # Make sure that valid arguments were given
        if args is None:
            raise ValueError(
                "A valid \"args\" namespace must be provided to initialize this model. The \"args\" appears as an optional argument to be compatible with PyTorch Lightning's checkpoint loading function.")
        if tokenizer is None:
            raise ValueError(
                "A valid \"tokenizer\" must be provided to initialize this model. The \"tokenizer\" appears as an optional argument to be compatible with PyTorch Lightning's checkpoint loading function.")
        # Update pad_token_id
        self.model = GPT2LMHeadModel.from_pretrained(
            args.gpt2_model_type, pad_token_id=tokenizer.eos_token_id)
        # Save hyperparameters
        self.save_hyperparameters(args)

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        # lgtm[py/call-to-non-callable]
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return outputs[0]

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        # lgtm[py/call-to-non-callable]
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        # lgtm[py/call-to-non-callable]
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        return optim.AdamW([p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], lr=self.hparams.learning_rate)
