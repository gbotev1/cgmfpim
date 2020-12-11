from pytorch_lightning import LightningModule
from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase, GPT2LMHeadModel, get_cosine_schedule_with_warmup
import torch
import torch.optim as optim
from typing import Dict, Union


class GPT2(LightningModule):

    def __init__(self,
                 args,
                 tokenizer: PreTrainedTokenizerBase,
                 ):
        super(GPT2, self).__init__()
        # Update both pad_token and newly added sep_token
        self.model = GPT2LMHeadModel.from_pretrained(
            args.gpt2_model_type, pad_token_id=tokenizer.eos_token_id, sep_token_id=tokenizer.sep_token_id)
        # Resize model's token embedding
        self.model.resize_token_embeddings(len(tokenizer))
        # Freeze encoder if requested
        if args.freeze_encoder:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        # Save hyperparameters
        self.save_hyperparameters(args)

    def set_num_train_steps(self, train_len: int) -> None:
        if self.hparams.gpus is None:
            num_devices = 1
        elif type(self.hparams.gpus) == int:
            num_devices = max(1, self.hparams.gpus)  # -1 bug here!
        elif type(self.hparams.gpus) == list:
            num_devices = len(self.hparams.gpus)
        elif type(self.hparams.gpus) == str:
            num_devices = len(self.hparams.gpus.split(','))  # -1 bug here!
        else:
            raise ValueError(
                'Unexpected type encountered for "gpus" keyword argument. Type should be one of Optional[Union[int, str, List[int]]].')
        effective_batch_size = self.hparams.batch_size * \
            self.hparams.accumulate_grad_batches * num_devices
        self.hparams.num_training_steps = train_len / \
            effective_batch_size * self.hparams.max_epochs

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return outputs[0]

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                                         'weight_decay': self.hparams.weight_decay}]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        return [optimizer], [scheduler]
