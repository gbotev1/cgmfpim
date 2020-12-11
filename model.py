from pytorch_lightning import LightningModule
from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase, GPT2DoubleHeadsModel, get_cosine_schedule_with_warmup
import torch
import torch.optim as optim
from typing import Dict, Union


class GPT2(LightningModule):

    def __init__(self,
                 args,
                 tokenizer: PreTrainedTokenizerBase,
                 batch_size: int = 1,
                 ):
        super(GPT2, self).__init__()
        # Update both pad_token and newly added sep_token
        self.model = GPT2DoubleHeadsModel.from_pretrained(
            args.gpt2_model_type, pad_token_id=tokenizer.eos_token_id, sep_token_id=tokenizer.sep_token_id)
        # Resize model's token embedding
        self.model.resize_token_embeddings(len(tokenizer))
        self.learning_rate = args.learning_rate
        self.num_warmup_steps = args.num_warmup_steps
        self.num_training_steps = 0  # Avoid problems with tuner
        self.weight_decay = args.weight_decay
        self.gpus = args.gpus
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.gradient_checkpointing = args.gradient_checkpointing
        self.num_epochs = args.max_epochs
        self.batch_size = batch_size
        self.save_hyperparameters(args)

    def set_num_train_steps(self, train_len: int) -> None:
        if self.gpus is None:
            num_devices = 1
        elif type(self.gpus) == int:
            num_devices = max(1, self.gpus)  # -1 bug here!
        elif type(self.gpus) == list:
            num_devices = len(self.gpus)
        elif type(self.gpus) == str:
            num_devices = len(self.gpus.split(','))  # -1 bug here!
        else:
            raise ValueError(
                'Unexpected type encountered for "gpus" keyword argument. Type should be one of Optional[Union[int, str, List[int]]].')
        effective_batch_size = self.batch_size * \
            self.accumulate_grad_batches * num_devices
        self.num_training_steps = train_len / effective_batch_size * self.num_epochs

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        # No need to log training loss explicitly: https://pytorch-lightning.readthedocs.io/en/latest/new-project.html#logging; is sync_dist=True though?

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_index: int) -> None:
        # Try to predict input IDs by setting them as labels (verified approach in documentation)
        outputs = self(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'labels': batch['input_ids']})
        loss = outputs[0]
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                                         'weight_decay': self.weight_decay}]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.num_warmup_steps, self.num_training_steps)
        return [optimizer], [scheduler]
