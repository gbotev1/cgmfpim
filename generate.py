from transformers import GPT2TokenizerFast, GPT2DoubleHeadsModel
import torch

# initialize tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
model = GPT2DoubleHeadsModel.from_pretrained('gpt2-medium', pad_token_id=tokenizer.eos_token_id)

# call this the way we called it recently
inputs = tokenizer.encode("One does not simply", return_tensors='pt')

# get model from checkpoint
output = model.generate(inputs, max_length=50, do_sample=True, top_p=0.95, top_k=50)