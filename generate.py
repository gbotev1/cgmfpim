from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from transformers import GPT2TokenizerFast
from model import GPT2


def main(args: Namespace):
    # Initialize tokenizer the same way we did when training (in MemesDataModule)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model_type)
    tokenizer.add_special_tokens(
        {'pad_token': tokenizer.eos_token, 'sep_token': '<|SEP|>'})
    # Load model weights from checkpoint
    model = GPT2.load_from_checkpoint(args.checkpoint, tokenizer=tokenizer)
    model.eval()  # Don't forget to put model in evaluation mode!
    # Prepare starter text
    prompts = []
    with open(args.infile) as infile:
        for line in infile:
            # Attach special separator token to complete prompt
            prompts.append(f'{line}{tokenizer.sep_token}')
    # Tokenize
    inputs = tokenizer(prompts, return_tensors='pt',
                       padding=True, truncation=True)
    # Predict
    output = model.model.generate(inputs, eos_token_id=tokenizer.eos_token_id, do_sample=True,
                                  max_length=args.max_length, top_p=args.top_p, top_k=args.top_k)
    # Save and print results
    with open(args.outfile, 'w') as outfile:
        for pred in output:
            # Detokenize encoding
            meme = tokenizer.decode(pred, skip_special_tokens=True)
            print(meme)
            outfile.write(f'{meme}\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='Generates meme captions from a GPT-2 model checkpoint with the given parameters for each line of starter text in the specified text file both printing and saving the results.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint filepath from which to load GPT-2 model weights')
    parser.add_argument('-i', '--infile', type=str, default='test_infile.txt',
                        help='filename in root directory of test primer text for model')
    parser.add_argument('-o', '--outfile', type=str, default='test_outfile.txt',
                        help='filename in root directory to store generated model samples')
    parser.add_argument('-g', '--gpt2_model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='pre-trained model ID string for GPT-2')
    parser.add_argument('-p', '--top_p', type=float, default=0.95,
                        help='Huggingface transformers argument description: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    parser.add_argument('-k', '--top_k', type=int, default=50,
                        help='Huggingface transformers argument description: The number of highest probability vocabulary tokens to keep for top-k-filtering. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    parser.add_argument('-l', '--max_length', type=int, default=50,
                        help='Huggingface transformers argument description: The maximum length of the sequence to be generated. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    main(parser.parse_args())
