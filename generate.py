from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from model import GPT2


def main(args: Namespace):
    # Initialize tokenizer the same way we did when training (in MemesDataModule)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model_type)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # Initialize appropriate model
    if args.use_pretrained:
        # Use vanilla pre-trained version
        model = GPT2LMHeadModel.from_pretrained(
            args.gpt2_model_type, pad_token_id=tokenizer.eos_token_id, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    else:
        # Load model weights from checkpoint
        model = GPT2.load_from_checkpoint(
            args.checkpoint, args=args, tokenizer=tokenizer)
    model.eval()  # Don't forget to put model in evaluation mode!
    # Predict, switching based on generation type
    model = model if args.use_pretrained else model.model
    outputs = model.generate(tokenizer.encode('Meme:\n\n', return_tensors='pt', padding=True, truncation=True), eos_token_id=tokenizer.eos_token_id, do_sample=True,
                             max_length=args.max_length, top_p=args.top_p, top_k=args.top_k, num_return_sequences=args.num_return_sequences)
    # Save and print results
    with open(args.outfile, 'w') as outfile:
        for pred in outputs:
            # Detokenize encoding
            meme = tokenizer.decode(pred, skip_special_tokens=True)
            if args.tag is not None:
                candidate = meme.find('Tags:')
                if candidate != -1 and args.tag in meme[candidate + 4:].split(','):
                    print(meme)
                    outfile.write(f'{meme}\n')
            else:
                print(meme)
                outfile.write(f'{meme}\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='Generates meme captions from a GPT-2 model checkpoint with the given parameters for each line of starter text in the specified text file both printing and saving the results.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint filepath from which to load GPT-2 model weights')
    parser.add_argument('-o', '--outfile', type=str, default='test_outfile.txt',
                        help='filename in root directory to store generated model samples')
    parser.add_argument('-g', '--gpt2_model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='pre-trained model ID string for GPT-2')
    parser.add_argument('-p', '--top_p', type=float, default=0.95,
                        help='Huggingface transformers argument description: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    parser.add_argument('-k', '--top_k', type=int, default=50,
                        help='Huggingface transformers argument description: The number of highest probability vocabulary tokens to keep for top-k-filtering. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    parser.add_argument('-l', '--max_length', type=int, default=1024,
                        help='Huggingface transformers argument description: The maximum length of the sequence to be generated. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    parser.add_argument('-n', '--num_return_sequences', type=int, default=100,
                        help='Huggingface transformers argument description: The number of independently computed returned sequences for each element in the batch. See https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate for more information.')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Whether to use the default pre-trained GPT-2 model instead of a fine-tuned one for comparison purposes')
    parser.add_argument('--tag', type=str, default=None,
                        help='Whether to filter generated memes that match provided tag')
    main(parser.parse_args())
