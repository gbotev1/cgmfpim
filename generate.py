from transformers import GPT2TokenizerFast, GPT2DoubleHeadsModel
import torch


def main(args):
    # initialize tokenizer and model
    with open(args.checkpoint, 'r') as fn:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2-medium', pad_token_id=tokenizer.eos_token_id)

        # call this the way we called it recently
        inputs = tokenizer.encode("One does not simply", return_tensors='pt')

        # get model from checkpoint
        output = model.generate(inputs, max_length=50, do_sample=True, top_p=0.95, top_k=50)

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate meme captions',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='filename of checkpoint')
    args = parser.parse_args()
    main(args)