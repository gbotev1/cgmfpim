from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import model

def main(args: Namespace):
    # initialize tokenizer and model
    with open(args.checkpoint, 'r') as fn:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2-medium', pad_token_id=tokenizer.eos_token_id)

        # call this the way we called it recently
        inputs = tokenizer.encode("One does not simply", return_tensors='pt')

        # get model from checkpoint
        output = model.generate(inputs, max_length=50, do_sample=True, top_p=0.95, top_k=50)

if __name__ == '__main__':
    parser = ArgumentParser(description='Generates meme captions from a GPT-2 model checkpoint with the given parameters for each line of starter text in the specified text file both printing and saving the results.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint', type=str, help='checkpoint filename from which to load GPT-2 model weights')
    parser.add_argument('-l', '--logs_dir', type=str,
                        default='lightning_logs', help="directory containing PyTorch Lightning's logs")
    main(parser.parse_args())
