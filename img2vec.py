# Keep imports lightweight
from PIL import Image
from torchvision.models import wide_resnet101_2
from os import path, listdir
from csv import reader as csv_reader
import torch
import torchvision.transforms as T
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from requests import get as requests_get
from io import BytesIO
from numpy import save, vstack


class Wide_ResNet_101_2:

    def __init__(self, data_dir: str, tsvname: str, outfile: str, captions: str, batch_size: int, timeout: float) -> None:
        # Save parameters
        self.data_dir = data_dir
        self.tsvname = tsvname
        self.outfile = outfile
        self.captions = captions
        self.batch_size = batch_size
        self.timeout = timeout
        # Pipeline set-up
        self.model = wide_resnet101_2(pretrained=True, progress=True)
        # Automatically use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)  # Move model to device
        self.model.eval()  # Don't forget to put model in evaluation mode!
        # Transform all images to be minimum allowed square model size for generalizability, efficiency, and batching
        self.transforms = T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC),  # Use bicubic interpolation for best quality
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])  # Recommended normalization for torchvision ImageNet pretrained models
        self.embeddings = []
        self.model.avgpool.register_forward_hook(lambda m, m_in, m_out: self.embeddings.append(
            m_out.data.detach().cpu().squeeze().numpy()))

    def run(self) -> None:
        lines = []
        image_batch = []
        counter = 0
        iterations = 0
        with open(path.join(self.data_dir, self.tsvname), newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            for line in tsv_reader:
                lines.append(line[0])
                try:
                    r = requests_get(line[1], stream=True,
                                     timeout=self.timeout)
                    image = Image.open(BytesIO(r.content))
                    r.close()  # Close stream when done
                    image_batch.append(self.transforms(image))
                    counter += 1
                    if counter % self.batch_size == 0:
                        # Run image batch
                        self.model(torch.stack(image_batch))
                        # Reset counter and image batch
                        counter = 0
                        image_batch = []
                        # Logging
                        iterations += 1
                        print(f'Batch {iterations} done!')
                except:
                    r.close()  # Make sure stream is closed
                    del lines[-1]  # Prune caption
        # Handle possible incomplete batch
        if counter != 0:
            self.model(torch.stack(image_batch))
            del image_batch, counter
        # Save embeddings
        save(path.join(self.data_dir, self.outfile), vstack(self.embeddings))
        # Save corresponding captions
        with open(path.join(self.data_dir, self.captions), 'w') as outfile:
            for line in lines:
                outfile.write(f'{line}\n')


if __name__ == "__main__":
    parser = ArgumentParser(description='img2vec',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-t', '--tsvname', type=str, default='gcc_full.tsv',
                        help='filename in local data directory of combined, detokenized GCC dataset captions')
    parser.add_argument('-o', '--outfile', type=str, default='embeddings.npy',
                        help='output filename to save in local data directory of embeddings of GCC dataset images')
    parser.add_argument('-c', '--captions', type=str, default='gcc_captions.txt',
                        help='output filename to save in local data directory of GCC dataset captions corresponding to images that were actually embedded')
    parser.add_argument('-b', '--batch_size', type=int, default=1024,
                        help='batch size to use for passing images through model to generate their embeddings')
    parser.add_argument('-w', '--timeout', type=int, default=1,
                        help='timeout in seconds for requests GET method')
    args = parser.parse_args()
    model = Wide_ResNet_101_2(
        args.data_dir, args.tsvname, args.outfile, args.captions, args.batch_size, args.timeout)
    model.run()
