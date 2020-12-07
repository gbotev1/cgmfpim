# Keep imports lightweight
from PIL import Image
from torchvision.models import wide_resnet101_2
from os import path, makedirs
from os import remove as os_remove
from shutil import rmtree
from csv import reader as csv_reader
from torch import cuda, Tensor
from torch import device as torch_device
import torchvision.transforms as T
from requests import get as requests_get
from io import BytesIO
from numpy import save, stack
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Optional, Tuple


class Wide_ResNet_101_2:

    def __init__(self, data_dir: str, tsvname: str, out_dir: str, captions_index: str, timeout: float, log_every: int) -> None:
        # Save parameters
        self.data_dir = data_dir
        self.tsvname = tsvname
        self.out_dir = out_dir
        self.captions_index = captions_index
        self.timeout = timeout
        self.log_every = log_every
        # Pipeline set-up
        self.model = wide_resnet101_2(pretrained=True, progress=True)
        # Automatically use GPU if available
        self.device = torch_device('cuda' if cuda.is_available() else 'cpu')
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Don't forget to put model in evaluation mode!
        # Transform all images to be minimum allowed square model size for generalizability and efficiency
        self.transforms = T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC),  # Use bicubic interpolation for best quality
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])  # Recommended normalization for torchvision ImageNet pretrained models
        self.embeddings = []
        self.model.avgpool.register_forward_hook(lambda m, m_in, m_out: self.embeddings.append(
            m_out.data.detach().cpu().squeeze().numpy()))

    def embed_line(self, i: int, line: List[str]) -> Optional[int]:
        try:
            r = requests_get(line[1], stream=True, timeout=self.timeout)
            image = Image.open(BytesIO(r.content))
            image = self.transforms(image).unsqueeze(0)  # Fake batch-size of 1
            image = image.to(self.device)
            self.model(image)
            del image
            return i
        except:
            return None

    def run(self) -> None:
        # If captions index file already exists, then delete it
        if path.isfile(path.join(self.data_dir, self.captions_index)):
            os_remove(path.join(self.data_dir, self.captions_index))
        # If embeddings directory already exists, then delete it, otherwise create it
        if path.exists(path.join(self.data_dir, self.out_dir)):
            rmtree(path.join(self.data_dir, self.out_dir))
        makedirs(path.join(self.data_dir, self.out_dir))
        # Switch logic based on CPU/GPU availability
        batches = 0
        caption_indices = []
        if cuda.is_available():
            tensors = []
            with open(path.join(self.data_dir, self.tsvname), newline='') as tsvfile:
                tsv_reader = csv_reader(tsvfile, delimiter='\t')
                for i, line in enumerate(tsv_reader):
                    result = self.embed_line(i, line)
                    if result is not None:
                        caption_indices.append(result)
                    if i % self.log_every == 0:
                        # Save batch
                        save(path.join(self.data_dir, self.out_dir,
                                       f'{batches}.npy'), stack(self.embeddings))
                        # Reset embeddings to ease memory pressure
                        self.embeddings = []
                        batches += 1
                        print(i)
            # Save caption indices
            save(path.join(self.data_dir, self.captions_index), caption_indices)
        else:
            with open(path.join(self.data_dir, self.tsvname), newline='') as tsvfile:
                tsv_reader = csv_reader(tsvfile, delimiter='\t')
                for i, line in enumerate(tsv_reader):
                    result = self.embed_line(i, line)
                    if result is not None:
                        caption_indices.append(result)
                    if i % self.log_every == 0:
                        print(i)
            save(path.join(self.data_dir, self.captions_index), caption_indices)
            save(path.join(self.data_dir, 'embeddings.npy'), stack(self.embeddings))


if __name__ == "__main__":
    parser = ArgumentParser(description="Generates 2048-dimensional embeddings for images from Google's Conceptual Captions dataset using a pretrained Wide ResNet-101-2 neural network on ImageNet. Automatically uses (a single) GPU if available, which is highly recommended as the CPU version currently has an extremely high memory overhead. Note that this program will append to the specified '--captions' TSV file, so it will delete it if it already exists!",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-t', '--tsvname', type=str, default='gcc_full.tsv',
                        help='filename in local data directory of combined, detokenized GCC dataset captions')
    parser.add_argument('-o', '--out_dir', type=str, default='embeddings',
                        help='output directory of partial batch results of embeddings of GCC dataset images in local data directory')
    parser.add_argument('-c', '--captions_index', type=str, default='gcc_captions.npy',
                        help='output filename to save in local data directory of indices in GCC dataset captions corresponding to images that were actually embedded')
    parser.add_argument('-w', '--timeout', type=float, default=1.0,
                        help="timeout in seconds for requests' GET method")
    parser.add_argument('-l', '--log_every', type=int, default=1024,
                        help='how many iterations to save embeddings and print status to stdout stream')
    args = parser.parse_args()
    model = Wide_ResNet_101_2(
        args.data_dir, args.tsvname, args.out_dir, args.captions_index, args.timeout, args.log_every)
    model.run()
