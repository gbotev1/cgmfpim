# Keep imports lightweight
from PIL import Image
from torchvision.models import wide_resnet101_2
from os import path, makedirs
from shutil import rmtree
from csv import reader as csv_reader
from torch import cuda, Tensor, device
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from requests import get as requests_get
from io import BytesIO
from numpy import stack
from pickle import dump, HIGHEST_PROTOCOL
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Optional
from queue import Queue, Empty
from warnings import simplefilter


class Wide_ResNet_101_2:

    def __init__(self, data_dir: str, tsvname: str, embed_dir: str, timeout: float, log_every: int) -> None:
        # Save parameters
        self.data_dir = data_dir
        self.tsvname = tsvname
        self.embed_dir = embed_dir
        self.timeout = timeout
        self.log_every = log_every
        # Turn PIL warnings into exceptions to filter out bad images
        simplefilter('error', Image.DecompressionBombWarning)
        simplefilter('error', UserWarning)
        # Automatically use GPU if available
        if not cuda.is_available():
            raise RuntimeError(
                'Must have CUDA installed in order to run this program.')
        self.device = device('cuda')
        self.model = wide_resnet101_2(pretrained=True, progress=True)
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Don't forget to put model in evaluation mode!
        # Use recommended sequence of transforms for ImageNet pretrained models
        self.transforms = Compose([Resize(256, interpolation=Image.BICUBIC),  # Default is bilinear
                                   CenterCrop(224),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        self.embeddings = Queue(self.log_every)  # Thread-safe
        self.model.avgpool.register_forward_hook(lambda m, m_in, m_out: self.embeddings.put(
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
        # If embeddings directory already exists, then delete it, otherwise create it
        if path.exists(path.join(self.data_dir, self.embed_dir)):
            rmtree(path.join(self.data_dir, self.embed_dir))
        makedirs(path.join(self.data_dir, self.embed_dir))

        batch = 0
        caption_indices = []
        with open(path.join(self.data_dir, self.tsvname), newline='') as tsvfile:
            tsv_reader = csv_reader(tsvfile, delimiter='\t')
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.embed_line, i, line)
                           for i, line in enumerate(tsv_reader)]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        caption_indices.append(result)
                        if len(caption_indices) == self.log_every:
                            tensors = []
                            for i in range(self.log_every):
                                tensors.append(self.embeddings.get())
                            batch += 1
                            with open(path.join(self.data_dir, self.embed_dir, f'{batch}.pickle'), 'wb') as outfile:
                                dump(zip(caption_indices, tensors),
                                     outfile, protocol=HIGHEST_PROTOCOL)
                            caption_indices = []
                            print(f'Saved {batch * self.log_every} images')
            # Save last incomplete batch if present
            if len(caption_indices) > 0:
                tensors = []
                remaining = 0
                while True:
                    try:
                        tensors.append(self.embeddings.get_nowait())
                        remaining += 1
                    except Empty:
                        break
                with open(path.join(self.data_dir, self.embed_dir, f'{batch + 1}.pickle'), 'wb') as outfile:
                    dump(zip(caption_indices, tensors),
                         outfile, protocol=HIGHEST_PROTOCOL)
                print(f'Saved {batch * self.log_every + remaining} images')


if __name__ == '__main__':
    parser = ArgumentParser(description="Generates 2048-dimensional embeddings for images from Google's Conceptual Captions dataset using a pretrained Wide ResNet-101-2 neural network on ImageNet. Must have CUDA in order to run. Note that this program will wipe the specified embedding sub-directory of the (specified) local data directory.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-t', '--tsvname', type=str, default='gcc_full.tsv',
                        help='filename in local data directory of combined, detokenized GCC dataset captions')
    parser.add_argument('-e', '--embed_dir', type=str, default='embeddings',
                        help='sub-directory of local data directory to save embedding dumps (with caption index) from GCC dataset')
    parser.add_argument('-w', '--timeout', type=float, default=1.0,
                        help="timeout in seconds for requests' GET method")
    parser.add_argument('-l', '--log_every', type=int, default=1024,
                        help='how many iterations to save embeddings and print status to stdout stream')
    args = parser.parse_args()
    model = Wide_ResNet_101_2(
        args.data_dir,
        args.tsvname,
        args.embed_dir,
        args.timeout,
        args.log_every)
    model.run()
