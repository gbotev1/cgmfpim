# Keep imports lightweight
from PIL import Image
from torchvision.models import wide_resnet101_2
from os import path, makedirs
from os import remove as os_remove
from shutil import rmtree
from csv import reader as csv_reader
from torch import device, cuda, Tensor
from torch import stack as torch_stack
import torchvision.transforms as T
from requests import get as requests_get
from io import BytesIO
from numpy import save, vstack
from numpy import stack as np_stack
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Optional, Tuple


class Wide_ResNet_101_2:

    def __init__(self, data_dir: str, tsvname: str, out_dir: str, captions_index: str, timeout: float, log_every: int, batch_size: int) -> None:
        # Save parameters
        self.data_dir = data_dir
        self.tsvname = tsvname
        self.out_dir = out_dir
        self.captions_index = captions_index
        self.timeout = timeout
        self.log_every = log_every
        self.batch_size = batch_size
        # Pipeline set-up
        self.model = wide_resnet101_2(pretrained=True, progress=True)
        # Automatically use GPU if available
        self.has_cuda = cuda.is_available()
        # Move model to device
        self.model.to(device('cuda' if self.has_cuda else 'cpu'))
        self.model.eval()  # Don't forget to put model in evaluation mode!
        # Transform all images to be minimum allowed square model size for generalizability, efficiency, and batching
        self.transforms = T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC),  # Use bicubic interpolation for best quality
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])  # Recommended normalization for torchvision ImageNet pretrained models
        self.embeddings = []
        self.model.avgpool.register_forward_hook(lambda m, m_in, m_out: self.embeddings.append(
            m_out.data.detach().cpu().squeeze().numpy()))

    def get_tensor(self, i: int, line: List[str]) -> Optional[Tuple[Tensor, int]]:
        try:
            r = requests_get(line[1], stream=True, timeout=self.timeout)
            image = Image.open(BytesIO(r.content))
            return self.transforms(image), i
        except:
            return None

    def embed_line(self, i: int, line: List[str]) -> Optional[int]:
        try:
            r = requests_get(line[1], stream=True, timeout=self.timeout)
            image = Image.open(BytesIO(r.content))
            image = self.transforms(image).unsqueeze(0)  # Fake batch-size of 1
            self.model(image)
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
        iters = 0
        batches = 0
        caption_indices = []
        if self.has_cuda:
            tensors = []
            with open(path.join(self.data_dir, self.tsvname), newline='') as tsvfile:
                tsv_reader = csv_reader(tsvfile, delimiter='\t')
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(
                        self.get_tensor, i, line) for i, line in enumerate(tsv_reader)]
                    for future in as_completed(futures):
                        iters += 1
                        result = future.result()
                        if result is not None:
                            tensors.append(result[0])
                            caption_indices.append([result[1]])
                            if len(tensors) == self.batch_size:
                                # Run batch through GPU
                                batch = torch_stack(tensors)
                                # Must have GPU in this branch
                                self.model(batch.to(device('cuda')))
                                del batch  # Free up GPU memory
                                # Reset batch when done
                                tensors = []
                        if iters % self.log_every == 0:
                            # Save batch
                            save(path.join(self.data_dir, self.out_dir,
                                           f'{batches}.npy'), vstack(self.embeddings))
                            # Reset embeddings to ease memory pressure
                            self.embeddings = []
                            batches += 1
                            print(iters)
            # Save caption indices
            save(path.join(self.data_dir, self.captions_index), caption_indices)
            # Check if incomplete batch present
            if len(tensors) > 0:
                batch = torch_stack(tensors)
                # Must have GPU in this branch
                self.model(batch.to(device('cuda')))
                del batch, tensors  # Might as well
                # Save incomplete batch
                save(path.join(self.data_dir, self.out_dir,
                               f'{batches}.npy'), vstack(self.embeddings))
                self.embeddings = []  # Might as well
        else:
            with open(path.join(self.data_dir, self.tsvname), newline='') as tsvfile:
                tsv_reader = csv_reader(tsvfile, delimiter='\t')
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(
                        self.embed_line, i, line) for i, line in enumerate(tsv_reader)]
                    for future in as_completed(futures):
                        iters += 1
                        result = future.result()
                        if result is not None:
                            caption_indices.append(result)
                        if iters % self.log_every == 0:
                            print(iters)
            save(path.join(self.data_dir, self.captions_index), caption_indices)
            save(path.join(self.data_dir, 'embeddings.npy'),
                 np_stack(self.embeddings))


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
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='GPU batch size to use if CUDA/GPU is available')
    args = parser.parse_args()
    model = Wide_ResNet_101_2(
        args.data_dir, args.tsvname, args.out_dir, args.captions_index, args.timeout, args.log_every, args.batch_size)
    model.run()
