# Keep imports lightweight
from PIL import Image
from glob import glob
from os import path
from torchvision.models import wide_resnet101_2
import torch
import torch.nn as nn
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
import torchvision.datasets as dataset
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import faiss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class faiss_embeddings_search:

    def __init__(self, img_dir: str, capt: str, embed: str) -> None:
        # Save parameters
        self.img_dir = img_dir
        with open(capt) as handle:
            self.capt = handle.readlines()
        print('Number of gcc captions =', len(self.capt))
        self.embed = np.load(embed)
        print('Number of gcc embeddings =', self.embed.shape[0])
        self.index = faiss.IndexFlatL2(2048)
        print(self.index.is_trained)
        self.index.add(self.embed)
        print(self.index.ntotal)
        # # Automatically use GPU if available
        # if not cuda.is_available():
        #     raise RuntimeError(
        #         'Must have CUDA installed in order to run this program.')
        # self.device = device('cuda')
        self.model = wide_resnet101_2(pretrained=True, progress=True)
        # # Move model to device
        # self.model.to(self.device)
        self.model.eval()  # Don't forget to put model in evaluation mode!
        self.model.fc = Identity()
        # Use recommended sequence of transforms for ImageNet pretrained models
        self.transforms = Compose([Resize(256, interpolation=Image.BICUBIC),  # Default is bilinear
                                   CenterCrop(224),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    def search_img_embed(self):
        for filename in glob(path.join(self.img_dir, '*.jpg')):  # assuming gif
            print(filename)
            # self.img_list.append(filename)
            image = Image.open(filename)
            image = self.transforms(image).unsqueeze(0)  # Fake batch-size of 1
            result = self.model(image)
            self.find_index(result.detach().cpu().numpy())

    def find_index(self, embedding, k=5):
        D, I = self.index.search(embedding, k)
        # print(self.img_list[-1])
        for i in I[0]:
            print(self.capt[i])
        print('banana')


if __name__ == '__main__':
    parser = ArgumentParser(description="Generates 2048-dimensional embeddings for images from Google's Conceptual Captions dataset using a pretrained Wide ResNet-101-2 neural network on ImageNet. Must have CUDA in order to run. Note that this program will wipe the specified embedding sub-directory of the (specified) local data directory.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-i', '--images_dir', type=str,
                        default='meme_templates_test', help='local data directory')
    parser.add_argument('-c', '--captions', type=str, default='data/gcc_captions.txt',
                        help='filename in local data directory of combined, detokenized GCC dataset captions')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings.npy',
                        help='filename for local embedding dumps (with caption index) from GCC dataset')

    args = parser.parse_args()
    img2cap = faiss_embeddings_search(
        args.image_dir,
        args.captions,
        args.embeddings)
    img2cap.search_img_embed()
