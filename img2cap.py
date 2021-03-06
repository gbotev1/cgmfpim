# Keep imports lightweight
from PIL import Image
from glob import glob
from os import path
from torchvision.models import wide_resnet101_2
from torch.nn import Module
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import faiss


class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Wide_ResNet_101_2:

    def __init__(self, args: Namespace, dim: int = 2048) -> None:
        self.data_dir = args.data_dir
        self.images_dir = args.images_dir
        with open(path.join(args.data_dir, args.captions)) as infile:
            self.captions = infile.readlines()
        self.embeddings = np.load(path.join(args.data_dir, args.embeddings))
        self.k = k
        self.metric = metric

        if self.metric == -1:
            # Cosine similarity
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        elif self.metric == 1:
            # Euclidean distance (no square root)
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
        elif self.metric == 23:
            # Mahalanobis distance
            self.index = faiss.IndexFlatL2(dim)
            x_centered = self.embeddings - self.embeddings.mean(0)
            self.transform = np.linalg.inv(np.linalg.cholesky(
                np.dot(x_centered.T, x_centered) / x_centered.shape[0])).T
            self.index.add(
                np.dot(self.embeddings, self.transform).astype(np.float32))
        elif self.metric == 0:
            # Inner project
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = faiss.IndexFlat(dim, self.metric)
            self.index.add(self.embeddings)

        self.model = wide_resnet101_2(pretrained=True, progress=True)
        self.model.eval()  # Don't forget to put model in evaluation mode!
        self.model.fc = Identity()
        # Use recommended sequence of transforms for ImageNet pretrained models
        self.transforms = Compose([Resize(256, interpolation=Image.BICUBIC),  # Default is bilinear
                                   CenterCrop(224),
                                   ToTensor(),
                                   Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    def run(self) -> None:
        for filename in glob(path.join(self.data_dir, self.images_dir, '*.jpg')):
            print(filename)
            image = self.transforms(Image.open(filename)).unsqueeze(
                0)  # Fake batch-size of 1
            embed = self.model(image).detach().numpy()
            if self.metric == -1:
                faiss.normalize_L2(embed)
            elif self.metric == 23:
                embed = np.dot(embed, self.transform).astype(np.float32)
            D, I = self.index.search(embed, self.k)
            for i in range(len(I[0])):
                print(self.captions[I[0][i]])
                print(D[0][i])
            print()  # For spacing


if __name__ == '__main__':
    parser = ArgumentParser(description='Helper script to embed input images and find closest captions in GCC database under specified metrics using the same pretrained Wide ResNet-101-2 from the img2vec.py script.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', type=str,
                        default='data', help='local data directory')
    parser.add_argument('-i', '--images_dir', type=str,
                        default='meme_templates_test', help='sub-directory of images in local data directory')
    parser.add_argument('-c', '--captions', type=str, default='gcc_captions.txt',
                        help='filename for textfile containing GCC captions')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings.npy',
                        help='filename for embeddings NumPy archive')
    parser.add_argument('-k', type=int, default=5,
                        help='nearest k neighbors to search for in GCC database')
    parser.add_argument('-m', '--metric', type=int, default=1,
                        help='FAISS metric type to use')
    img2cap = Wide_ResNet_101_2(parser.parse_args())
    img2cap.run()
