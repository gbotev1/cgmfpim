# Keep imports lightweight
from PIL import Image
from torchvision.models import wide_resnet101_2
from os import listdir
import torch
import torchvision.transforms as T
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class Wide_ResNet_101_2:

    def __init__(self):
        self.model = wide_resnet101_2(pretrained=True, progress=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)  # Move model to device
        self.model.eval()  # Don't forget to put model in evaluation mode!
        self.transforms = T.Compose([T.Resize((224, 224), interpolation=Image.BICUBIC), 
                                    T.ToTensor(), 
                                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])
        self.outputs = []
        self.model.avgpool.register_forward_hook(lambda m, m_in, m_out: self.outputs.append(m_out.data.detach().cpu().squeeze().numpy()))

    def load(self, dir: str, batch_size: int):
        imgs = []
        for i, filename in enumerate(listdir(dir)):
            imgs.append(Image.open(filename))
            if i % batch_size == 0:
                self.transforms(imgs)


    def img2vec(self, dir: str, batch_size: int):
        img = Image.open(name)
        img = self.transforms(img)
        self.model(img.unsqueeze(0))
        return self.outputs


if __name__ == "__main__":
  parser = ArgumentParser(description='img2vec', formatter_class=ArgumentDefaultsHelpFormatter)
  args = parser.parse_args()
  temp = Img2VecResNet()
  dog1 = transforms(Image.open('dog.jpg'))

  print(temp.img2vec())

