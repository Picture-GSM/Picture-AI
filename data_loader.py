import os
from os import listdir
from os.path import join
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize


class Dataset(Dataset):
    def __init__(self, path, transform=None, style=None):
        super(Dataset, self).__init__()
        self.path = path
        self.style = style
        if self.style:
            self.path = self.path + '/style'
            self.files = [join(self.path, x) for x in listdir(self.path)]
            print(self.path)
        else:
            self.path = self.path + '/content'
            self.files = [join(self.path, x) for x in listdir(self.path)]
            print(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        img = self.transform(img)

        return img