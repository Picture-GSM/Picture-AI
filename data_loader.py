import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize


class Dataset(Dataset):
    def __init__(self, path, transformS=None, transformC=None):
        super(Dataset, self).__init__()
        self.path = path
        self.style_files = [x for x in os.listdir(os.path.join(path, 'style'))]
        self.content_files = [x for x in os.listdir(os.path.join(path, 'content'))]
        self.transformS = transformS
        self.transformC = transformC
        self.totensor = ToTensor()

        print(self.style_files)

    def __len__(self):
        return min(len(self.style_files), len(self.content_files))

    def __getitem__(self, index):
        style_img = plt.imread(os.path.join(self.path + '/style', self.style_files[index]))
        content_img = plt.imread(os.path.join(self.path + '/content', self.content_files[index]))

        '''if (style_img.dtype == np.uint8) and (content_img.dtype == np.uint8):
            style_img /= 255.0
            content_img /= 255.0'''

        if self.transformS:
            style_img = self.transform(style_img)
        if self.transformC:
            content_img = self.transformC(content_img)

        style_img = ToTensor()(style_img)
        content_img = ToTensor()(content_img)

        data = {'style': style_img, 'content': content_img}
        return data


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class CenterCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.shape

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        h, w = data['label'].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))

        return data