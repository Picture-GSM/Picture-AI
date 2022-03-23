# This code was taken from:https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py
# by naoto0804

import numpy as np
from torch.utils import data


def InfiniteSampler(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, dataset):
        self.num = dataset.__len__()

    def __iter__(self):
        return iter(InfiniteSampler(self.num))

    def __len__(self):
        return 2 ** 31