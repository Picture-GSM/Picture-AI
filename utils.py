import numpy as np
import torch

def calc_mean_std(feat, eps=1e-5):
    n, c, h, w = feat.shape
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)

    return feat_mean, feat_std
