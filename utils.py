import numpy as np


def calc_mean_std(feat, eps=1e-5):
    n, c, h, w = feat.shape
    feat_var = np.var(feat.reshape(n, c, -1), axis=2) + eps
    feat_std = np.sqrt(feat_var).reshape(n, c, 1, 1)
    feat_mean = np.mean(feat.reshape(n, c, -1), axis=2).reshape(n, c, 1, 1)

    return feat_mean, feat_std