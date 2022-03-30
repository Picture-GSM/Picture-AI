import numpy as np
import torch


def calc_mean_std(feat, eps=1e-5):
    n, c, h, w = feat.shape
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)

    return feat_mean, feat_std


def _calc_feat_flatten_mean_std(feat):
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)

    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.t() + torch.eye(3))

    source_f_norm_transfer = torch.mm(_mat_sqrt(target_f_cov_eye),
                                      torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)), source_f_norm))

    source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())