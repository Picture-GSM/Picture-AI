import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import Encoder, Decoder
from data_loader import Dataset
from sampler import InfiniteSamplerWrapper
from utils import calc_mean_std


def adjust_learning_rate(optimizer, iteration_count):
    lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser(description='Train style transfer')
parser.add_argument('--crop_size', default=256, type=int)
parser.add_argument('--max_iter', default=160000, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--num_worker', default=0, type=int)
parser.add_argument('--lr_decay', default=5e-5, type=float)

opt = parser.parse_args()

log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

crop_size = opt.crop_size
max_iter = opt.max_iter

transform = Compose([Resize(512), RandomCrop(256), ToTensor()])

style_set = Dataset('./dataset', transform=transform, style=True)
style_iter = iter(DataLoader(dataset=style_set,
                             num_workers=opt.num_worker,
                             batch_size=opt.batch_size,
                             sampler=InfiniteSamplerWrapper(style_set)))
content_set = Dataset('./dataset', transform=transform, style=False)
content_iter = iter(DataLoader(dataset=content_set,
                               num_workers=opt.num_worker,
                               batch_size=opt.batch_size,
                               sampler=InfiniteSamplerWrapper(content_set)))

out_path = './training_results'
if not os.path.exists(out_path):
    os.makedirs(out_path)

encoder = Encoder().to(device)
decoder = Decoder(encoder=encoder).to(device)

optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.lr)

loss_fn = nn.MSELoss()

model_path = './models'
if not os.path.exists(model_path):
    os.makedirs(model_path)

if os.listdir(model_path):
    decoder.load_state_dict(torch.load('models/decoder.pth'))

for i in tqdm(range(max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    style_feats, t = encoder(content_images, style_images)
    g_t_feats = decoder(t)

    loss_c = loss_fn(g_t_feats[-1], t)
    input_mean, input_std = calc_mean_std(g_t_feats[0])
    target_mean, target_std = calc_mean_std(style_feats[0])
    loss_s = loss_fn(input_mean, target_mean) + loss_fn(input_std, target_std)
    for i in range(1, 4):
        input_mean, input_std = calc_mean_std(g_t_feats[i])
        target_mean, target_std = calc_mean_std(style_feats[i])
        loss_s += loss_fn(input_mean, target_mean) + loss_fn(input_std, target_std)

    loss = loss_c + loss_s

    if i % 5000 == 0:
        plt.imshow(g_t_feats)
        plt.show()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % 10000 == 0:
        torch.save(decoder.state_dict(), model_path + '/decoder_%d.pth' % (i / 10000))