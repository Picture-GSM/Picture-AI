import torch
import torch.nn as nn

import torchvision.models as models
from utils import calc_mean_std


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        print('encoder')
        self.model = models.vgg19(pretrained=True).features[:21]
        self.enc_1 = nn.Sequential(*self.model[:4])
        self.enc_2 = nn.Sequential(*self.model[4:11])
        self.enc_3 = nn.Sequential(*self.model[11:18])
        self.enc_4 = nn.Sequential(*self.model[18:31])

    def extract_input_image(self, input):
        results = [input]
        for i in range(1, 5):
            func = getattr(self, 'enc_{:d}'.format(i))
            results.append(func(results[-1]))
        return results[1:]

    def encoder(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def forward(self, content, style, alpha=1.0):
        style_feats = self.extract_input_image(style)
        content_feats = self.encoder(content)
        t = AdaIN(content_feats, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feats
        return style_feats, t


class Decoder(nn.Module):
    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 256)
        self.up3 = UpBlock(256, 256)
        self.up4 = UpBlock(256, 256)
        self.up5 = UpBlock(256, 128)
        self.up6 = UpBlock(128, 128)
        self.up7 = UpBlock(128, 64)
        self.up8 = UpBlock(64, 64)
        self.up9 = UpBlock(64, 3, False)
        self.encoder = encoder

    def forward(self, x):
        x = self.up1(x)
        x = self.upsample(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.upsample(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.upsample(x)
        x = self.up8(x)
        save = self.up9(x)

        x = self.encoder.extract_input_image(save)

        return x, save


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True, kernel=3, pad=1):
        super(UpBlock, self).__init__()
        block = list()
        block += [nn.ReflectionPad2d(pad)]
        block += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel)]
        if relu:
            block += [nn.ReLU()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


def AdaIN(content, style):
    size = content.size()
    style_mean, style_std = calc_mean_std(style)
    content_mean, content_std = calc_mean_std(content)

    normalized = (content - content_mean.expand(size)) / content_std.expand(size)

    return normalized * style_std.expand(size) + style_mean.expand(size)



