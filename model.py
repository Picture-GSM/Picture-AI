import torch
import torch.nn as nn

import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = models.vgg19(pretrained=True).features[:21]

    def forward(self, x):
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
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
        x = self.up9(x)

        return x


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