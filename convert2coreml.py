import coremltools as ct
import urllib
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from model import Model, AdaIN


to_tensor = transforms.ToTensor()

input_image_content = Image.open('./input/content/modern.png').convert('RGB')
input_image_style = Image.open('./input/style/style2.png').convert('RGB')

input_tensor_content = to_tensor(input_image_content).unsqueeze(0)
input_tensor_style = to_tensor(input_image_style).unsqueeze(0)


class ConvertModel(nn.Module):
    def __init__(self):
        super(ConvertModel, self).__init__()
        self.model = Model()
        self.model.decoder.state_dict(torch.load('./models/decoder_3.pth', map_location='cpu'))
        self.model.encoder

    def forward(self, content, style, alpha=0.6):
        content_f = self.model.encoder(content)
        style_f = self.model.encoder(style)
        feat = AdaIN(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return self.model.decoder(feat)


model = ConvertModel().eval()

trace = torch.jit.trace(model, (input_tensor_content, input_tensor_style))

# bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225], scale=1./(0.226*255.0)
input_1 = ct.ImageType(name='input_1', shape=input_tensor_content.shape)
input_2 = ct.ImageType(name='input_2', shape=input_tensor_style.shape)

model = ct.convert(trace, inputs=[input_1, input_2])

model.type = 'styleTransfer'
model.save('styleTransfer.mlmodel')