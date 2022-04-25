import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile


from model import Model, AdaIN


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


model = ConvertModel()

dummy_input = torch.rand(1, 3, 512, 512)

traced_model = torch.jit.trace(model, (dummy_input, dummy_input))
optimized_model = optimize_for_mobile(traced_model)

optimized_model.save('style_transfer.pt')

model_load = torch.jit.load('style_transfer.pt')
