import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils import coral
from model import Encoder, Decoder, AdaIN


def style_transfer(encoder, decoder, content, style, alpha=1.0):
    content_f = encoder(content)
    style_f = encoder(style)
    feat = AdaIN(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def transformer(size):
    transform_list = list()
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', type=str, default='./input/content')
parser.add_argument('--style_dir', type=str, default='./input/style')
parser.add_argument('--decoder', type=str, default='./models/decoder_3.pth')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='0에 가까울 수록 스타일 이미지가 많이 적용 됩니다.')
parser.add_argument('--preserve_color', type=bool, default=False)

parser.add_argument('--content_size', type=int, default=512)
parser.add_argument('--style_size', type=int, default=512)
parser.add_argument('--save_ext', type=str, default='.jpg')
parser.add_argument('--output', type=str, default='./output')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

if args.content_dir:
    content_paths = [f for f in Path(args.content_dir).glob('*')]

if args.style_dir:
    style_paths = [f for f in Path(args.style_dir).glob('*')]

encoder = Encoder()
decoder = Decoder()

decoder.load_state_dict(torch.load(args.decoder, map_location=device))

encoder.to(device)
decoder.to(device)

c_transform = transformer(args.content_size)
s_transform = transformer(args.style_size)

for content_path in content_paths:
    for style_path in style_paths:
        content = c_transform(Image.open(str(content_path)).convert('RGB'))
        style = s_transform(Image.open(str(style_path)).convert('RGB'))
        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(encoder, decoder, content, style, args.alpha)
        output = output.cpu()

        output_name = output_dir / '{:s}_stylized_{:s}_3_06.png'.format(
            content_path.stem, style_path.stem
        )
        save_image(output, str(output_name))