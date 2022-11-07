import torch
import os

from argparse import ArgumentParser
from model import Generator
from PIL import Image
from torchvision import transforms, utils
from torch import nn
from train import get_subspace
from tqdm import tqdm

def generate(model, result_path, n_samples, args):
    with torch.no_grad():
        model.eval()
        os.makedirs(result_path, exist_ok=True)
        if args.noise is not None:
            z_anch = torch.load(args.noise)
        for i in tqdm(range(n_samples), desc='Generating...'):
            if args.noise is None:
                noise = torch.randn(1, 512).cuda()
            else:
                noise = get_subspace(args, z_anch, size=1)
            fake_x, _ = model([noise], truncation=args.truncation, truncation_latent=None, input_is_latent=False, randomize_noise=False)
            utils.save_image(fake_x, os.path.join(result_path, '{}.png'.format(i)), normalize=True, value_range=(-1, 1))


if __name__=='__main__':
    argparser=ArgumentParser()
    argparser.add_argument('--checkpoint', type=str, required=True)
    argparser.add_argument('--result_path', type=str, required=True)
    argparser.add_argument('--exp_name', type=str,required=True)
    argparser.add_argument('--n_samples', type=int, default=1000)
    argparser.add_argument('--truncation', type=int,default=1)
    argparser.add_argument('--truncation_mean', type=int,default=4096)
    argparser.add_argument('--size', type=int, default=256)
    argparser.add_argument('--latent', type=int, default=512)
    argparser.add_argument('--n_mlp', type=int, default=8)
    argparser.add_argument('--channel_multiplier', type=int, default=2)
    argparser.add_argument('--subspace_std', type=float, default=0.1)
    argparser.add_argument('--noise', type=str, default=None)
    args = argparser.parse_args()
    torch.manual_seed(10)
    
    model = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).cuda()
    model.load_state_dict(torch.load(args.checkpoint)['g_ema'], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    generate(model, args.result_path, args.n_samples, args)