import argparse
import math
import random
import os
import sys
from tkinter import E
from calc_inception import load_patched_inception_v3
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import viz
from copy import deepcopy
import numpy
import lpips

from PIL import Image


try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Extra
from model import Patch_Discriminator as Discriminator  # , Projection_head
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment

from datetime import datetime

import torchvision.transforms.functional as TF
from fid import calc_fid, extract_feature_from_samples
from intra_lpips import calc_intra_lpips
import pickle

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True, is_generator=False):
    for name, param in model.named_parameters():
        param.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True,
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def get_subspace(args, init_z, vis_flag=False, size=None):
    std = args.subspace_std
    if size is None:
        bs = args.batch if not vis_flag else args.n_sample
    else:
        bs = size
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema, device, g_source, d_source, current_task, dataset, inception, mean_latent):
    loader = sample_data(loader)

    imsave_path = os.path.join('samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # this defines the anchor points, and when sampling noise close to these, we impose image-level adversarial loss (Eq. 4 in the paper)
    init_z = torch.randn(args.n_train, args.latent, device=device)
    torch.save(init_z, os.path.join(model_path, f'{current_task}_noise.pt'))
    pbar = range(args.iter)
    sfm = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    sim = nn.CosineSimilarity()
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    # this defines which level feature of the discriminator is used to implement the patch-level adversarial loss: could be anything between [0, args.highp] 
    lowp, highp = 0, args.highp

    # the following defines the constant noise used for generating images at different stages of training
    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    requires_grad(g_source, False, is_generator=True)
    requires_grad(d_source, False, is_generator=False)
    sub_region_z = get_subspace(args, init_z.clone(), vis_flag=True)
    
    if args.source_latent is None:
        z = torch.randn(4096, 512, device='cuda')
        with torch.no_grad():
            w = g_source.style(z)
            mean_source_latent = w.mean(0, keepdim=True).unsqueeze(1).repeat(1, 14, 1)
    else:
        mean_source_latent = torch.load(args.source_latent).cuda() # [1, 512]
    latent_ckpt = torch.load(f'stylegan2_inversion_wplus/{current_task}/result_file.pt')
    real_target_latents = [latent_ckpt[i]['latent'] for i in range(10)]
    real_target_latents = torch.stack(real_target_latents, dim=0) # [10, 512]
    mean_target_latent = real_target_latents.mean(0).unsqueeze(0) # [1, 512]
    mean_source_latent = mean_source_latent.detach()
    mean_target_latent = mean_target_latent.detach()
    real_target_latents = real_target_latents.detach()
    
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()

    for idx in pbar:
        i = idx + args.start_iter
        which = i % args.subspace_freq # defines whether we sample from anchor region in this iteration or other

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)
        requires_grad(generator, False, is_generator=False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)
        if which > 0:
            # sample normally, apply patch-level adversarial loss
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            # sample from anchors, apply image-level adversarial loss
            noise = [get_subspace(args, init_z.clone())]

        fake_img, _ = generator(noise)

        if args.augment:
            real_img, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        real_pred, _ = discriminator(
            real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp), real=True)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        extra.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()
        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss
        requires_grad(generator, True, is_generator=True)
        requires_grad(discriminator, False)
        requires_grad(extra, False)
        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = [get_subspace(args, init_z.clone())]

        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        g_loss = g_nonsaturating_loss(fake_pred)
        
        # distance loss
        z = torch.randn(4, 512, device=device)
        with torch.no_grad():
            _, source_feats = g_source([z], return_feats=True)
            _, mean_source_feats = g_source([mean_source_latent], input_is_latent=True, return_feats=True)
        _, target_feats = generator([z], return_feats=True)
        _, mean_target_feats = generator([mean_target_latent], input_is_latent=True, return_feats=True)
        f1 = [] # source content, source style
        f2 = [] # source content, target style
        f3 = [] # target content, target style
        for source_feat, target_feat, mean_source_feat, mean_target_feat in zip(source_feats, target_feats, mean_source_feats, mean_target_feats):
            f1.append(adaptive_instance_normalization(source_feat, mean_source_feat.repeat(4, 1, 1, 1)))
            f2.append(adaptive_instance_normalization(source_feat, mean_target_feat.repeat(4, 1, 1, 1)))
            f3.append(adaptive_instance_normalization(target_feat, mean_target_feat.repeat(4, 1, 1, 1)))
            
        num_layers = len(f1)
            
        style_loss = 0
        content_loss = 0
        for l in range(num_layers):
            style_loss = style_loss + (1 - F.cosine_similarity(f1[l].view(4, -1), f2[l].view(4, -1), dim=1)).mean()
            content_loss = content_loss + (1 - F.cosine_similarity(f2[l].view(4, -1), f3[l].view(4, -1), dim=1)).mean()
        style_loss = style_loss / num_layers
        content_loss = content_loss / num_layers
            
        new_loss_g = args.lambda_style * style_loss + args.lambda_content * content_loss
            

        g_loss = g_loss + new_loss_g
        ##########
        loss_dict["g"] = g_loss
        if isinstance(new_loss_g, int) and new_loss_g == 0:
            loss_dict['new_loss_g'] = 0
        else:
            loss_dict["new_loss_g"] = new_loss_g

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        g_regularize = i % args.g_reg_every == 0

        # to save up space
        # del rel_loss, g_loss, d_loss, fake_img, fake_pred, real_img, real_pred, anchor_feat, compare_feat, dist_source, dist_target, feat_source, feat_target
        # del g_loss, d_loss, fake_img, fake_pred, real_img, real_pred
        del g_loss, d_loss, fake_img, fake_pred, real_pred
        
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

       
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        new_loss_g_val = loss_reduced["new_loss_g"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"Task: {current_task}; "
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}; new_loss_g: {new_loss_g_val:.4f};"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        "New Loss (G)": new_loss_g_val
                    }
                )
            if i % args.img_freq == 0:
                with torch.set_grad_enabled(False):
                    generator.eval()
                    sample, _ = generator([sample_z.data])
                    sample_subz, _ = generator([sub_region_z.data])
                    utils.save_image(
                        sample,
                        f"%s/{current_task}_{str(i).zfill(6)}.png" % (imsave_path),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    del sample
                    generator.train()
            if (i % args.save_freq == 0) and (i>0):
                torch.save(
                    {
                        "g_ema": generator.state_dict(),
                        # uncomment the following lines only if you wish to resume training after saving. Otherwise, saving just the generator is sufficient for evaluations

                        #"g": g_module.state_dict(),
                        #"g_s": g_source.state_dict(),
                        #"d": d_module.state_dict(),
                        #"g_optim": g_optim.state_dict(),
                        #"d_optim": d_optim.state_dict(),
                    },
                    f"{model_path}/{current_task}_{str(i).zfill(6)}.pt",
                )
            
            # Compute metrics
            if (i % 1000 == 0):
                with torch.no_grad():
                    features = extract_feature_from_samples(generator, inception, args.truncation, mean_latent, batch_size=64, n_sample=5000, device=device)
                    features = features.numpy()
                    
                sample_mean = np.mean(features, 0)
                sample_cov = np.cov(features, rowvar=False)
                with open(args.inception, "rb") as f:
                    embeds = pickle.load(f)
                    real_mean = embeds["mean"]
                    real_cov = embeds["cov"]
                
                fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                
                wandb.log({'fid (inception-v3)': fid})
                print(f'FID (Inception-v3): {fid:.4f}')
                

def get_params(model):
    weight_list = []
    for name, param in model.named_parameters():
        if name.find('conv.weight') >= 0:
            if name.find('conv.weight_modulation') >= 0:
                weight_list.append(param)
            else:
                continue
        else:
            weight_list.append(param)
    
    return weight_list

def load_model_norm(model):
    th_m = torch.tensor(1e-5)
    stdd = 1.0
    dict_all = model.state_dict()
    model_dict = model.state_dict()
    for k, v in dict_all.items():
        # print(k)
        if k.find('conv.weight') >= 0:
            if k.find('conv.weight_modulation') >= 0:
                continue
            else:
                # print("Key: ", k)
                # print("Value: ", v)
                w_mu = v.mean([3, 4], keepdim=True)
                w_std = v.std([3, 4], keepdim=True) * stdd
                # print(w_mu.size())
                # print(w_std.size())
                dict_all[k].data = (v - w_mu)/(w_std + th_m)
                dict_all[k + '_modulation.style_gama'].data = w_std.data
                dict_all[k + '_modulation.style_beta'].data = w_mu.data

    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iter", type=int, default=5002)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--img_freq", type=int, default=500)
    parser.add_argument("--kl_wt", type=int, default=1000)
    parser.add_argument("--highp", type=int, default=1)
    parser.add_argument("--subspace_freq", type=int, default=4)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_res", type=int, default=128)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--subspace_std", type=float, default=0.1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--source_key", type=str, default='ffhq')
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", dest='augment', action='store_true')
    parser.add_argument("--no-augment", dest='augment', action='store_false')
    parser.add_argument("--augment_p", type=float, default=0.0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--n_train", type=int, default=10)
    #added
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--lambda_contrast', type=float, default=0.001)
    parser.add_argument('--source_latent', type=str, default=None)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--lambda_style', type=float, default=1)
    parser.add_argument('--lambda_content', type=float, default=1)

    args = parser.parse_args()
    if args.data_path[-1] == '/':
        args.data_path = args.data_path[:-1] # processed_data/Sketches/10shot/1
    args.task = "_".join(args.data_path.split('/')[-3:])
    torch.manual_seed(1)
    random.seed(1)

    inception = load_patched_inception_v3().to(device)
    inception.eval()

    n_gpu = 4
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    d_source = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    extra = Extra().to(device)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    module_source = ['landscapes', 'red_noise',
                     'white_noise', 'hands', 'mountains', 'handsv2']
    
    for name, param in generator.named_parameters():
        param.requires_grad = True
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass


        generator.load_state_dict(ckpt["g"], strict=False)
        g_source.load_state_dict(ckpt_source["g"], strict=False)

        # d_source = nn.parallel.DataParallel(d_source)
        # discriminator = nn.parallel.DataParallel(discriminator)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        d_source.load_state_dict(ckpt_source["d"])

        # if 'g_optim' in ckpt.keys():
        #     g_optim.load_state_dict(ckpt["g_optim"])
        # if 'd_optim' in ckpt.keys():
        #     d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    args.inception = f"./inception_{args.task.split('_')[0]}_full.pkl"
    
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = generator.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    dataset = MultiResolutionDataset(args.data_path, transform, args.size)
    loader = data.DataLoader(dataset, batch_size=args.batch, sampler=data_sampler(dataset, shuffle=True, distributed=False), drop_last=True, num_workers=2)
    
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.exp, name=args.task)

    s = 0
    entire_s = 0
    for name, param in generator.named_parameters():
        if param.requires_grad:
            print(name)
            s += torch.numel(param)
        if 'weight_modulation' not in name:
            entire_s += torch.numel(param)
    print('(Generator) Trainable parameters:', s)
    print('(Generator) Entire parameters:', entire_s)
    print("=" * 100)
    s = 0
    entire_s = 0
    for name, param in discriminator.named_parameters():
        if param.requires_grad:
            print(name)
            s += torch.numel(param)
        if 'weight_modulation' not in name:
            entire_s += torch.numel(param)
    print('(Discriminator) Trainable parameters:', s)
    print('(Discriminator) Entire parameters:', entire_s)
    print("=" * 100)
    train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, None, device, g_source, d_source, args.task, dataset, inception, mean_latent)