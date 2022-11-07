import argparse
import random
import torch
import os
import sys
import lpips

import torch.nn as nn
import numpy as np

from torchvision import utils, transforms
from tqdm import tqdm
from PIL import Image
from dataset import MultiResolutionDataset
from torch.utils.data import DataLoader

def intra_cluster_dist(args):
    lpips_fn = lpips.LPIPS(net='vgg').to(args.device)

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    real_dataset = MultiResolutionDataset(path=args.real_path, transform=transform, resolution=256)
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)
    fake_images = os.listdir(args.fake_path)
    cluster = {}
    for i in range(10):
        cluster[i] = []
    # clustering
    pbar = tqdm(fake_images, desc='Clustering...')
    for fake_image_path in pbar:
        image_path = os.path.join(args.fake_path, fake_image_path)
        fake_image = Image.open(image_path)
        fake_tensor = transform(fake_image).to(args.device)
        dists = []
        for i, real_tensor in enumerate(real_loader):
            real_tensor = real_tensor.to(args.device)
            dist = lpips_fn(fake_tensor, real_tensor)
            dists.append(dist.item())
        closest_cluster = np.argmin(dists)
        cluster[closest_cluster].append(os.path.join(args.fake_path, fake_image_path))

    # compute average pairwise distance
    print('Clustered as :', [len(cluster[c]) for c in cluster])
    dists = []
    cluster_size = args.cluster_size
    cluster = {c: cluster[c][:cluster_size] for c in cluster}
    total_length = sum([len(cluster[c]) * (len(cluster[c]) - 1) for c in cluster]) // 2
    with tqdm(range(total_length), desc='Computing...') as pbar:
        for c in cluster:
            temp = []
            cluster_length = len(cluster[c])
            for i in range(cluster_length):
                img1 = Image.open(cluster[c][i])
                img1 = transform(img1).cuda()
                for j in range(i + 1, cluster_length):
                    img2 = Image.open(cluster[c][j])
                    img2 = transform(img2).cuda()
                    pairwise_dist = lpips_fn(img1, img2)
                    temp.append(pairwise_dist.item())
                    pbar.update(1)
            dists.append(np.mean(temp))
    dists = np.array(dists)
    print(f'Intra-Cluster LPIPS : {dists[~np.isnan(dists)].mean():.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str, required=True) # processed_data/Sketches/10shot/0
    parser.add_argument('--fake_path', type=str, required=True) # fake_images/LFS/Sketches_Sketches_5000
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cluster_size', type=int, default=50)
    args = parser.parse_args()
    intra_cluster_dist(args)