import torch
import lpips

import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MultiResolutionDataset

def calc_intra_lpips(fake_images, real_images_path, cluster_size=50, device='cuda'):
    assert len(fake_images.shape) == 4
    assert isinstance(real_images_path, str)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    real_dataset = MultiResolutionDataset(path=real_images_path, transform=transform, resolution=256)
    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)
    
    cluster = {}
    for i in range(10):
        cluster[i] = []
    
    b, _, _, _ = fake_images.shape
    for i in range(b):
        dists = []
        for real_image in real_loader:
            real_image = real_image.to(device)
            with torch.no_grad():
                dist = lpips_fn(fake_images[i, :, :, :].unsqueeze(0), real_image)
                dists.append(dist.item())
        closest_cluster = np.argmin(dists)
        cluster[closest_cluster].append(i)
        
    dists = []
    cluster = {c: cluster[c][:cluster_size] for c in cluster}
    for c in cluster:
        temp = []
        cluster_length = len(cluster[c])
        for i in range(cluster_length):
            img1 = fake_images[cluster[c][i], :, :, :].unsqueeze(0)
            for j in range(i + 1, cluster_length):
                img2 = fake_images[cluster[c][j], :, :, :].unsqueeze(0)
                with torch.no_grad():
                    pairwise_dist = lpips_fn(img1, img2)
                    temp.append(pairwise_dist.item())
        dists.append(np.mean(temp))
    dists = np.array(dists)
    return dists[~np.isnan(dists)].mean()
    