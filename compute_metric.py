# The purpose of this code is to compute the metrics of generated samples. Here we include PSNR, SSIM, LPIPS and FID.
# Before running the code, please remember to modify the folder name and task name.
# To start running, the command is simply: python3 compute_metric.py

from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from os import listdir

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

import PIL
import torchvision.transforms.functional as transform
import torchvision.utils as tvu
import torchvision.transforms as transforms

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image.fid import FrechetInceptionDistance


device = 'cuda:0'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

task = 'inpainting'
nums_samples_start = 0
num_samples = 1000

label_root = Path(f'./results/{task}/label')
normal_recon_root = Path(f'./results/{task}/recon')

psnr_normal_list = []
lpips_normal_list = []
ssim_normal_list = []
fid = FrechetInceptionDistance(feature=2048).to(device)

for idx in tqdm(range(nums_samples_start, num_samples)):
    fname = str(idx).zfill(5)

    label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]
    psnr_normal = peak_signal_noise_ratio(label, normal_recon)

    psnr_normal_list.append(psnr_normal)
    
    normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)
    
    fid_recon = (normal_recon.view(1, 3, 256, 256) * 255).to(dtype=torch.uint8)
    fid_label = (label.view(1, 3, 256, 256) * 255).to(dtype=torch.uint8)
    normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.
    normal_d = loss_fn_vgg(normal_recon, label)
    lpips_normal_list.append(normal_d)
    fid.update(fid_label, real=True)
    fid.update(fid_recon, real=False)

psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)
fid_score = fid.compute().item()

print(f'Normal PSNR: {psnr_normal_avg}')
print(f'Normal LPIPS: {lpips_normal_avg}')
print(f'Normal FID: {fid_score}')

with torch.no_grad():
    for idx in tqdm(range(nums_samples_start, num_samples)):
        fname = str(idx).zfill(5)
        PIL_image = PIL.Image.open(label_root / f'{fname}.png')
        orig = transform.to_tensor(PIL_image)[:3, :, :].cuda()
        
        PIL_image = PIL.Image.open(normal_recon_root / f'{fname}.png')
        recon = transform.to_tensor(PIL_image)[:3, :, :].cuda()
        
        orig = orig.reshape(1, *orig.shape)
        recon = recon.reshape(1, *recon.shape)
        ssim_d =  ssim(orig, recon)
        ssim_normal_list.append(ssim_d)

ssim_normal_avg = sum(ssim_normal_list) / len(ssim_normal_list)
print(f'Normal SSIM: {ssim_normal_avg}')
