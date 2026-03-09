#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    if mask is not None:
        mse = torch.sum((((img1 - img2)) ** 2) * mask) / torch.sum(mask)
    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def highpass_filter(img):
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=img.dtype, device=img.device).reshape(1, 1, 3, 3)
    if img.shape[1] > 1:
        kernel = kernel.repeat(img.shape[1], 1, 1, 1)
    return F.conv2d(img, kernel, padding=1, groups=img.shape[1])

def laplacian_of_gaussian_filter(kernel_size=5, sigma=1.0):
    """Creates a 2D Laplacian of Gaussian kernel."""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (sigma ** 4) * torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel - kernel.mean()  # normalize to zero mean
    kernel = kernel / kernel.abs().sum()  # normalize energy
    return kernel

def apply_log(img, kernel_size=5, sigma=1.0):
    """Artifact error based on Laplacian of Gaussian to detect unnatural sharp transitions."""
    kernel = laplacian_of_gaussian_filter(kernel_size, sigma).to(img.device)
    kernel = kernel.expand(img.shape[1], 1, kernel_size, kernel_size)  # for each channel
    return F.conv2d(img, kernel, padding=kernel_size // 2, groups=img.shape[1])

def artifact_sensitive_l1(img1, img2, mask=None):
    """
    img1, img2: [B, C, H, W]
    mask:      [B, 1, H, W] or None
    """
    def gradient(img):
        grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

        pad_x = torch.zeros_like(img)
        pad_x[:, :, :, :-1] = grad_x
        pad_y = torch.zeros_like(img)
        pad_y[:, :, :-1, :] = grad_y

        return pad_x + pad_y

    grad1 = gradient(img1)
    grad2 = gradient(img2)
    diff1 = torch.abs(grad1 - grad2)

    hp1 = highpass_filter(img1)
    hp2 = highpass_filter(img2)
    diff2 = torch.abs(hp1 - hp2)

    log1 = apply_log(img1)
    log2 = apply_log(img2)
    diff3 = torch.abs(log1 - log2)

    diff = diff3

    if mask is not None:
        # assert mask.shape[1] == 1, f"Mask should have shape [B, 1, H, W], now with {mask.shape}"
        loss = (diff * mask).sum() / mask.sum()
    else:
        loss = diff.mean()

    visualize_error_map(diff[0].mean(dim=0), title="Edge-aware Error Map")
    return loss

def visualize_error_map(error_map, title="Edge-aware Error Map", cmap="inferno"):
    """
    error_map: 2D torch tensor [H, W]
    """
    error_map_np = error_map.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(error_map_np, cmap=cmap)
    plt.colorbar(label="Edge Error")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()