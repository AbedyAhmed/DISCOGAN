import os
import torch
import torchvision.utils as vutils

def save_sample(a, b, ab, ba, out_dir, step):
    os.makedirs(out_dir, exist_ok=True)
    grid = torch.cat([a, ab, b, ba], dim=0)
    grid = vutils.make_grid(grid, nrow=max(1, a.size(0)), normalize=True, value_range=(-1,1))
    vutils.save_image(grid, os.path.join(out_dir, f'step_{step:07d}.png'))

def save_ckpt(state, out_dir, name='last.pt'):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, name))
