import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import occamypy as o
import eikonal2d as k
import scipy.io as scio
from PIL import Image
import tqdm

import os
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image
from utils import restore_checkpoint

from models import utils as mutils
from models import ddpm as ddpm_model

import sampling
from sde_lib import VESDE, VPSDE, subVPSDE
import datasets

from configs.vp import marmousi_ddpm_continuous as configs
config = configs.get_config()

if config.training.sde.lower() == 'vpsde':
   ckpt_filename = "/home/caoxiang/Desktop/diffusion_model/score_sde_pytorch_training/workdir/seismic/128/checkpoints/checkpoint_7.pth"
   sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
   sampling_eps = 1e-5
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

# device 
device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')

# create basic model
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

# create model
score_model = ddpm_model.DDPM(config)
score_model = score_model.to(device)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, device)
ema.copy_to(score_model.parameters())

score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous= True)

input_eval_dir = "/home/caoxiang/Desktop/subspace_posterior_sampling/score_sde_for_seismic_DPS/eval_folder/vertical_varying_stepsize"
sample_dir = "/home/caoxiang/Desktop/subspace_posterior_sampling/score_sde_for_seismic_DPS/denoised_eval_folder/vertical_varying_stepsize"

for idx in range(400):
    print(idx)
    image = Image.open(os.path.join(input_eval_dir, "sample_"+str(idx)+".png")).convert('L')  # 'L' mode means single-channel (grayscale)
    image = transforms.Compose([
                transforms.Resize((128, 128)),  # 调整图片大小
                transforms.ToTensor(),  # 将图片转换为PyTorch张量
                ])(image).unsqueeze(0).to(device)

    f_image = scaler(image)

    t = torch.tensor([0.02]).to(device)
    timestep = (t * (sde.N - 1) / sde.T).long()
    score = score_fn(f_image, t)

    # Compute posterior mean for x_0_hat 
    sqrt_alphas_cumprod_t = sde.sqrt_alphas_cumprod.to(t.device)[timestep]
    alphas_cumprod_t = sde.alphas_cumprod.to(t.device)[timestep]
    coef_1 = (1.0/sqrt_alphas_cumprod_t).to(t.device)
    coef_2 = ((1.0 - alphas_cumprod_t)/sqrt_alphas_cumprod_t).to(t.device)

    f_image_0_hat = coef_1[:, None, None, None] * f_image + coef_2[:, None, None, None] * score
    
    # Clean Image
    image_clear = inverse_scaler(f_image_0_hat)

    # Save Image
    this_sample_dir = os.path.join(sample_dir, "sample_"+str(idx)+".png")
    nrow = int(np.sqrt(image_clear.shape[0]))
    image_grid = make_grid(image_clear, nrow, padding=2)

    with open(this_sample_dir, "wb") as fout:
        save_image(image_grid, fout)
