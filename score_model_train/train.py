import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

import losses
import datasets
import sde_lib
import sampling

from models import ddpm as ddpm_model
from models.ema import ExponentialMovingAverage

from configs.vp import AI4Scup2_ddpm_continuous as configs

# Set Config
config = configs.get_config()
base_resolution = config.data.image_size
resolution = 64

times = int(np.log2(base_resolution/resolution))
workdir = "workdir/AI4Scup2/" + str(resolution)
config.data.image_size = resolution

# downsampling/upsampling
def downsample(x):
    return torch.nn.AvgPool2d(2, stride=2, padding=0)(x)

def repeat(func, x, n):
    for _ in range(n):
        x = func(x)
    return x

# Create Logger 
os.makedirs(workdir, exist_ok=True)
gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
handler = logging.StreamHandler(gfile_stream)
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel('INFO')

# Create directories for experimental logs
sample_dir = os.path.join(workdir, "samples")
os.makedirs(sample_dir, exist_ok=True)

# Create checkpoints director;
# Checkpoints folder 是在训练过程中的参数存储，Checkpoints_meta folder 是加载训练时去检测有没有保存的模型可以继续训练, 不然就重新开始; 
checkpoint_dir = os.path.join(workdir, "checkpoints")
checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

# Create Model for Parallel Training; 
score_model = ddpm_model.DDPM(config)
score_model = torch.nn.DataParallel(score_model, config.device_ids)
score_model = score_model.cuda(device=config.device_ids[0])

# define optimizer; 
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
optimizer = losses.get_optimizer(config, score_model.parameters())
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

# Resume training when intermediate checkpoints are detected; 
state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
initial_step = int(state['step'])

# Build data iterators; 
train_loader, eval_loader, _ = datasets.get_dataset_pytorch(config)

# Create data normalizer and its inverse; 
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

#set sde type
if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-4
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

optimize_fn = losses.optimization_manager(config)
continuous = config.training.continuous
reduce_mean = config.training.reduce_mean
likelihood_weighting = config.training.likelihood_weighting

train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean, continuous=continuous, likelihood_weighting=likelihood_weighting)
eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean, continuous=continuous, likelihood_weighting=likelihood_weighting)

if config.training.snapshot_sampling:
    sampling_shape = (16, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps) #这个是用 inverse_scaler 调整回了原本的 [0,1] 上！ 

# In case there are multiple hosts (e.g., TPU pods), only log to host 0
logging.info("Starting training loop at step %d." % (initial_step,))

for epoch in range(initial_step, config.training.epochs+1):
    loss = 0
    for step, (batch) in enumerate(train_loader):
        batch_gpu = batch.cuda(device=config.device_ids[0])
        batch_gpu = repeat(downsample, batch_gpu, times)
        batch_gpu = scaler(batch_gpu)

        batch_loss = train_step_fn(state, batch_gpu)
        loss += batch_loss
    
    if epoch % config.training.log_freq == 0:
        logging.info("Epoch: %d, training_loss: %.5e" % (epoch, loss.item()))

    if epoch != 0 and epoch % config.training.snapshot_freq_for_preemption == 0:
        #这个是用来存储实时的，方便下一次训练的时候能够从当前state开始训练；
        save_checkpoint(checkpoint_meta_dir, state)
        
    if epoch % config.training.eval_freq == 0:
        eval_loss = 0
        for step, (batch) in enumerate(eval_loader):
            batch_gpu = batch.cuda(device=config.device_ids[0]) #[0,1]
            #下面这两个操作是可以交换的！ 
            batch_gpu = repeat(downsample, batch_gpu, times) #[0,1]
            batch_gpu = scaler(batch_gpu) #[-1,1]

            batch_loss = eval_step_fn(state, batch_gpu)
            eval_loss += batch_loss
            break 

        logging.info("Epoch: %d, eval_loss: %.5e" % (epoch, eval_loss.item()))

    if epoch % config.training.snapshot_freq == 0 or epoch == config.training.epochs:
        # Save the checkpoint.
        save_epoch = epoch // config.training.snapshot_freq
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_epoch}.pth'), state)
    
        # Generate and save samples
        if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, "epoch_{}".format(epoch))
            os.makedirs(this_sample_dir, exist_ok=True)
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid(sample, nrow, padding=2)
            sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
            
            with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                np.save(fout, sample)

            with open(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                save_image(image_grid, fout)

