# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glob
from PIL import Image
from skimage.transform import resize


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset_pytorch(config):
  """
  Create data loaders for training and evaluation. But using the torch.Dataset
  """
  # Compute batch size for this worker.
  if config.data.dataset == 'Marmousi':
    data_dir = "/home/caoxiang/Desktop/Datasets/Marmousi/"
    train_split_name = 'train'
    eval_split_name = 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
          super(Create_Custom_Dataset, self).__init__()
          self.image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
          self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图片大小
            transforms.ToTensor(),  # 将图片转换为PyTorch张量
        ])
          
      def __len__(self):
          return len(self.image_files)

      def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('L')  # 'L' mode means single-channel (grayscale)
        image = self.transform(image)
        
        return image

    train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)

  
  elif config.data.dataset == 'KIT4':
    data_dir = "/home/caoxiang/Desktop/Datasets/KIT4/samples_fig/"
    train_split_name = 'train'
    eval_split_name = 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
          super(Create_Custom_Dataset, self).__init__()
          self.image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
          self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图片大小
            transforms.ToTensor(),  # 将图片转换为PyTorch张量
        ])
          
      def __len__(self):
          return len(self.image_files)

      def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('L')  # 'L' mode means single-channel (grayscale)
        image = self.transform(image)
        
        return image

    train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)
  
  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')


  return train_loader, eval_loader, data_dir 
