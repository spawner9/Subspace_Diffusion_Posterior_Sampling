import jax
import tensorflow as tf

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glob
import sys 
from PIL import Image

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


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size = (resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset_pytorch(config):
  """Create data loaders for training and evaluation. But using the torch.Dataset
  
  """
  # Compute batch size for this worker.
  if config.data.dataset == 'Marmousi':
    data_dir = "datasets/Marmousi/samples_fig/"
    train_split_name = 'train'
    eval_split_name = 'eval'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  class Create_Custom_Dataset(data.Dataset):
    def __init__(self, data_dir, split):
        super(Create_Custom_Dataset, self).__init__()
        self.image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
        
    def __len__(self):
        return len(self.image_files)
    
    def parse_fn(image_file):
      image = Image.open(image_file).convert('L')  # 'L' mode means single-channel (grayscale)
      transform = transforms.Compose([
          transforms.Resize((128, 128)),  # 调整图片大小
          transforms.ToTensor(),  # 将图片转换为PyTorch张量
      ])
      return transform(image)

    def __getitem__(self, idx):
      return self.parse_fn(image_file = self.image_files[idx])
  
  train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
  train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
  eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
  eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 2)

  return train_loader, eval_loader, data_dir    
    

def get_dataset(config, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  if config.data.dataset == 'Marmousi':
    data_dir = "datasets/Marmousi/samples_fig/"
    train_split_name = 'train'
    eval_split_name = 'eval'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')
  
  def parse_fn(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, size=(128, 128))
    image = tf.cast(image, tf.float32) / 255.0

    return image
  
  def create_custom_dataset(data_dir, split):
    # Step 1: 准备数据集
    image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
    
    # Step 2: 创建`tf.data.Dataset`对象
    dataset = tf.data.Dataset.from_tensor_slices((image_files))

    # Step 3: 对数据集进行预处理和转换
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.shuffle(buffer_size=len(image_files))
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Step 4: 返回数据集对象
    return dataset.prefetch(prefetch_size)

  train_ds = create_custom_dataset(data_dir, train_split_name)
  eval_ds = create_custom_dataset(data_dir, eval_split_name)

  return train_ds, eval_ds, data_dir

  