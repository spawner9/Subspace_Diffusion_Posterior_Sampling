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
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
from torchvision.utils import make_grid, save_image
import numpy as np
import abc

import torch 
from torch.autograd import Function

import numpy as np

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

import os

_CORRECTORS = {}
_PREDICTORS = {}
_OPERATORS = {}
_CONDITIONING_METHODS= {}

def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)
  
def register_operator(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _OPERATORS:
      raise ValueError(f'Already registered operator with name: {local_name}')
    _OPERATORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)
  
def register_condition_method(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CONDITIONING_METHODS:
      raise ValueError(f'Already registered condition method with name: {local_name}')
    _CONDITIONING_METHODS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]

def get_corrector(name):
  return _CORRECTORS[name]

def get_operator(name):
  return _OPERATORS[name]

def get_conditioning_method(name, operator, scale):
  return _CONDITIONING_METHODS[name](operator, scale)


####################
# Condition Method #
####################

class ConditioningMethod(abc.ABC):
    def __init__(self, operator):
        self.operator = operator
        
    def grad_and_value(self, x_prev, x_0_hat):
        norm = self.operator.forward(x_0_hat)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        return norm_grad, norm
  
    @abc.abstractmethod
    def conditioning(self, x_t, measurement):
        pass

@register_condition_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, scale):
        super().__init__(operator)
        self.scale = scale
    
    def conditioning(self, x_prev, x_0_hat):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat)
        dx_t = norm_grad * self.scale/torch.sqrt(norm)
        return dx_t, norm

#############
# Predictor #
#############

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean
 
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    
    #Compute posterior mean for x_0_hat 
    sqrt_alphas_cumprod_t = sde.sqrt_alphas_cumprod.to(t.device)[timestep]
    alphas_cumprod_t = sde.alphas_cumprod.to(t.device)[timestep]
    coef_1 = (1.0/sqrt_alphas_cumprod_t).to(t.device)
    coef_2 = ((1.0 - alphas_cumprod_t)/sqrt_alphas_cumprod_t).to(t.device)
    
    x_0_hat = coef_1[:, None, None, None] * x + coef_2[:, None, None, None] * score

    return x, x_mean, x_0_hat[0:1,:,:,:] 

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


#############
# Corrector #
#############

class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


###########
# Sampler #
###########

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)


def get_dps_sde_sampler(sde, shape, predictor, condition_sampler, scaler, inverse_scaler, 
                    n_steps=1, probability_flow=False, continuous=False, denoise= False,
                    eps=1e-3, device='cuda'):
  """Create a DPS sampler.
  """
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  
  condition_sample_fn  = functools.partial(condition_sampler.conditioning)

  def dps_sampler(model, idx):
    """ The DPS sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    
    # Initial sample
    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

    for i in range(sde.N):
      t = timesteps[i]
      vec_t = torch.ones(shape[0], device=t.device) * t

      x = x.requires_grad_()
      x_t, x_mean, x_0_hat = predictor_update_fn(x, vec_t, model=model)
      #inverse scaler + clamp
      x_0_hat = inverse_scaler(x_0_hat).clamp(0.05,0.95)
      dx_t, norm = condition_sample_fn(x, x_0_hat)
      x = x_t - dx_t
      
      x = x.detach_()
      norm = norm.detach_() 
      

    return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return dps_sampler


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise= False, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      for i in range(sde.N):
        # print(i)
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean, _ = predictor_update_fn(x, vec_t, model=model)
           
      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler

def get_dps_ode_sampler(sde, shape, condition_sampler, inverse_scaler,
                    denoise=True, rtol=1e-4, atol=1e-4,
                    method='RK23', eps=1e-5, device='cuda'):
  
  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x
  
  def drift_fn(model, x, t):
    """ Get drift term """
    def score_fn(x, t): 
      #这个函数 无缝衔接的 把 DDPM 训练好的模型转化为 score-based 模型！这里的 DDPM 中的 Markov chain 是通过 Score-based SDE 推导出来的，即 betas 离散化由 beta(t)/N 给出！      labels = t * 999
      labels = t * 999
      score = model(x, labels)
      std = sde.marginal_prob(torch.zeros_like(x), t)[1] # std 和 除以的\bar alpha factor 有关！
      score = -score / std[:, None, None, None]
      return score
    
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]
  
  def x_0_hat_fn(model, x, t):
    """ Compute posterior mean for x_0_hat """
    def score_fn(x, t):
      labels = t * 999
      score = model(x, labels)
      std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      score = -score / std[:, None, None, None]
      return score
    
    # Compute posterior mean for continuous sde!
    _, std, mean_x = sde.marginal_prob_1(x, t)
    x_0_hat = x/mean_x +  std ** 2 * score_fn(x, t)/mean_x
    
    return x_0_hat
  

  def ode_sampler(model, z=None):
    if z is None:
      # If not represent, sample the latent code from the prior distibution of the SDE.
      x = sde.prior_sampling(shape).to(device)
    else:
      x = z

    def ode_func(t, x):
      x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
      
      x = x.requires_grad_()
      vec_t = torch.ones(shape[0], device=x.device) * t
      
      x_0_hat = x_0_hat_fn(model, x, vec_t)
      x_0_hat = inverse_scaler(x_0_hat).clamp(0.05,0.95)
      dx_t, norm = condition_sampler.conditioning(x, x_0_hat) 
      
      dx_t = dx_t.detach_()
      x = x.detach_()
      vec_t = vec_t.detach_()

      drift, diffusion = sde.sde(x, vec_t)
      #drift = drift_fn(model, x, vec_t) + diffusion[:, None, None, None] ** 2 * dx_t * 0.5
      drift = drift_fn(model, x, vec_t) + dx_t
      
      return to_flattened_numpy(drift)
    
    # Black-box ODE solver for the probability flow ODE
    solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                    rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
    
    # Denoising is equivalent to running one predictor step without adding noise
    if denoise:
      x = denoise_update_fn(model, x)

    x = inverse_scaler(x)
    return x, nfe
  
  return ode_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise= False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler



