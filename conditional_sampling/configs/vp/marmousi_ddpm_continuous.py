# 这个具体的设定需要在 default 上修改！
from configs.default_marmousi_configs import get_default_configs

def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.predictor = 'ancestral_sampling' #暂时用 ancestral_sampling 回归到 DDPM 的方式 #euler_maruyama, reverse_diffusion, ancestral_sampling, none
  sampling.corrector = 'none' # langevin, ald, none
  sampling.operator = 'eikonal' # eikonal
  sampling.condition_method = 'ps' # eikonal 本来这个地方是 sampling.method = "pc" or "ode"

  # data
  data = config.data
  data.centered = True #将数据从 [0,1] 归一化到 [-1,1]! #这个不管有没有, 都要做
  
  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
