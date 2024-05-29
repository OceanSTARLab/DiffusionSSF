## sampling code
import matplotlib.pyplot as plt
import matplotlib
import torch
from models.ema import ExponentialMovingAverage

from pathlib import Path
import controllable_generation
from utils import restore_checkpoint, clear_color, clear

import models
from models import utils as mutils
from models import ncsnpp
import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets
import losses
from scipy import io
import numpy as np

from configs.ve import hycom_ncsnpp_deep_continuous as configs

ckpt_filename = "./hycomlog/checkpoints/checkpoint_7.pth"
config = configs.get_config()
# config.model.num_scales = num_scales
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sampling_eps = 1e-5

random_seed = 0

# sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
# state = dict(step=0, model=score_model, ema=ema)
optimizer = losses.get_optimizer(config, score_model.parameters())
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_optimizer=True)
ema.copy_to(score_model.parameters())

# Directory to save samples.
save_root = Path(f'./results')
save_root.mkdir(parents=True, exist_ok=True)

sampling_method = 'vanila'
if sampling_method == 'vanila':#unconditional sampling
    print("unconditional sampling")
    sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    print("========sampling========")
    samples, n = sampling_fn(score_model)#抽样的核心
    # 保存声速数据
    samples = samples.detach().cpu().numpy()
    file_name = "ssf_gene.mat"
    io.savemat(save_root / file_name, {'ssf_gene':samples})
    print("========finish========")

else:
    pass



