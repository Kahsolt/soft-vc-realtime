#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/02 

from traceback import print_exc

import torch
import torchaudio.transforms as TAT
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
sample_rate = 16000       # fixed to the pretrained soft-vc setting


def init(sr_in=sample_rate, sr_out=sample_rate):
  global hubert, acoustic, hifigan
  global resampler_in, resampler_out

  print('Loading soft-vc models...')
  hubert   = torch.hub.load("bshall/hubert:main",          "hubert_soft").to(device)
  acoustic = torch.hub.load("bshall/acoustic-model:main",  "hubert_soft").to(device)
  hifigan  = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").to(device)

  IDENTITY = lambda _:_
  resampler_in  = TAT.Resample(orig_freq=sr_in, new_freq=sample_rate)  if sr_in  != sample_rate else IDENTITY
  resampler_out = TAT.Resample(orig_freq=sample_rate, new_freq=sr_out) if sr_out != sample_rate else IDENTITY


@torch.inference_mode()
def convert(y:np.ndarray) -> np.ndarray:
  try:
    y = y.transpose([1, 0])                   # [T, C=1]
    source = torch.from_numpy(y)              # [C=1, T]
    source = resampler_in(source)
    source = source.unsqueeze(0).to(device)   # [B=1, C=1, T]
    
    units = hubert.units(source)
    mel = acoustic.generate(units).transpose(1, 2)
    target = hifigan(mel)

    y_hat = target.squeeze(0)                 # [C=1, T]
    y_hat = resampler_out(y_hat)
    y_hat = y_hat.cpu().numpy()
    y_hat = y_hat.transpose([1, 0])           # [T, C=1]

    return y_hat
  except:
    print_exc()
    return y
