# -*- coding: utf-8 -*-
"""Noise_scheduler.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19rbIWqtJJRhyyktuvLvHr2VU26Pos02x
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets_gen import get_mnist_,get_cifar10
import math

im_data=get_mnist_(normalize=True)

def get_acost(t):
  s=0.008
  t=t/1000
  t=(t+s)/(1+s)
  t0=s/(1+s)
  f_s=math.pow(math.cos(t*((math.pi)/2)),2)
  f_0=math.pow(math.cos(t0*((math.pi)/2)),2)
  at=f_s/f_0
  return at

def get_at(t):
  T=1000
  at=1
  for i in range(t):
    t=t/T
    at=at*(t*0.9999+(1-t)*0.98)
  return at

#print(get_at(100))

def noise_scheduler(x0,epsilon_noise,t):
  at=get_at(t)
  xt=math.sqrt(at)*x0+math.sqrt(1-at)*epsilon_noise
  return xt





