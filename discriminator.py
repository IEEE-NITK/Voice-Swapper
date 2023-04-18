import math

import torch
import torchvision
import torch.nn as nn


class Discriminator(nn.Module):
  def __init__(self,input_nc=3, output_nc=3, filters=64, n_layers=0):
    netD = []
        
    # input is (nc) x 256 x 256
    netD.append(nn.Conv2d(input_nc+output_nc, filters, kernel_size=4,stride=2))
    netD.append(nn.LeakyReLU(0.2, True))
    
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1,n_layers):
      nf_mult_prev = nf_mult
      nf_mult = math.min(2^n,8)
      netD.append(nn.Conv2d(filters * nf_mult_prev, filters * nf_mult, kernel_size=4, stride=2))
      netD.append(nn.BatchNorm2d(filters * nf_mult))
      netD.append(nn.LeakyReLU(0.2, True))
    
    # state size: (filters*M) x N x N
    nf_mult_prev = nf_mult
    nf_mult = math.min(2^n_layers,8)
    netD.append(nn.Conv2d(filters * nf_mult_prev, filters * nf_mult, kernel_size=4,stride=1))
    netD.append(nn.BatchNorm2d(filters * nf_mult))
    netD.append(nn.LeakyReLU(0.2, True))
    
    # state size: (filters*M*2) x (N-1) x (N-1)
    netD.append(nn.Conv2d(filters * nf_mult, 1, kernel_size=4,stride=1))
    # state size: 1 x (N-2) x (N-2)
    
    netD.append(nn.Sigmoid())
    # state size: 1 x (N-2) x (N-2)
    
    self.model = nn.Sequential(*netD)

  def forward(self,x):
    return self.model(x)
