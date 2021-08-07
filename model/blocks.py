import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Conv2dReLU(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect', batch_norm=True):
        super().__init__()
        if batch_norm:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(
                    in_dim, out_dim, kernel_size=kernel_size, 
                    stride=stride, padding=padding, padding_mode=padding_mode
                )),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(
                    in_dim, out_dim, kernel_size=kernel_size, 
                    stride=stride, padding=padding, padding_mode=padding_mode
                )),
                nn.LeakyReLU(0.2)
            )
    
    def forward(self, x):
        return self.block(x)