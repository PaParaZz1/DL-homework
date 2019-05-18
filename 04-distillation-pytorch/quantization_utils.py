import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers

def round_bit(x, num_bit):
    assert(isinstance(x, torch.Tensor))
    assert(isinstance(num_bit, numbers.Integral))
    max_val = 2 ** num_bit - 1
    round_term = torch.floor(x*max_val + 0.5) / max_val - x
    round_term = round_term.detach()
    y = x + round_term
    return y


def quantize_weight(x, num_bit):
    assert(isinstance(x, torch.Tensor))
    assert(isinstance(num_bit, numbers.Integral))
    scale = torch.abs(x).mean() * 2
    round_term = round_bit(torch.clamp(x/scale, -0.5, 0.5)+0.5, num_bit) - 0.5
    round_term = round_term*scale - x
    round_term = round_term.detach()
    y = x + round_term
    return x


def proc(x, multiplier, num_bit):
    x = torch.clamp(x*multiplier, 0, 1)
    x = round_bit(x, num_bit)
    return x


class QuantConv2d(nn.Module):
    def __init__(self, name, in_channels, out_channels, kernel_size, stride=1, padding=0,
                out_f_num_bit=None, w_num_bit=None, is_train=False, proc_multiplier=0.1):
        super(QuantConv2d, self).__init__()
        self.name = name
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.w_num_bit = w_num_bit
        self.out_f_num_bit = out_f_num_bit
        self.proc_multiplier = proc_multiplier

        if isinstance(kernel_size, numbers.Integral):
            kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(self.weight_init_(torch.FloatTensor(out_channels, in_channels, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(affine=False)
        self.affine_k = nn.Parameter(torch.ones(out_channels, 1, 1))
        self.affine_b = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def weight_init_(self, x):
        x = x

    def forward(self, x):
        self.weight = quantize_weight(self.weight, num_bit=self.w_num_bit)
        x = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        x= self.bn(x)
        x = (torch.abs(self.affine_k)+1.0) * x + self.affine_b
        if self.out_f_num_bit != 0:
            x = proc(x, self.proc_multiplier, self.out_f_num_bit)
        return x
