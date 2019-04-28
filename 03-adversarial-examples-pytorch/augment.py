import torch
import torch.nn.functional as F
import numpy as np
from generate_gauss import gauss2D

class AffineTransform(object):
    def __init__(self, range_a=(0.8, 1.5), range_b=(-15, 15), done_p=0.5):
        self.range_a = range_a
        self.range_b = range_b
        self.done_p = done_p

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[0] == 3)
        if np.random.uniform(0, 1) < self.done_p:
            return x
        a = np.random.uniform(self.range_a[0], self.range_a[1], size=1)
        b = np.random.uniform(self.range_b[0], self.range_b[1], size=1)
        x = a*x+b
        return x.clamp(0, 1)


class SaltPepperNoise(object):
    def __init__(self, p, done_p=0.5):
        self.p = p
        self.done_p = done_p

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[0] == 3)
        if np.random.uniform(0, 1) < self.done_p:
            return x
        mask = np.random.uniform(0, 1, size=x.shape)
        mask = torch.from_numpy(mask)
        mask = torch.where(mask<self.p, -1, mask)
        mask = torch.where(mask>1-self.p, 2, mask)

        x = torch.where(mask==-1, 0, x)
        x = torch.where(mask==2, 1, x)
        return x


class Blur(object):
    def __init__(self, blur_kernel_shape, sigma, done_p=0.5):
        assert(blur_kernel_shape[0]%2 == 1)
        self.shape = blur_kernel_shape[0]
        self.blur_kernel = generate_gauss(blur_kernel_shape, sigma)
        self.blur_kernel_torch = torch.from_numpy(self.blur_kernel).view(1, 1, self.shape, -1)
        self.done_p = done_p

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[0] == 3)
        if np.random.uniform(0, 1) < self.done_p:
            return x
        x = x.unsqueeze(0)
        x = F.conv2d(x, self.blur_kernel_torch, stride=1, padding=self.shape//2)
        return x.squeeze(0)
