import torch
import torch.nn.functional as F
import numpy as np
from generate_gauss import gauss2D


class AffineTransform(object):
    def __init__(self, range_a=(0.7, 1.3), range_b=(-0.15, 0.15), done_p=0.5):
        self.range_a = range_a
        self.range_b = range_b
        self.done_p = done_p

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[0] == 3)
        if np.random.uniform(0, 1) > self.done_p:
            return x
        a = np.random.uniform(self.range_a[0], self.range_a[1], size=1)
        b = np.random.uniform(self.range_b[0], self.range_b[1], size=1)
        a, b = torch.from_numpy(a).float(), torch.from_numpy(b).float()
        x = a*x+b
        return x.clamp(0, 1)


class SaltPepperNoise(object):
    def __init__(self, p, done_p=0.5):
        self.p = p
        self.dead_tensor = torch.FloatTensor([-1])
        self.hot_tensor = torch.FloatTensor([2])
        self.dead_point = torch.FloatTensor([0])
        self.hot_point = torch.FloatTensor([1])
        self.done_p = done_p

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[0] == 3)
        if np.random.uniform(0, 1) > self.done_p:
            return x
        mask = np.random.uniform(0, 1, size=x.shape)
        mask = torch.from_numpy(mask).float()
        mask = torch.where(mask < self.p, self.dead_tensor, mask)
        mask = torch.where(mask > 1-self.p, self.hot_tensor, mask)

        x = torch.where(mask == self.dead_tensor, self.dead_point, x)
        x = torch.where(mask == self.hot_tensor, self.hot_point, x)
        return x


class Blur(object):
    def __init__(self, blur_kernel_shape, sigma, done_p=0.5):
        assert(blur_kernel_shape[0] % 2 == 1)
        self.shape = blur_kernel_shape[0]
        self.blur_kernel = gauss2D(blur_kernel_shape, sigma)
        self.blur_kernel_torch = torch.from_numpy(
            self.blur_kernel).view(1, 1, self.shape, -1).float()
        self.blur_kernel_torch = self.blur_kernel_torch.repeat(3, 1, 1, 1)
        self.done_p = done_p

    def __call__(self, x):
        assert(isinstance(x, torch.Tensor))
        assert(x.shape[0] == 3)
        if np.random.uniform(0, 1) > self.done_p:
            return x
        x = x.unsqueeze(0)
        x = F.conv2d(x, self.blur_kernel_torch, stride=1,
                     padding=self.shape//2, groups=3)
        return x.squeeze(0)
