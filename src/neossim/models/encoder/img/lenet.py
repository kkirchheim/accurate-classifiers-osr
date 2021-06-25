"""

:see Tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
from torch import nn
import torch.nn.functional as F

from .encoder import EncoderBase


class LeNet(EncoderBase):

    def __init__(self, in_channels=1, pool="max", **kwargs):
        super(LeNet, self).__init__(in_channels=in_channels, n_features=16)
        self.conv1 = nn.Conv2d(in_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        if pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError()

    def forward(self, x):
        n_batch = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.pool(x).view(n_batch, -1)
        return x
