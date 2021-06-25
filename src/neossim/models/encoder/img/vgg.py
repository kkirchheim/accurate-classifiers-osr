"""
Vgg Net
"""
import torch
import torchvision as tv
from torch import nn
from .encoder import ImageEncoder


class VGGEncoder(ImageEncoder):
    """
    Wrapper class for the torchvision VGG
    """
    def __init__(self, in_channels, n_features=4096, *pargs, dropout=0.5, pool="mean", **kwargs):
        super(VGGEncoder, self).__init__(in_channels=in_channels, n_features=n_features)

        if in_channels != 3:
            raise ValueError

        if pool != "mean":
            raise ValueError

        if dropout != 0.5:
            raise ValueError(f"Vgg does not support other dropout rates than the default 0.5")

        self.vgg = self._create_vgg(**kwargs)
        del self.vgg.classifier

        self.dense_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_features),
            nn.ReLU(True),
            # nn.Dropout(), # we remove this dropout layer as dropout is already contained in the softmax classifiers
            # implementation
        )

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

    def _create_vgg(self):
        raise NotImplementedError


class VGG11(VGGEncoder):
    def _create_vgg(self, **kwargs):
        return tv.models.vgg11(**kwargs)


class VGG19(VGGEncoder):
    def _create_vgg(self, **kwargs):
        return tv.models.vgg19(**kwargs)

