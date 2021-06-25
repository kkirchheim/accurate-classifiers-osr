"""
Densely Connected Convolutional Networks
https://arxiv.org/abs/1608.06993
"""
import torch
import torch.nn.functional as F
import torchvision as tv
from .encoder import ImageEncoder


class DenseNetEncoder(ImageEncoder):
    """
    Wrapper class for the torchvision densenet
    """
    def __init__(self, in_channels, n_features, *pargs, dropout=0.0, pool="mean", **kwargs):
        super(DenseNetEncoder, self).__init__(in_channels=in_channels, n_features=n_features)

        if in_channels != 3:
            raise ValueError

        if pool != "mean":
            raise ValueError

        self.densenet = self._create_densenet(dropout=dropout, **kwargs)
        del self.densenet.classifier

    def forward(self, x):
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def _create_densenet(self):
        raise NotImplementedError

    def custom_load_pretrained(self, path):
        state_dict = torch.load(path)
        del state_dict["classifier.weight"]
        del state_dict["classifier.bias"]
        self.densenet.load_state_dict(state_dict)


class DenseNet121(DenseNetEncoder):
    def _create_densenet(self, dropout=0.0, **kwargs):
        return tv.models.densenet121(drop_rate=dropout, **kwargs)


class DenseNet201(DenseNetEncoder):
    def _create_densenet(self, dropout=0.0, **kwargs):
        return tv.models.densenet201(drop_rate=dropout, **kwargs)

