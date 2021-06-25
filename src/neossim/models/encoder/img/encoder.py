import torch.nn as nn
import logging


log = logging.getLogger(__name__)


class EncoderBase(nn.Module):
    """
    Base Class for encoders. They take some data and transform them into an n-dimensional vector.

    """
    def __init__(self, n_features, **kwargs):
        super(EncoderBase, self).__init__()
        self._n_features = n_features

    @property
    def n_features(self):
        return self._n_features

    def custom_load_pretrained(self, path):
        """
        Load pretrained encoder from path
        """
        raise NotImplementedError


class ImageEncoder(EncoderBase):
    def __init__(self, in_channels, n_features, **kwargs):
        super(ImageEncoder, self).__init__(n_features, **kwargs)
        self._in_channels = in_channels

    @property
    def in_channels(self):
        return self._in_channels


