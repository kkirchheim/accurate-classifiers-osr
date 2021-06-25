"""
Common components of neural networks
"""
import logging
from collections import defaultdict

import torch
import torch.nn as nn

#
import osr.utils

log = logging.getLogger(__name__)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size=256, w=1, h=1):
        super(UnFlatten, self).__init__()
        self.size = size
        self.w = w
        self.h = h

    def forward(self, x):
        return x.view(x.size(0), self.size, self.w, self.h)


class RunningCenters(nn.Module):
    """
    Module that tracks running centers of embeddings for each class
    """

    def __init__(self, n_classes, n_embedding):
        super(RunningCenters, self).__init__()
        self.n_classes = n_classes
        self.n_embedding = n_embedding

        # create buffer for centers. those buffers will be updated during training, and are fixed during evaluation
        running_centers = torch.empty(size=(self.n_classes, self.n_embedding), requires_grad=False).double()
        num_batches_tracked = torch.empty(size=(1,), requires_grad=False).double()

        self.register_buffer("centers", running_centers)
        self.register_buffer("num_batches_tracked", num_batches_tracked)
        self.reset()

    def reset(self):
        nn.init.zeros_(self.centers)
        nn.init.zeros_(self.num_batches_tracked)

    def calculate_centers(self, embeddings, target):
        mu = torch.full(size=(self.n_classes, self.n_embedding), fill_value=float('NaN'), device=embeddings.device)

        for clazz in target.unique(sorted=False):
            mu[clazz] = embeddings[target == clazz].mean(dim=0)  # all instances of this class

        return mu

    def forward(self, x, y):
        batch_classes = y.unique(sorted=False)
        mu = self.calculate_centers(x, y)

        # update running mean centers
        cma = mu[batch_classes] + self.centers[batch_classes] * self.num_batches_tracked
        self.centers[batch_classes] = cma / (self.num_batches_tracked + 1)
        self.num_batches_tracked += 1

        return x

    def calculate_distances(self, embeddings):
        # TODO: using squared distance ...
        distances = osr.utils.torch_get_squared_distances(self.centers, embeddings)
        return distances


class TensorBuffer:
    """
    Used to buffer some tensors
    """

    def __init__(self, device="cpu"):
        self._buffer = defaultdict(list)
        self.device = device

    def append(self, key, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Can not handle value type {type(value)}")

        # log.debug(f"Adding tensor with key {key} to buffer shape=({value.size()})")
        value = value.detach().to(self.device)

        self._buffer[key].append(value)
        return self

    def __getitem__(self, item):
        return self.get(item)

    def sample(self, key) -> torch.Tensor:
        index = torch.randint(0, len(self._buffer[key]), size=(1,))
        return self._buffer[key][index]

    def get(self, key) -> torch.Tensor:
        if key not in self._buffer:
            raise KeyError(key)

        v = torch.cat(self._buffer[key])
        # log.debug(f"Retrieving from buffer {key} with shape={v.size()}")
        return v

    def clear(self):
        log.debug("Clearing buffer")
        self._buffer.clear()
        return self

    def save(self, path):
        """Save buffer to disk"""
        d = {k: self.get(k).cpu() for k in self._buffer.keys()}
        log.debug(f"Saving tensor buffer to {path}")
        torch.save(d, path)
        return self

