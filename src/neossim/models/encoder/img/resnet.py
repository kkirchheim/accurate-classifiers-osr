import torch
import torch.nn as nn
import torchvision.models as models
import logging

from .encoder import ImageEncoder


log = logging.getLogger(__name__)


class BaseResnetEncoder(ImageEncoder):
    def __init__(self, freeze=False, pool="mean", **kwargs):
        resnet = self._create(pretrained=False)
        super(BaseResnetEncoder, self).__init__(in_channels=3, n_features=resnet.fc.in_features)

        if len(kwargs) > 0:
            log.warning("Unused arguments in encoder: ")
        # can only assign after __init__ call
        self.resnet: nn.Module = resnet

        del self.resnet.fc
        del self.resnet.avgpool

        if pool == "mean":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError

        if freeze:
            log.debug("Freezing encoder")
            for param in self.parameters():
                param.requires_grad = False

    def _create(self, **kwargs):
        raise NotImplementedError

    def forward(self, x):
        n_batch = x.shape[0]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.pool(x).view(n_batch, -1)
        return x

    def custom_load_pretrained(self, path):
        state_dict = torch.load(path)

        del state_dict["fc.weight"]
        del state_dict["fc.bias"]

        self.resnet.load_state_dict(state_dict)
        # self.resnet.conv1.weight = state_dict["conv1.weights"].copy()
        # self.resnet.bn1 = state_dict["bn1.weight"].copy()
        # self.resnet.layer1 = state_dict["layer1.weights"].copy()
        # self.resnet.layer2 = state_dict["layer2.weights"].copy()
        # self.resnet.layer3 = state_dict["layer3.weights"].copy()
        # self.resnet.layer4 = state_dict["layer4.weights"].copy()


class Resnet50Encoder(BaseResnetEncoder):
    def __init__(self, **kwargs):
        super(Resnet50Encoder, self).__init__(**kwargs)

    def _create(self, **kwargs):
        return models.resnet50(**kwargs)


class Resnet18Encoder(BaseResnetEncoder):
    def __init__(self, **kwargs):
        super(Resnet18Encoder, self).__init__(**kwargs)

    def _create(self, **kwargs):
        return models.resnet18(**kwargs)


class Resnet101Encoder(BaseResnetEncoder):
    def __init__(self, **kwargs):
        super(Resnet101Encoder, self).__init__(**kwargs)

    def _create(self, **kwargs):
        return models.resnet101(**kwargs)


class WideResnet50Encoder(BaseResnetEncoder):
    def __init__(self,  **kwargs):
        super(WideResnet50Encoder, self).__init__(**kwargs)

    def _create(self, **kwargs):
        return models.wide_resnet50_2(**kwargs)


class WideResnet101Encoder(BaseResnetEncoder):
    def __init__(self,  **kwargs):
        super(WideResnet101Encoder, self).__init__(**kwargs)

    def _create(self, **kwargs):
        return models.wide_resnet101_2(**kwargs)

