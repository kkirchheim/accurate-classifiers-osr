"""
Contains functions related to target and input transformation, e.g. augmentation and target index mapping.
"""
import logging

import torchvision.transforms as T

from neossim.pipeline.img import ImageNormalizer
from neossim.pipeline.txt import TextVectorizer, TextPadding

log = logging.getLogger(__name__)


class Pipeline:
    """
    Data processing pipeline
    """
    def __init__(self, config, augment: bool = False):
        self._steps = []
        self.is_augment = augment
        self.is_fitted = False

        for component in config:
            if is_augmentation_step(component) and not self.is_augment:
                log.debug(f"Skipping {get_step_name(component)}")
                continue

            step = create_pipeline_step(component)
            self._steps.append(step)

        self._composed = T.Compose(self._steps)

    def __call__(self, *args, **kwargs):
        return self._composed(*args, **kwargs)

    def fit(self, dataset):
        log.info("Fitting pipeline to data")
        for step in self._steps:
            if hasattr(step, "fit"):
                step.fit(dataset)

        self.is_fitted = True


def is_augmentation_step(component) -> bool:
    name = get_step_name(component)
    if name == "text-vectorize":
        return False
    if name == "img-normalize":
        return False
    if name == "to-tensor":
        return False

    return True


def get_step(step_name):
    """
    # transformer configuration
    transform:
      size: [32, 32]
      normalize: True
      hflip: 0
      vflip: 0
      rotation: 0
      scale:
        min: 0.7
        max: 1.0
    """
    if step_name == "text-vectorize":
        return TextVectorizer
    if step_name == "text-padding":
        return TextPadding
    if step_name == "img-normalize":
        return ImageNormalizer
    if step_name == "hflip":
        return T.RandomHorizontalFlip
    if step_name == "vflip":
        return T.RandomVerticalFlip
    if step_name == "rotate":
        return T.RandomRotation
    if step_name == "resized-crop":
        return T.RandomResizedCrop
    if step_name == "resize":
        return T.Resize
    if step_name == "to-tensor":
        return T.ToTensor
    else:
        raise ValueError(f"Unknown pipeline step: '{step_name}'")


def get_step_name(component):
    log.info(component)
    return list(component.keys())[0]


def create_pipeline_step(component):
    name = get_step_name(component)
    kwargs = list(component.values())[0] or dict()
    log.debug(f"Creating pipeline step: '{name}' -> {kwargs}")
    clazz = get_step(name)

    return clazz(**kwargs)


