"""
Naive plugin system for datasets and models.
"""
import logging
from typing import List, Tuple

from pytorch_lightning import LightningModule, Callback

from neossim.models.encoder.img import *
from neossim.models.encoder.img.encoder import EncoderBase
from neossim.models.softmax import *

# callbacks
from neossim.models.softmax.callback_mcd import MonteCarloDropout
from neossim.models.softmax.callback_openmax import OpenMax
from neossim.models.softmax.callback_odin import ODIN
from neossim.models.softmax.callback_random import RandomClassifier
from neossim.models.softmax.callback_ocsvm import OneClassSVM
from neossim.models.softmax.callback_tempscaling import TemperatureScaling


# datasets
from osr.dataset.img import *
from osr.dataset.text import *

log = logging.getLogger(__name__)

CALLBACK_MAP = {
    "mcd": MonteCarloDropout,
    "odin": ODIN,
    "openmax": OpenMax,
    "random": RandomClassifier,
    "ocsvm": OneClassSVM,
    "temperature": TemperatureScaling
}

DATASET_MAP = {
    # image
    "mnist": MNIST,
    "kmnist": KMNIST,
    "fmnist": FashionMNIST,
    "cifar-10": CIFAR10,
    "cifar-100": CIFAR100,
    "noise-gauss": GaussianNoise,
    "noise-uniform": UniformNoise,
    "svhn": SVHN,
    "cub-200": Cub2011,
    "stanford-cars": StanfordCars,
    "tiny-imagenet": TinyImagenet,
    "imagenet-2012-64x64": Imagenet2012_64x64,

    # text
    "newsgroup-20": NewsGroup20,
    "reuters-52": Reuters52
}

MODEL_MAP = {
    "softmax": SoftmaxModel,
}

ENCODER_MAP = {
    "resnet-18": Resnet18Encoder,
    "densenet-121": DenseNet121,
    "lenet-5": LeNet,
}

DECODER_MAP = {
}


class ModelFactory:
    """
    Creates model components and assembles them into a single model
    """
    __instance = None

    @staticmethod
    def instance():
        if ModelFactory.__instance is None:
            ModelFactory()

        return ModelFactory.__instance

    def __init__(self):
        if ModelFactory.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelFactory.__instance = self



    @staticmethod
    def get_class(mapping, config):
        name = config["name"]
        clazz = mapping.get(name)
        if clazz is None:
            raise ValueError(f"Class unknown: {config.name}")

        return clazz

    @staticmethod
    def create_instance(mapping, config):
        kwargs = {k: v for k, v in config.items() if not (k == "name") or (k == "pretrained" and type(v) == str)}
        log.info(f"Encoder arguments: {kwargs}")
        clazz = ModelFactory.get_class(mapping, config)
        return clazz(**kwargs)

    @staticmethod
    def create_encoder(encoder_config):
        todo_custom_init = None

        if "pretrained" in encoder_config:
            pretrained = encoder_config["pretrained"]
            if isinstance(pretrained, str):
                todo_custom_init = encoder_config["pretrained"]
                log.info("Custom pretrained model path found.")

        encoder: EncoderBase = ModelFactory.create_instance(ENCODER_MAP, encoder_config)

        if todo_custom_init:
            log.info("Loading custom pretrained model path found.")
            encoder.custom_load_pretrained(todo_custom_init)

        return encoder

    @staticmethod
    def create_decoder(decoder_config):
        return ModelFactory.create_instance(DECODER_MAP, decoder_config)

    @staticmethod
    def get_model_class(arch_name):
        clazz = MODEL_MAP.get(arch_name)

        if clazz is None:
            raise ValueError(f"Unknown Model Class: '{clazz}'")

        return clazz

    @staticmethod
    def get(arch_config, opti_config, sched_config, num_classes, **other_kwargs) -> Tuple[LightningModule, List[Callback]]:
        """
        TODO:
        """
        kwargs = {
            "num_classes": num_classes,
            "arch_config": arch_config,
            "optimizer_config": opti_config,
            "scheduler_config": sched_config,
        }

        kwargs.update(other_kwargs)

        # CNN, text, ...
        if "encoder" in arch_config:
            encoder = ModelFactory.create_encoder(arch_config.encoder)
            kwargs["encoder"] = encoder

        # AE specific
        if "decoder" in arch_config:
            decoder = ModelFactory.create_decoder(arch_config.decoder)
            kwargs["decoder"] = decoder

        clazz = ModelFactory.get_class(MODEL_MAP, arch_config)
        model = clazz(**kwargs)

        callbacks = []

        if "callbacks" in arch_config:
            for item in arch_config["callbacks"]:
                clazz = CALLBACK_MAP.get(item)
                config = arch_config["callbacks"][item]

                kwargs = {
                    "num_classes": num_classes,
                }
                kwargs.update({k: v for k, v in config.items()})

                cb = clazz(**kwargs)
                callbacks.append(cb)

        return model, callbacks


class DatasetFactory:
    """
    Creates Datasets
    """
    __instance = None

    @staticmethod
    def instance():
        if DatasetFactory.__instance is None:
            DatasetFactory()

        return DatasetFactory.__instance

    def __init__(self):
        if DatasetFactory.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DatasetFactory.__instance = self
        self.config = None

    @staticmethod
    def configure(config):
        DatasetFactory.instance().config = config
        log.info(f"Configured dataset factory {config}")

    @staticmethod
    def _prepare_args(config, download=True):
        log.debug(f"Preparing kwargs. Raw Config: {config}")
        default_kwargs = {
            "download": download
        }
        default_kwargs.update(config)

        # FIXME
        if "name" in default_kwargs.keys():
            del default_kwargs["name"]

        return default_kwargs

    @staticmethod
    def get_class(dataset_name):
        clazz = DATASET_MAP.get(dataset_name)
        if clazz is None:
            raise ValueError(f"Unknown Dataset: '{dataset_name}', class not found.")

        return clazz

    @staticmethod
    def get(dataset_name) -> OSRVisionDataset:
        if DatasetFactory.instance().config is None:
            raise ValueError("Unconfigured")

        if dataset_name not in DatasetFactory.instance().config:
            raise ValueError(f"Unknown dataset: '{dataset_name}' not found in tool configuration.")

        dataset_config = DatasetFactory.instance().config.get(dataset_name)
        kwargs = DatasetFactory._prepare_args(dataset_config)
        clazz = DatasetFactory.get_class(dataset_name)

        log.info(f"Dataset Factory: {dataset_name} -> {kwargs}")
        instance = clazz(**kwargs)

        # add name attribute
        setattr(instance, "name", dataset_name)

        return instance
