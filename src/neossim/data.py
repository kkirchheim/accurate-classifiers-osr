import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import neossim.utils as utils
from neossim.factory import DatasetFactory
from neossim.pipeline import Pipeline
from osr.ossim import DynamicOSS
from osr.ossim.ossim import TargetMapping

log = logging.getLogger(__name__)


class OSSDataModule(pl.LightningDataModule):
    """
    PYTorch Data module wrapping an open set simulation.

    Loads dataset, creates OSSIM, creates and fits pipelines.
    """

    def __init__(self, ossim_config, transform_config, train_batch_size, inference_batch_size, n_workers, seed=0):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = n_workers
        self.transform_config = transform_config

        self.ossim: DynamicOSS = None
        self.seed = seed

        self.ossim_config = ossim_config

        self.target_mapping = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None

        # pipelines for preprocessing
        self.train_pipeline = None
        self.test_pipeline = None

    def prepare_data(self):
        """
        OPTIONAL, called only on 1 GPU/machine
        """
        log.info("Preparing data module")
        if self.has_prepared_data and not self.has_setup_fit:
            # NOTE: for some reason, has_prepared is set by plightning BEFORE!!!! the method is called.
            #  Therefore we check if we are in between prepare and fit
            self._has_prepared_data = True
        else:
            log.warning("Double prepared called")
            return

        log.info("Setting up Open Set Simulation")
        dataset = self.load_dataset()
        self.ossim = DynamicOSS(
            dataset=dataset,
            train_size=self.ossim_config["split"]["samples"]["train"],
            val_size=self.ossim_config["split"]["samples"]["val"],
            test_size=self.ossim_config["split"]["samples"]["test"],
            kkc=self.ossim_config["split"]["classes"]["kk"],
            kuc=self.ossim_config["split"]["classes"]["ku"],
            uuc_val=self.ossim_config["split"]["classes"]["uu"]["val"],
            uuc_test=self.ossim_config["split"]["classes"]["uu"]["test"],
            seed=self.seed)

        # create mapping
        self.target_mapping = TargetMapping(self.ossim.kkc, self.ossim.kuc, self.ossim.uuc)
        log.debug(f"Created Target Mapping: {self.target_mapping}")

    def load_dataset(self):
        return DatasetFactory.instance().get(self.ossim_config["dataset"]["name"])

    def setup(self, stage):
        """
        OPTIONAL, called for every GPU/machine (assigning state is OK)
        """
        log.info("Setting up datamodule")

        train_dataset = self.ossim.train_dataset()

        log.info("Setting up pipelines")
        # TODO: this should be done in the preparation step
        if not self.train_pipeline:
            self.train_pipeline = Pipeline(self.transform_config, augment=True)
            self.train_pipeline.fit(train_dataset)

        if not self.test_pipeline:
            self.test_pipeline = Pipeline(self.transform_config, augment=False)
            self.test_pipeline.fit(train_dataset)

        # load dataset and use split
        if stage == "fit":
            self.train_dataset = self.ossim.train_dataset()
            utils.set_transformer(self.train_dataset, self.train_pipeline, self.target_mapping)

            self.val_dataset = self.ossim.val_dataset()
            utils.set_transformer(self.val_dataset, self.test_pipeline, self.target_mapping)

            self.test_dataset = self.ossim.test_dataset()
            utils.set_transformer(self.test_dataset, self.test_pipeline, self.target_mapping)

        elif stage == "test":
            self.test_dataset = self.ossim.test_dataset()
            utils.set_transformer(self.test_dataset, self.test_pipeline, self.target_mapping)

            self.val_dataset = self.ossim.val_dataset()
            utils.set_transformer(self.val_dataset, self.test_pipeline, self.target_mapping)

            self.train_dataset = self.ossim.train_dataset()
            utils.set_transformer(self.train_dataset, self.test_pipeline, self.target_mapping)
        else:
            raise ValueError

    # return the dataloader for each split
    def train_dataloader(self):
        """
        We are setting drop_last=True because otherwise, we might get a batch with only a single item. Batch-Norm, e.g.
        in the resnet, can not handle this and will throw an exception.
        """
        log.debug(f"Creating training loader for dataset with len {len(self.train_dataset)}")
        loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                            num_workers=self.num_workers, shuffle=True, drop_last=True)

        return loader

    def val_dataloader(self):
        log.debug(f"Creating validation loader for dataset with len {len(self.val_dataset)}")
        loader = DataLoader(self.val_dataset, batch_size=self.inference_batch_size,
                            num_workers=self.num_workers, shuffle=True)
        return loader

    def test_dataloader(self):
        log.debug(f"Creating test loader for validation dataset with len {len(self.test_dataset)}")
        loader = DataLoader(self.test_dataset, batch_size=self.inference_batch_size,
                            num_workers=self.num_workers, shuffle=True)
        return loader
