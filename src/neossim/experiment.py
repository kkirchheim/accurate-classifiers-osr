import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, GPUStatsMonitor, EarlyStopping
from torch.utils.data import DataLoader

from neossim import utils
from neossim.callbacks import ExperimentLogger
from neossim.data import OSSDataModule
from neossim.factory import ModelFactory, DatasetFactory

log = logging.getLogger(__name__)


class Experiment:
    """
    Class representing a single experiments. Experiments usually consist of sampling an open set calculation,
    fitting a model to the training data, and evaluating on the samples test set.
    """

    def __init__(self, config, pl_loggers=None, user_callbacks=[], **trainer_kwargs):
        """
        :param config:
        :param pl_loggers: pytorch lightning logger
        :param user_callbacks:
        :param trainer_kwargs:
        """
        self.config = config
        self.loggers = pl_loggers

        self.split = None
        self.target_mapping = None
        self.dataset: OSSDataModule = None
        self.model = None

        self.date = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.name = str(config.comment or self.date)

        self.trainer: pl.Trainer = None
        self.trainer_kwargs = trainer_kwargs

        # TODO: make configurable
        self.trainer_callbacks = self._create_default_callbacks()
        self.trainer_callbacks.extend(user_callbacks)
        self.progress_log = ExperimentLogger(self)
        self.trainer_callbacks.append(self.progress_log)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.taredown()

    def __repr__(self):
        return f"{self.name}@{self.date}"

    @property
    def log_dir(self):
        """
        Where to store data for this experiment
        """
        if self.loggers is None:
            return "."

        return utils.get_tb_logger(self.loggers).log_dir

    def setup(self):
        with utils.ContextGuard(self, "Setup"):
            log.info(f"Working directory : {os.getcwd()}")
            log.info(f"Logger directory: {self.log_dir}")
            pl.seed_everything(self.config.seed)

            self.dataset = OSSDataModule(
                self.config.ossim,
                self.config.transform,
                train_batch_size=self.config.training.batch_size,
                inference_batch_size=self.config.evaluation.batch_size,
                n_workers=self.config.training.workers,
                seed=self.config.seed
            )

            # prepare has to be called to set up the split
            self.dataset.prepare_data()

            # save setup
            self._save_dataset_split()
            utils.save_target_mapping(self.log_dir, self.dataset.target_mapping)
            utils.save_config(self.config, self.log_dir)

            # create model and model specific callbacks
            self.model, model_callbacks = self._create_model()
            self.trainer_callbacks.extend(model_callbacks)

            # create trainer.
            self.trainer = self._create_trainer(**self.trainer_kwargs)

    def _save_dataset_split(self):
        """
        Save dataset split information
        """
        log.info("Saving split")
        split = dict()
        split["indices"] = self.dataset.ossim.indices
        split["classes"] = self.dataset.ossim.classes
        utils.save_split(split, directory=self.log_dir)

    def _create_model(self):
        # classes in train should be set by now
        # assert self.config.ossim.n_classes.in_train == len(self.config.ossim.classes.in_train)

        num_classes =  self.config.ossim["split"]["classes"]["kk"]
        if isinstance(num_classes, str):
            num_classes = len(num_classes.split(","))

        return ModelFactory.instance().get(
            self.config.architecture,
            self.config.optimizer,
            self.config.scheduler,
            num_classes,
            training=self.config.training,
            transform=self.config.transform
        )

    def run(self):
        """
        Fit pipeline and model to data
        """
        with utils.ContextGuard(self, "Data Preparation"):
            self.dataset.setup("fit")
            utils.save_pipeline(self.log_dir, self.dataset.train_pipeline, train=True)
            utils.save_pipeline(self.log_dir, self.dataset.test_pipeline, train=False)

        with utils.ContextGuard(self, "Model Fitting"):
            self.trainer.fit(self.model, datamodule=self.dataset)

        if self.trainer.interrupted:
            raise KeyboardInterrupt

    def evaluate(self, is_test=False) -> pd.DataFrame:
        """

        :param is_test: True, if test should be run.
        :return:
        """
        results = []

        with utils.ContextGuard(self, "Evaluation"):
            loaders = []
            stages = []
            names = []

            # train dataset
            train_loader = self._prepare_train_loader_for_test()
            loaders.append(train_loader)
            stages.append("train")
            names.append(self.config.ossim.dataset.name)

            # val dataset
            loaders.append(self.dataset.val_dataloader())
            stages.append("val")
            names.append(self.config.ossim.dataset.name)

            if is_test:
                loaders.append(self.dataset.test_dataloader())
                stages.append("test")
                names.append(self.config.ossim.dataset.name)

            # evaluate on other sets
            for dataset_name in self.config.evaluation.datasets:
                log.info(f"Preparing dataset: {dataset_name}")
                loader = self._prepare_openset_test_set(dataset_name)
                loaders.append(loader)
                stages.append("openset")
                names.append(dataset_name)

            for name, stage, loader in zip(names, stages, loaders):
                entry = self._eval_on_dataset(loader, subset=stage, dataset=name)
                results.append(entry)

        with utils.ContextGuard(self, "Store Results"):
            # store results
            df = pd.DataFrame(results)
            path_csv = os.path.join(self.log_dir, "results.csv")
            log.info(f"Saving results as {path_csv}")
            df.to_csv(path_csv)

            path_pkl = os.path.join(self.log_dir, "results.pkl")
            log.info(f"Saving results as {path_pkl}")
            df.to_pickle(path_pkl)

            return df

    def _prepare_train_loader_for_test(self):
        train_dataset = self.dataset.train_dataset
        test_pipeline = utils.load_pipeline(directory=self.log_dir, train=False)
        mapping = utils.load_target_mapping(directory=self.log_dir)
        utils.set_transformer(train_dataset, test_pipeline, target_mapping=mapping)
        train_loader = DataLoader(train_dataset,
                                  num_workers=self.config.evaluation.workers,
                                  batch_size=self.config.evaluation.batch_size)
        return train_loader

    def _prepare_openset_test_set(self, name) -> DataLoader:
        """
        Load the test dataset with the given name, add a target mapping and transformations
        """
        pipeline = utils.load_pipeline(directory=self.log_dir, train=False)
        dataset = DatasetFactory.instance().get(name)

        # ood-samples get -1 as default class label.
        # FIXME: setting target mapping to a lambda should be documented somewhere
        utils.set_transformer(dataset, pipeline, target_mapping=lambda x: -2000)
        loader = DataLoader(dataset,
                            num_workers=self.config.evaluation.workers,
                            batch_size=self.config.evaluation.batch_size)

        return loader

    def taredown(self):
        del self.model
        self.progress_log.close()

    def _create_trainer(self, **kwargs):
        """
        Creates trainer. this method has way to manny side effects. QnD.
        """
        return pl.Trainer(
            gpus=self.config.gpus,
            deterministic=True,
            logger=self.loggers,
            max_epochs=self.config.training.epochs,
            terminate_on_nan=True,
            callbacks=self.trainer_callbacks,
            # progress_bar_refresh_rate=0,
            # limit_val_batches=100,  # TODO
            # limit_test_batches=100,
            # gradient_clip_val=10,
            # process_position=1,
            **kwargs
        )

    def _create_result_entry(self, result, subset, dataset):
        entry = {
            "Experiment": self.name,
            "Date": self.date,
            "Dataset": dataset,
            "Stage": subset,
        }
        entry.update(result)

        hypers = utils.get_hypers(
            arch_config=self.config["architecture"],
            optimizer_config=self.config["optimizer"],
            scheduler_config=self.config["scheduler"],
            training=self.config["training"],
            evaluation=self.config["evaluation"],
            ossim=self.config["ossim"]
        )

        entry.update(hypers)

        return entry

    def _create_default_callbacks(self, monitor="Loss/val", mode="min") -> List:
        """
        Creates default callbacks.

        Per default, we monitor 'Accuracy/val' for model checkpointing.
        """
        callbacks = []

        if self.loggers is not None:
            # for some reason, this callback only works if we have a logger
            # we will get: "Cannot use ... callback with Trainer that has no logger."
            callbacks.append(LearningRateMonitor())

            checkpoint = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                save_last=True,
                filename="{epoch}",
                dirpath=os.path.join(self.log_dir, "checkpoints"),
                verbose=True
            )
            callbacks.append(checkpoint)

            stopping = EarlyStopping(
                monitor="Loss/val",
                min_delta=0.0,
                mode="min",
                patience=5
            )
            callbacks.append(stopping)

            if not os.getenv("SLURM_JOB_NAME"):
                # NOTE: we unset this when using tune/slurm, as on slurm,
                #  gpu monitoring dramatically slows down training
                callbacks.append(GPUStatsMonitor())

        return callbacks

    def _eval_on_dataset(self, loader, subset, dataset) -> dict:
        """

        """
        log.info(f"Evaluating on {dataset}/{subset}")
        log.info(f"Best checkpoint: {self.trainer.checkpoint_callback.best_model_path} "
                 f"with {self.trainer.checkpoint_callback.best_model_score}")
        self._clear_previous_results()
        result = self.trainer.test(test_dataloaders=loader)[0]
        entry = self._create_result_entry(result, subset=subset, dataset=dataset)
        return entry

    def _clear_previous_results(self):
        """
        Clear results from previous tests on previous datasets.
        If we do not do this, cached values will show up in the evaluation results.
        """
        logger_connector = self.trainer.logger_connector
        logger_connector.callback_metrics.clear()
        logger_connector.evaluation_callback_metrics.clear()
        logger_connector.logged_metrics.clear()
        logger_connector.progress_bar_metrics.clear()
        logger_connector.eval_loop_results.clear()
        logger_connector.cached_results.reset()

        if hasattr(self.model, "_results"):
            self.model._results.clear()
