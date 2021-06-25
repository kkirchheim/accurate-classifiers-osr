import logging
from os.path import join

import pytorch_lightning as pl
import torch


class ExperimentLogger(pl.callbacks.Callback):
    """
    Adds logging for individual experiments. Everything that is logged will also be written into
    a file in the experiment directory.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        self._log = logging.getLogger(__name__)
        self.handler = self._create_handler()
        logging.getLogger().addHandler(self.handler)

    def _create_handler(self):
        path = join(self.experiment.log_dir, "log.log")
        fmt = f"[%(levelname)s] (%(processName)s)  %(asctime)s - {self.experiment.name} " \
              f"(seed {self.experiment.config.seed})> %(message)s"
        ch = logging.FileHandler(filename=path)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(fmt=fmt))
        return ch

    def close(self):
        self._log.info("Closing logging handle.")
        logging.getLogger().removeHandler(self.handler)

    def setup(self, trainer, pl_module, stage: str):
        """Called when fit or test begins"""
        self._log.info(f"Setup stage '{stage}'")

    def teardown(self, trainer, pl_module, stage: str):
        """Called when fit or test ends"""
        self._log.info(f"Teardown stage '{stage}'")

    def on_init_start(self, trainer):
        """Called when the trainer initialization begins, model has not yet been set."""
        self._log.info(f"Init started")

    def on_init_end(self, trainer):
        """Called when the trainer initialization ends, model has not yet been set."""
        self._log.info(f"Init ended")

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins"""
        self._log.info("Fitting model")

    def on_fit_end(self, trainer, pl_module):
        """Called when fit ends"""
        self._log.info("Fitting model ended")

    def on_sanity_check_start(self, trainer, pl_module):
        """Called when the validation sanity check starts."""
        self._log.info("Sanity Check stated")

    def on_sanity_check_end(self, trainer, pl_module):
        """Called when the validation sanity check ends."""
        self._log.info("Sanity Check finished")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the train batch ends."""
        metrics = trainer.logged_metrics
        metrics = {name: value.item() if isinstance(value, torch.Tensor) else value for name, value in metrics.items()}
        self._log.debug(f"Trained one batch: {batch_idx} Metrics: {metrics}")

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        self._log.info(f"Training epoch {trainer.current_epoch} started")

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """Called when the train epoch ends."""
        self._log.info(f"Finished training epoch {trainer.current_epoch}")

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the val epoch begins."""
        self._log.info(f"Validation epoch {trainer.current_epoch} started")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, trainer, pl_module):
        """Called when the test epoch begins."""
        self._log.info("Test epoch started")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        self._log.info("Test epoch ended")

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        pass

    def on_batch_start(self, trainer, pl_module):
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the validation batch begins."""
        self._log.debug(f"Validating one batch: {batch_idx}")
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        pass

    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        self._log.info("Training stated")

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        self._log.info("Training ended")

    def on_pretrain_routine_start(self, trainer, pl_module):
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, trainer, pl_module):
        """Called when the pretrain routine ends."""
        pass

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        self._log.info("Test started")

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        self._log.info("Test ended")

    def on_keyboard_interrupt(self, trainer, pl_module):
        """Called when the training is interrupted by KeyboardInterrupt."""
        pass

    def on_save_checkpoint(self, trainer, pl_module):
        """Called when saving a model checkpoint, use to persist state."""
        self._log.info("Saving Checkpoint")

    def on_load_checkpoint(self, checkpointed_state):
        """Called when loading a model checkpoint, use to reload state."""
        self._log.info("Loading Checkpoint")
