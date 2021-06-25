import pytorch_lightning as pl
import logging
import torch
from neossim import utils

log = logging.getLogger(__name__)


class RandomClassifier(pl.callbacks.Callback):
    """
    Implements a callback for a random classifier, for sanity checking

    Requires that the used module has an eval-tensor-buffer.
    """

    def __init__(self, val=True, test=True, **kwargs):
        self.use_in_val = val
        self.use_in_test = test

    def eval_epoch_end(self, pl_module, stage):
        y_hat = pl_module.eval_buffer["y_hat"]
        y = pl_module.eval_buffer["y"]

        random_scores = torch.rand(size=(y.shape[0],))
        utils.log_osr_metrics(pl_module, random_scores, stage, y, method="random")
        utils.log_uncertainty_metrics(pl_module, random_scores, stage, y, y_hat, method="random")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self.eval_epoch_end(pl_module, "test")

