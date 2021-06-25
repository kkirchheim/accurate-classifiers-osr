import pytorch_lightning as pl
import logging
from osr.odin import odin_preprocessing
from neossim import utils
import torch.nn.functional as F

log = logging.getLogger(__name__)


class ODIN(pl.callbacks.Callback):
    """
    Implements ODIN Preprocessing

    Requires that the used module has an eval-tensor-buffer.
    """
    BUFFER_KEY_ODIN_LOGITS = "odin_logits"

    def __init__(self, eps, t, val=False, test=True, **kwargs):
        self.epsilon = eps
        self.temperature = t

        self.use_in_val = val
        self.use_in_test = test

    def eval_epoch_end(self, pl_module, stage):
        log.info(f"Evaluating ODIN in stage {stage}")
        logits_odin = pl_module.eval_buffer[self.BUFFER_KEY_ODIN_LOGITS]
        y_hat = pl_module.eval_buffer["y_hat"]
        y = pl_module.eval_buffer["y"]

        confidence_odin, y_hat_odin = logits_odin.softmax(dim=1).max(dim=1)
        utils.log_osr_metrics(pl_module, confidence_odin, stage, y, method="ODIN")
        utils.log_uncertainty_metrics(pl_module, confidence_odin, stage, y, y_hat_odin, method="ODIN")
        utils.log_error_detection_metrics(pl_module, confidence_odin, stage, y, y_hat, method="ODIN")
        utils.log_score_histogram(pl_module, stage, confidence_odin, y, y_hat_odin, method="ODIN")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self.eval_epoch_end(pl_module, "test")

    def eval_batch(self, pl_module, batch, batch_idx, stage):
        log.debug(f"ODIN on {batch_idx}")
        x, y = batch

        x = x.to(pl_module.device)
        x_odin = odin_preprocessing(pl_module, F.nll_loss, x, eps=self.epsilon, temperature=self.temperature)
        logits_odin = pl_module(x_odin) / self.temperature
        pl_module.eval_buffer.append(self.BUFFER_KEY_ODIN_LOGITS, logits_odin)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self.eval_batch(pl_module, batch, batch_idx, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self.eval_batch(pl_module, batch, batch_idx, "test")
