import pytorch_lightning as pl
import logging
import torch

from neossim import utils

log = logging.getLogger(__name__)


class MonteCarloDropout(pl.callbacks.Callback):
    """
    Implements Monte Carlo Dropout

    Requires that the used module has an eval-tensor-buffer.
    """
    BUFFER_KEY_MC_CONF = "mc_conf"
    BUFFER_KEY_MC_PREDICTION = "mc_pred"

    def __init__(self, num_classes, rounds, val=False, test=True, **kwargs):
        self.rounds = rounds
        self.use_in_val = val
        self.use_in_test = test
        self.num_classes = num_classes

    def eval_epoch_end(self, pl_module, stage):
        log.info(f"Evaluating MCD in stage {stage}")
        confidence_mcd = pl_module.eval_buffer[self.BUFFER_KEY_MC_CONF]
        y_hat = pl_module.eval_buffer["y_hat"]
        y = pl_module.eval_buffer["y"]

        utils.log_osr_metrics(pl_module, confidence_mcd, stage, y, method="MCD")
        utils.log_uncertainty_metrics(pl_module, confidence_mcd, stage, y, y_hat, method="MCD")
        utils.log_error_detection_metrics(pl_module, confidence_mcd, stage, y, y_hat, method="MCD")
        utils.log_score_histogram(pl_module, f"{stage}/MCD", confidence_mcd, y, y_hat)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self.eval_epoch_end(pl_module, "test")

    @staticmethod
    def monte_carlo_dropout(num_classes, pl_module, x: torch.Tensor, rounds=10):
        """
        Runs several rounds of Monte-Carlo-Dropout
        """
        # TODO: check if dropout is active
        pl_module.train()  # activate dropout
        results = torch.zeros(size=(x.size(0), num_classes), device=pl_module.device)

        x = x.to(pl_module.device)

        with torch.no_grad():
            for i in range(rounds):
                results += pl_module(x).softmax(dim=1)

        results /= rounds
        pl_module.eval()  # deactivate dropout again

        return results.max(dim=1)

    def eval_batch(self, pl_module, batch, batch_idx, stage):
        log.debug(f"MCD on {batch_idx}")
        x, y = batch
        mc_conf, mc_pred = self.monte_carlo_dropout(self.num_classes, pl_module, x, rounds=self.rounds)
        pl_module.eval_buffer.append(self.BUFFER_KEY_MC_CONF, mc_conf)
        pl_module.eval_buffer.append(self.BUFFER_KEY_MC_PREDICTION, mc_pred)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self.eval_batch(pl_module, batch, batch_idx, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self.eval_batch(pl_module, batch, batch_idx, "test")
