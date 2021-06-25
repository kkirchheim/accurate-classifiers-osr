import pytorch_lightning as pl
import logging
import torch

from osr.openmax import OpenMax as OSROpenMax
from neossim import utils

log = logging.getLogger(__name__)


class OpenMax(pl.callbacks.Callback):
    """
    Implements a callback for an OpenMax Layer. At the end of each evaluation
    epoch this layer is fitted on the training data and evaluated on the new data.

    Requires that the used module has an eval-tensor-buffer and a train-tensor-buffer.
    """
    ATTRIBUTE_KEY = "openmax_layer"
    BUFFER_KEY_OPENMAX = "openmax"

    def __init__(self, tailsize, alpha, euclid_weight, val=True, test=True, **kwargs):
        self.tailsize = tailsize
        self.euclid_weight = euclid_weight
        self.alpha = alpha
        self.use_in_val = val
        self.use_in_test = test

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins"""
        self.add_openmax_layer(pl_module)

    def add_openmax_layer(self, pl_module):
        """
        Adds openmax layer to the given module
        """
        log.info(f"Adding openmax layer")
        if hasattr(pl_module, self.ATTRIBUTE_KEY):
            log.warning(f"OpenMax: attribute exists")
        else:
            om = OSROpenMax(self.tailsize, self.alpha, self.euclid_weight)
            setattr(pl_module, self.ATTRIBUTE_KEY, om)

    def on_test_start(self, trainer, pl_module):
        # fit on correct predictions
        try:
            y_train = pl_module.train_buffer["y"].numpy()
            y_hat_train = pl_module.train_buffer["logits"].argmax(dim=1).numpy()
            logits_train = pl_module.train_buffer["logits"].numpy()
        except:
            log.warning(f"Could not retreive data from buffer")
            return

        correct = y_train == y_hat_train
        openmax: OSROpenMax = getattr(pl_module, self.ATTRIBUTE_KEY)

        if correct.sum() <= 0:
            log.info(f"No correct predictions. Skipping OpenMax fitting.")
        else:
            log.info(f"Fitting OpenMax Layer")
            openmax.fit(logits_train[correct], y_train[correct])
            log.info(f"Fitting OpenMax Layer finished")

    def eval_epoch_end(self, pl_module, stage):
        openmax: OSROpenMax = getattr(pl_module, self.ATTRIBUTE_KEY)

        if openmax.is_fitted:
            log.info(f"Evaluating OpenMax in stage {stage}")

            y_hat = pl_module.eval_buffer["y_hat"]
            logits = pl_module.eval_buffer["logits"]
            y = pl_module.eval_buffer["y"]

            confidence_openmax = 1 - openmax.predict(logits.cpu().numpy())[:, 0]
            confidence_openmax = torch.tensor(confidence_openmax)
            pl_module.eval_buffer.append(self.BUFFER_KEY_OPENMAX, confidence_openmax)

            utils.log_osr_metrics(pl_module, confidence_openmax, stage, y, method="OpenMax")
            utils.log_uncertainty_metrics(pl_module, confidence_openmax, stage, y, y_hat, method="OpenMax")
            utils.log_error_detection_metrics(pl_module, confidence_openmax, stage, y, y_hat, method="OpenMax")
            utils.log_score_histogram(pl_module, f"{stage}/OpenMax", confidence_openmax, y, y_hat)
        else:
            log.warning(f"OpenMax Layer is fitted.")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self.eval_epoch_end(pl_module, "test")

