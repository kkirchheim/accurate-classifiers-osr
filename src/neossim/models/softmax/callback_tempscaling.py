import pytorch_lightning as pl
import logging
import osr
from neossim import utils

log = logging.getLogger(__name__)


class TemperatureScaling(pl.callbacks.Callback):
    """
    Implements a callback for temperature scaling.

    Requires that the used module has an eval-tensor-buffer.
    """

    def __init__(self, t, val=True, test=True, **kwargs):
        self.temperature = t
        self.use_in_val = val
        self.use_in_test = test

    def eval_epoch_end(self, pl_module, stage):
        log.info(f"Evaluating Temperature Scaling in stage {stage}")
        y_hat = pl_module.eval_buffer["y_hat"]
        y = pl_module.eval_buffer["y"]
        logits = pl_module.eval_buffer["logits"]

        confidence_temp = (logits / self.temperature).softmax(dim=1).max(dim=1)[0]
        if osr.utils.contains_known_and_unknown(y):
            utils.log_osr_metrics(pl_module, confidence_temp, stage, y, method="TempScaling")
            utils.log_uncertainty_metrics(pl_module, confidence_temp, stage, y, y_hat, method="TempScaling")
            utils.log_error_detection_metrics(pl_module, confidence_temp, stage, y, y_hat, method="TempScaling")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self.eval_epoch_end(pl_module, "test")

    # TODO: fit to NLL
    # if self.use_temperature_scaling():
    #     if False:  # use scaling, but do not optimize
    #         log.info(f"Optimizing temperature")
    #         # Optimize the temperature by minimizing the NLL on the validation set.
    #         known = utils.is_known(y_val)
    #         correct = y_val == y_hat_val
    #         temperature = osr.utils.optimize_temperature(
    #             logits_val[known & correct], y_val[known & correct], init=self._temperature)
    #         utils.get_tb_writer(self).add_scalar(f"Temperature/val", temperature, global_step=self.global_step)
    #         self.temperature = temperature

