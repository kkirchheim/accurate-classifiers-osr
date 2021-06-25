import pytorch_lightning as pl
import logging
import torch
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM as SkOneClassSVM

from neossim import utils

log = logging.getLogger(__name__)


class OneClassSVM(pl.callbacks.Callback):
    """
    Implements a callback for a one class SVM.
    """
    ATTRIBUTE_KEY_OCSVM = "one_class_svm"
    BUFFER_KEY_OCSVM = "one_class_svm"

    def __init__(self, val=True, test=True, **kwargs):
        self.use_in_val = val
        self.use_in_test = test

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins"""
        self.add_openmax_layer(pl_module)

    def add_one_class_svm(self, pl_module):
        """
        Adds oNE CLASS SVM to the given module
        """
        svm = SkOneClassSVM()
        setattr(pl_module, self.ATTRIBUTE_KEY_OCSVM, svm)

    def get_one_class_svm(self, pl_module) -> SkOneClassSVM:
            return getattr(pl_module, self.ATTRIBUTE_KEY_OCSVM)

    def eval_epoch_end(self, pl_module, stage):
        y_train = pl_module.train_buffer["y"].numpy()
        y_hat_train = pl_module.train_buffer["logits"].argmax(dim=1).numpy()
        logits_train = pl_module.train_buffer["logits"].numpy()
        correct = y_train == y_hat_train

        ocsvm = self.get_one_class_svm(pl_module)

        if correct.sum() <= 0:
            log.info(f"No correct predictions. Skipping OpenMax fitting.")
        else:
            log.info(f"Fitting OpenMax Layer")
            ocsvm.fit(logits_train[correct])
            log.info(f"Fitting OpenMax Layer finished")

        y = pl_module.eval_buffer["y"].numpy()
        y_hat = pl_module.eval_buffer["logits"].argmax(dim=1).numpy()
        logits = pl_module.eval_buffer["logits"].numpy()

        # eval one-class-svm
        try:
            log.info(f"Evaluating One-Class-SVM")
            svm_scores = self.one_class_svm.decision_function(logits)
            svm_scores = torch.tensor(svm_scores)
            self.eval_buffer.append(self.BUFFER_KEY_OCSVM, svm_scores)

            utils.log_osr_metrics(pl_module, svm_scores, stage, y, method="OC-SVM")
            utils.log_uncertainty_metrics(pl_module, svm_scores, stage, y, y_hat, method="OC-SVM")
            utils.log_score_histogram(pl_module, f"{stage}/OC-SVM", svm_scores, y, y_hat)
        except NotFittedError:
            log.warning(f"One Class SVM not fitted.")
            pass

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "test")

