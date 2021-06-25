"""
A softmax neural network, to be used as a baseline.

See "A baseline for detecting misclassified and out-of-distribution examples in neural networks".

TODO:
- add support for noise injection during prediction
"""
import logging

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as metrics
import torch
import torch.nn as nn

import osr.utils
from neossim import utils
from neossim.models.misc import TensorBuffer

log = logging.getLogger(__name__)


class SoftmaxModel(pl.LightningModule):

    def __init__(self, num_classes, encoder, arch_config, optimizer_config, scheduler_config,
                 save_train_buffer=False, save_eval_buffer=False, **kwargs):
        super(SoftmaxModel, self).__init__()

        self._arch_config = arch_config
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config

        hypers = utils.get_hypers(arch_config, optimizer_config, scheduler_config, **kwargs)
        self.save_hyperparameters(hypers)

        self.num_classes = num_classes
        self.encoder = encoder

        self.loss = self.create_loss()
        self.relu = self.create_nonlinearity()

        self._dropout = nn.Dropout(p=self._arch_config.get("dropout") or 0)

        if self.n_hidden > 0:
            self.hidden = nn.Linear(self.n_features, self.n_hidden)
            self.classifier = nn.Linear(self.n_hidden, self.num_classes)
        else:
            self.classifier = nn.Linear(self.n_features, self.num_classes)

        self._init_weights()

        # add running centers
        self.train_buffer = TensorBuffer()
        self.eval_buffer = TensorBuffer()

        self.save_train_buffer = save_train_buffer
        self.save_eval_buffer = save_eval_buffer

    def create_nonlinearity(self):
        # TODO: make configurable
        return nn.LeakyReLU(0.2)

    def create_loss(self):
        return torch.nn.CrossEntropyLoss()

    @property
    def dropout(self):
        """Dropout rate"""
        return self._dropout.p

    @dropout.setter
    def dropout(self, value):
        # TODO: check if config is writeable, if it is now, we will die here.
        log.info(f"Setting Dropout to {value}")
        self._arch_config.dropout = value
        self._dropout.p = value

    @property
    def in_channels(self):
        return self.encoder.in_channels

    @property
    def n_features(self):
        return self.encoder.n_features

    @property
    def n_hidden(self):
        return self._arch_config.n_hidden or 0

    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "val")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx, loader_idx=None):
        return self.eval_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "test")

    def training_epoch_end(self, outputs) -> None:
        if self.save_train_buffer:
            path = utils.get_dumpfile(self, "train")
            self.train_buffer.save(path)

    def forward(self, x):
        x = self.encoder(x)
        # NOTE: we apply dropout after the encoder
        x = self._dropout(x)

        if self.n_hidden > 0:
            x = self.hidden(x)
            x = self.relu(x)
            # TODO: apply dropout once more?
            x = self._dropout(x)

        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.loss(logits, y)
        y_hat = logits.argmax(dim=1)
        acc = metrics.accuracy(y, y_hat, num_classes=self.num_classes)

        self.log("Loss/train", loss)
        self.log("Accuracy/train", acc, prog_bar=True)

        self.train_buffer.append("logits", logits)
        self.train_buffer.append("y", y)

        results = {
            "loss": loss,
            "y": y,
            "logits": logits
        }
        return results

    def eval_step(self, batch, stage, *foo, **bar) -> None:
        x, y = batch

        logits = self(x)
        y_hat = logits.argmax(dim=1)

        self.eval_buffer.append("logits", logits)
        self.eval_buffer.append("y", y)
        self.eval_buffer.append("y_hat", y_hat)

    def eval_epoch_end(self, outputs, stage):
        log.debug(f"Eval epoch ended (stage: {stage})")
        y_hat = self.eval_buffer["y_hat"]
        y = self.eval_buffer["y"]
        logits = self.eval_buffer["logits"]

        # evaluate softmax thresholding
        confidence = logits.softmax(dim=1).max(dim=1)[0]
        utils.log_osr_metrics(self, confidence, stage, y, method="softmax")
        utils.log_uncertainty_metrics(self, confidence, stage, y, y_hat, method="softmax")
        utils.log_error_detection_metrics(self, confidence, stage, y, y_hat)

        # log once without specific method, to set defaults, used for example in HPO
        utils.log_uncertainty_metrics(self, confidence, stage, y, y_hat)
        utils.log_osr_metrics(self, confidence, stage, y, prog_bar=True)
        utils.log_error_detection_metrics(self, confidence, stage, y, y_hat)

        # log some score histograms
        utils.log_score_histogram(self, stage, confidence, y, y_hat)

        utils.log_classification_metrics(self, stage, y, y_hat, logits=logits)

        # save buffer
        if self.save_eval_buffer or stage == "test":
            path = utils.get_dumpfile(self, stage)
            self.eval_buffer.save(path)

        return None

    def on_train_epoch_start(self) -> None:
        self.train_buffer.clear()

    def on_validation_epoch_start(self) -> None:
        self.eval_buffer.clear()

    def on_test_epoch_start(self) -> None:
        self.eval_buffer.clear()

    def configure_optimizers(self):
        optimizer = utils.create_optimizer(self._optimizer_config, self)
        scheduler = utils.create_scheduler(self._scheduler_config, optimizer)
        return [optimizer], [scheduler]
