#!/usr/bin/env python
# coding: utf-8
"""
Hyperparameter optimization using ray tune
"""
import copy
import logging
import os
import sys
import time
from functools import partial
from os.path import abspath

import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from neossim import utils
from neossim.constants import *
from neossim.engine import load_effective_config
from neossim.experiment import Experiment
from neossim.factory import DatasetFactory

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.trial import Trial
except ImportError:
    click.secho(f"Package ray[tune] is missing.", fg="red")
    sys.exit(EXIT_FAILURE)

log = logging.getLogger(__name__)
trial_count = 0


def create_trial_name(trial: Trial):
    """Create pretty trial names"""
    global trial_count
    name = f"trial-{trial_count:05d}"
    trial_count += 1
    return name


class TuneReportCallback(pl.Callback):
    """
    FIXME: since we fetch the validation data from the progress bar (as it seems to be unavailable) anywhere else,
        the models have to log all metrics to the progressbar. We should fetch the values from somewhere else instead.

    """

    def __init__(self, save_checkpoints=False):
        """
        Saving checkpoints is disabled as it requires massive amounts of storage.
        """
        self.log = logging.getLogger(__name__)
        self.save_checkpoints = save_checkpoints

    def on_validation_end(self, trainer, pl_module):
        self.log.debug(trainer.__dict__)
        print(trainer.progress_bar_metrics)
        metrics = trainer.progress_bar_metrics

        auroc = float("NaN")
        aupr = float("NaN")
        acc = float("NaN")
        loss = float("NaN")

        if "AUROC/OSR/val" in metrics:
            auroc = metrics["AUROC/OSR/val"]

        if "AUPR-OUT/OSR/val" in metrics:
            aupr = metrics["AUPR-OUT/OSR/val"]

        if "Accuracy/val" in metrics:
            acc = metrics["Accuracy/val"]

        if "Loss/val" in metrics:
            loss = metrics["Loss/val"]

        tune.report(loss=loss, accuracy=acc, aupr=aupr, auroc=auroc)

    def on_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if self.save_checkpoints:
            with tune.checkpoint_dir(step=pl_module.current_epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                log.info(f"Saving Checkpoint to {path}")
                torch.save((pl_module.state_dict()), path)


def slurm_jobname_workaround():
    """
    Adjust jobname, because otherwise, pytorch light will try to register signal handlers,
    which will result in an error, because it is not started in the main thread
    """
    job_name = os.environ.get('SLURM_JOB_NAME')
    if job_name and job_name != 'bash':
        # PL will not register handlers for jobs with this name
        os.environ['SLURM_JOB_NAME'] = "bash"


def merge_configs(config, tune_config):
    """
    Create dotlist config from random config and merge with given config

    :param config:
    :param tune_config:
    :return:
    """
    new_config = copy.deepcopy(config)
    dotlist = [f"{key}={value}" for key, value in tune_config.items()]
    tune_conf = OmegaConf.from_dotlist(dotlist)
    log.info(tune_conf)
    new_config = OmegaConf.merge(new_config, tune_conf)
    return new_config


def run_session(tune_config, org_config) -> None:
    """
    This function will be run by ray tune in an individual process.
    """
    utils.configure_logging()
    proc_log = logging.getLogger(__name__)

    device = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"CUDA_VISIBLE_DEVICES: {device} for trial {tune.get_trial_name()} "
          f"with id {tune.get_trial_id()} in directory {tune.get_trial_dir()}")

    try:
        slurm_jobname_workaround()

        effective_config = merge_configs(org_config, tune_config)
        proc_log.info(effective_config)

        DatasetFactory.instance().configure(effective_config.tool.datasets)

        proc_log.debug(OmegaConf.to_yaml(effective_config))
        callbacks = [TuneReportCallback()]

        with Experiment(effective_config, user_callbacks=callbacks, progress_bar_refresh_rate=0) as experiment:
            experiment.run()
            experiment.evaluate()
    except ValueError as e:
        proc_log.exception(e)
        print(f"Got Exception in {tune.get_trial_name()}")
        fault_terminated = True
    except Exception as e:
        proc_log.exception(e)
        raise e
    else:
        pass

    time.sleep(5)  # wait to free mem etc.
    return None


def create_search_space(config):
    """
    Creates the search space for the HPO. This is QnD hard-code at the moment.
    """
    space = {
        "architecture.dropout": tune.choice([x / 20.0 for x in range(0, 14, 1)]),
        "optimizer.learning_rate": tune.choice(
            [0.1, 0.05, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.00001]),
        "training.batch_size": tune.choice([32, 64, 128, 256]),
        "optimizer.name": tune.choice(["sgd", "adam"])
    }

    # encoder specific
    if config.architecture.encoder.name in ["dcnn", "fcnn"]:
        space["n_features"]: tune.choice([32, 64, 128, 256, 512])

    # architecture specific
    if config.architecture.name == "ceb":
        space["architecture.n_branch"] = tune.choice([64, 128, 256, 512])
        space["architecture.n_hidden"] = tune.choice([0, 32, 64, 128, 256])
        # ceb has another hidden layer
        space["architecture.lambda"] = tune.uniform(0.1, 10.0)
    elif config.architecture.name == "embedding-center":
        space["architecture.n_hidden"] = tune.choice([0, 64, 128])
        space["architecture.final_batchnorm"] = tune.choice([True, False])
    elif config.architecture.name == "embedding-ii":
        space["training.batch_size"] = tune.choice([128, 256, 512])
        space["architecture.n_hidden"] = tune.choice([0, 32, 64, 128])
        space["architecture.final_batchnorm"] = tune.choice([True, False])
    elif config.architecture.name == "embedding-triplet":
        space["architecture.margin"] = tune.uniform(0.5, 100.0)
        space["training.batch_size"] = tune.choice([128, 256, 512])
        space["architecture.hard_mining"] = tune.choice([True, False])
        space["architecture.squared"] = tune.choice([True, False])
        space["architecture.n_hidden"] = tune.choice([0, 64, 128])
        space["architecture.final_batchnorm"] = tune.choice([True, False])
    elif config.architecture.name == "aux-center":
        space["architecture.lambda"] = tune.uniform(-1.1, 10.0)
        # TODO: hidden missing

    # Auto encoder
    elif config.architecture.name == "ae":
        space["architecture.n_embedding"] = tune.choice([x / 10.0 for x in range(0, 10, 1)])
    elif config.architecture.name == "exp-vae":
        space["architecture.z_dim"] = tune.choice(range(2, 32))

    # Text classification tasks
    elif config.architecture.name == "plain-lstm":
        del space["dropout"]  # no dropout ...
        space["architecture.lstm_hidden_dim"] = tune.choice(range(2, 32))
        space["architecture.lstm_dropout"] = tune.choice(tune.choice([x / 20.0 for x in range(0, 14, 1)]), )
        space["architecture.lstm_layers"] = tune.choice(range(2, 10))
        # word embedding features
        space["architecture.n_features"] = tune.choice(range(2, 20))

    # dataset specific
    if config.ossim.dataset.name == "cub-200" \
            or config.ossim.dataset.name == "stanford-cars" \
            or config.ossim.dataset.name == "imagenet":
        # reduce batch size for large images
        space["training.batch_size"] = tune.choice([64, 128])
    if config.ossim.dataset.name == "tiny-imagenet":
        space["training.batch_size"] = tune.choice([64, 128, 256])

    # scheduler specific
    if config.scheduler.name == "ExponentialLR":
        space["scheduler.gamma"] = tune.loguniform(0.999, 0.99)

    return space


@click.command()
@click.argument("config")
@click.argument('overrides', nargs=-1)
@click.option("--name", "name", type=str, default=str(time.time()))
@click.option("--base-conf", "tool_config", type=click.Path(exists=True), default=DEFAULT_TOOL_CONF)
@click.option("--samples", "num_samples", type=int, default=100)
@click.option("--ray-address", "ray_address", type=str, default=None)
@click.option("--cpu", "cpus_per_trail", type=int, default=10, help="CPUs for a single Job.")
@click.option("--gpu", "gpus_per_trial", type=float, default=1.0, help="GPUs for a single Job.")
@click.option("--cpu-available", "cpus_total", type=int, default=10, help="Total Number of CPUs")
@click.option("--gpu-available", "gpus_total", type=float, default=1.0, help="Total Number of GPUs")
@click.option("--grace-period", "grace_period", type=int, default=20)
@click.option("--seed", "seed", type=int, default=42)
@click.option("--log", "log_file", default=str, help="Where to log. Hyphen for stdout.")
@click.option("--metric", "tune_metric", type=str, default="accuracy", help="What to tune.")
@click.option("--mode", "tune_mode", type=click.Choice(["min", "max"], case_sensitive=False), default="max")
def main(config, overrides, name, tool_config, num_samples, gpus_per_trial, cpus_per_trail, ray_address,
         cpus_total, gpus_total, grace_period, log_file, tune_metric, tune_mode, seed):
    """
    Run Hyperparameter optimization with the given a config template
    """
    pl.seed_everything(seed)
    utils.configure_logging(log_file)

    # connect to ray cluster
    ray.init(address=ray_address, num_cpus=cpus_total, num_gpus=gpus_total)
    log.info("Nodes in the Ray cluster:")
    log.info(ray.nodes())

    config = load_effective_config(user_config_path=config, tool_config_path=tool_config, overrides=overrides)
    space = create_search_space(config)

    # we have to make the external data path absolute, because tune changes the working directory
    external_data_dir = abspath(config["tool"]["paths"]["external"])
    log.info(f"External Data path: {external_data_dir}")
    config["tool"]["paths"]["external"] = external_data_dir

    # we use the accuracy and not the loss as our target metric, because for some loss functions,
    # the range/magnitude of the losses depend on the hyper parameters, so a run with higher loss can actually
    # have better performance
    scheduler = ASHAScheduler(
        metric=tune_metric,
        mode=tune_mode,
        max_t=200,
        grace_period=grace_period,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[k for k in space.keys()],
        metric_columns=["loss", "accuracy", "aupr", "auroc", "training_iteration", "timestep_this_iter"],
        max_error_rows=0,
        max_report_frequency=1)

    resources_per_trial = {"cpu": cpus_per_trail, "gpu": gpus_per_trial}

    train_fn = partial(
        run_session,
        org_config=config)

    try:
        analysis = tune.run(
            train_fn,
            name=name,
            resources_per_trial=resources_per_trial,
            config=space,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=config.tool.paths.raytune,
            verbose=1,
            trial_name_creator=create_trial_name
        )

        out_path = os.path.join(config.tool.paths.raytune, "analysis.pkl")
        df = analysis.results_df
        df.to_pickle(out_path)

        out_path = os.path.join(config.tool.paths.raytune, "analysis.csv")
        df = analysis.results_df
        df.to_csv(out_path)

    except tune.TuneError as e:
        log.exception(e)
        click.secho(f"Tune terminated with an error {e}'", fg="red")
    else:
        click.secho("Finished successfully", fg="green")


if __name__ == "__main__":
    main()
