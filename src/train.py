#!/usr/bin/env python
# coding: utf-8
"""
Run experiments
"""
import logging
import os
import sys
from datetime import datetime

import click

from neossim import utils
from neossim.constants import *
from neossim.engine import run_experiment, run_experiments, load_effective_config
from neossim.factory import DatasetFactory

log = logging.getLogger(__name__)


def main(model_config_path: str, tool_base_config_path: str, overrides, n_runs, is_test,
         workers=1) -> None:
    """
    Assembles the config, configures factories and runs experiments.

    :param model_config_path:
    :param tool_base_config_path:
    :param overrides:
    :param n_runs:
    :param is_test:
    :param workers:
    :return:
    """
    config = load_effective_config(
        model_config_path,
        tool_base_config_path,
        overrides
    )

    log.info(f"Working directory : {os.getcwd()}")
    DatasetFactory.instance().configure(config.tool.datasets)

    # setup
    experiment_name = str(config.comment or datetime.now().strftime("%Y%m%d-%H-%M-%S"))

    root_dir = get_root_dir(config)

    if n_runs is not None:
        if n_runs == 0:
            log.info("Dry run. Nothing to do.")
            return

        run_experiments(config, experiment_name, is_test, n_runs, root_dir, workers)
    else:
        config.seed = config.seed or 0
        run_experiment(config, experiment_name, is_test, root_dir)

    return


def get_root_dir(config):
    train_ds_name = config.ossim.dataset.name
    if train_ds_name is None:
        raise ValueError("Training 'ossim.dataset.name' must not be empty.")
    arch_name = config.architecture.name
    save_dir = join(config.tool.paths.output, train_ds_name, arch_name)  # where to store data
    return save_dir


def try_main(model_config, base_config, overrides, n_runs, is_test, workers) -> int:
    try:
        main(model_config, base_config, overrides, n_runs, is_test, workers)
        log.info("Finished Successfully")
    except Exception as e:
        log.exception(e)
        log.fatal(str(e))
        return EXIT_FAILURE

    return EXIT_SUCCESS


@click.command()
@click.version_option(version="0.0.4")
@click.argument("user_config", metavar="CONFIG")
@click.argument('overrides', nargs=-1)
@click.option("--tool-conf", "tool_conf", type=click.Path(exists=True), default=DEFAULT_TOOL_CONF,
              help="Path to the basic tool configuration.")
@click.option("--runs", "n_runs", type=int, default=None)
@click.option("--test", "is_test", type=bool, default=False, is_flag=True, help="Evaluate on test set. "
                                                                                "Be careful with this")
@click.option("--workers", "workers", type=int, default=1, help="Processes to run in parallel.")
@click.option("--log", "log_path", type=click.Path(), default=None, help="File to log to. Use - for stdout.")
def cli(user_config, tool_conf, overrides, n_runs, is_test, workers, log_path):
    """
    Basic command line interface for training.
    Configuration values can be overridden using dotlist notation, e.g. 'training.epochs=1'
    """
    if log_path is None:
        log_path = join(DEFAULT_LOG_DIR, f"{datetime.now().strftime('%Y%m%d-%H-%M-%S-%f')}.log")

    print(f"Logging to {os.path.abspath(log_path)}")
    utils.configure_logging(log_path, stderr=False)

    status = try_main(user_config, tool_conf, overrides, n_runs, is_test, workers)

    if status == EXIT_SUCCESS:
        click.secho("Finished Successfully.", fg="green", file=sys.stderr)
    else:
        click.secho("Finished unsuccessfully.", fg="red", file=sys.stderr)

    sys.exit(status)


if __name__ == "__main__":
    cli()
