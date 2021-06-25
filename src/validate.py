#!/usr/bin/env python
# coding: utf-8
import logging
import sys

import click

from neossim import Experiment
from neossim.constants import EXIT_SUCCESS, EXIT_FAILURE, DEFAULT_TOOL_CONF
from neossim.engine import load_effective_config
from neossim.factory import DatasetFactory
from neossim.utils import configure_logging

log = logging.getLogger(__name__)


def main(user_conf, tool_conf, overrides):
    """

    """
    config = load_effective_config(user_conf, tool_conf, overrides)
    DatasetFactory.configure(config.tool.datasets)

    with Experiment(config) as experiment:
        log.info("Prepared")


@click.command()
@click.version_option(version="0.0.2")
@click.argument("model_config", metavar="CONF")
@click.argument("overrides", nargs=-1)
@click.option("--tool-conf", "tool_conf", type=click.Path(exists=True), default=DEFAULT_TOOL_CONF,
              help="Path to the basic tool configuration.")
@click.option("--log", "log_path", type=click.Path(), default=None, help="File to log to. Use - for stdout.")
def cli(model_config, tool_conf, overrides, log_path):
    """
    Validate configuration files.
    """
    configure_logging(log_path, stderr=False)

    try:
        main(model_config, tool_conf, overrides)
        click.secho("Config Valid.", fg="green", file=sys.stderr)
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        log.exception(e)
        click.secho("Config Invalid.", fg="red", file=sys.stderr)

    sys.exit(EXIT_FAILURE)


if __name__ == '__main__':
    cli()
