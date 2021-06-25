"""
Constants, e.g. paths which we can not place in the configuration.
"""
from os.path import join, dirname

# default base configuration files
DEFAULT_TOOL_CONF = join(dirname(__file__), "..", "..", "config", "tool.yaml")

# path for data, e.g. datasets, experiments, logs, ...
DEFAULT_DATA_DIR = join(dirname(__file__), "..", "..", "data")
DEFAULT_LOG_DIR = join(DEFAULT_DATA_DIR, "logs")

# command line interface exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
