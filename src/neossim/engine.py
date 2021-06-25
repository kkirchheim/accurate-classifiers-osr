"""
Methods for experiment scheduling, resource management, saving results etc.
"""
import copy
import logging
import multiprocessing as mp
import os
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from os import makedirs
from os.path import join
from typing import List, Union
from itertools import cycle

import pandas as pd
from omegaconf import OmegaConf, ListConfig, DictConfig
from pytorch_lightning import loggers as loggers

#
from neossim.experiment import Experiment

log = logging.getLogger(__name__)


device_map = dict()


def get_dedicated_cuda_devce():
    """
    Gets the cuda device for the current worker
    """
    global device_map
    worker_id = get_worker_id()
    return device_map.get(worker_id)


def init_device_map(n_workers):
    """
    Initialize the mapping that assigns cuda devuces to workers.

    Current implementation evenly distributes workers across devices.
    """
    assert n_workers > 0
    global device_map
    cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
    log.info(f"Found {len(cuda_devs)} CUDA_VISIBLE_DEVICES.")

    for worker_id, cuda_device in zip(range(1, n_workers + 1), cycle(cuda_devs)):
        log.info(f"Initializing device mapping: {worker_id} -> {cuda_device}")
        device_map[worker_id] = cuda_device


def run_experiment(config, experiment_name, is_test, save_dir, **kwargs) -> pd.DataFrame:
    """
    Run single experiment

    :param config:
    :param experiment_name:
    :param is_test:
    :param save_dir:
    :return:
    """
    try:
        OmegaConf.set_readonly(config, True)

        worker_id = get_worker_id()
        cuda_dev = get_dedicated_cuda_devce()

        if cuda_dev is not None:
            log.info(f"Using CUDA_DEVICE '{cuda_dev}'")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_dev

        pl_logger = create_pl_loggers(save_dir, experiment_name, config.seed)

        with Experiment(config, pl_logger, process_position=worker_id - 1) as experiment:
            experiment.run()
            result_df = experiment.evaluate(is_test)

        # add runner information
        result_df["seed"] = pd.Series([config.seed] * len(result_df), dtype=int)
        result_df["worker-id"] = pd.Series([worker_id] * len(result_df), dtype=int)

        if cuda_dev:
            result_df["cuda-device"] = pd.Series([cuda_dev] * len(result_df), dtype=str)

        return result_df
    except Exception as e:
        # log exception here, because the main process will not have stack trace information afaik
        log.exception(e)
        raise e


def get_worker_id():
    proc = mp.current_process()
    log.info(f"Worker Identity: {proc._identity}")
    if proc._identity:
        worker_id = mp.current_process()._identity[0]
    else:
        worker_id = 1
    return worker_id


def run_experiments(config, experiment_name, is_test, n_runs, save_dir, workers=1) -> pd.DataFrame:
    """
    Run multiple replicas of the same experiment, all with a different seed
    """
    root_dir = join(save_dir, experiment_name)  # will be the logger root
    log.info(f"Running {n_runs} experiments. Using root '{root_dir}'. ")

    os.makedirs(root_dir, exist_ok=True)

    # save base config once
    OmegaConf.save(config, join(root_dir, "config.yaml"))

    jobs = create_jobs(config, experiment_name, is_test, n_runs, save_dir)

    if workers > 1:
        log.info(f"Running {workers} Experiments in parallel")
        dfs = run_experiments_parallel(workers, jobs, root_dir)
    else:
        dfs = run_experiments_sequential(jobs, root_dir)

    save_dataframes(root_dir, dfs)
    assert len(dfs) == n_runs
    return pd.concat(dfs)


def create_jobs(config_template, experiment_name, is_test, n_runs, save_dir):
    """

    """
    jobs = []
    for run in range(n_runs):
        #
        cfg = copy.deepcopy(config_template)
        cfg.seed = run

        # args passed to run_experiment

        args = {
            "config": cfg,
            "experiment_name": experiment_name,
            "is_test": is_test,
            "save_dir": save_dir,
            "attempt": 0
        }
        jobs.append(args)
    return jobs


def run_experiments_sequential(jobs, root_dir):
    """
    Runs multiple experiments sequentially.
    Saves aggregated dataframes after each run.

    :param jobs:
    :param root_dir:
    :return:
    """
    results = []

    for args in jobs:
        result = run_experiment(**args)
        results.append(result)
        save_dataframes(root_dir, results)
    return results


def run_experiments_parallel(n_workers, jobs, root_dir) -> List[pd.DataFrame]:
    results = []

    init_device_map(n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = dict()

        for n, args in enumerate(jobs):
            future = pool.submit(run_experiment, **args)
            futures[future] = args

        for future in as_completed(futures):
            args = futures[future]
            log.info(f"Job completed: {args}")
            e = future.exception()
            if e is not None:
                attempt = args['attempt']
                log.error(f"Parallel job with seed {args['config'].seed} failed. Attempt: {attempt}")
                log.exception(e)

                # FIXME: if we do not raise e, we run into problems in multiprocessing
                raise e
            else:
                result = future.result()
                results.append(result)

            if len(results) > 0:
                save_dataframes(root_dir, results)

    return results


def save_dataframes(root_dir, dfs):
    result_df = pd.concat(dfs)

    csv_path = join(root_dir, "results.csv")
    log.info(f"Saving results to {csv_path}")
    log.info(f"Saving results to {csv_path}")
    result_df.to_csv()

    pkl_path = join(root_dir, "results.pkl")
    log.info(f"Saving results to {csv_path}")
    result_df.to_pickle(pkl_path)


def create_pl_loggers(save_dir, experiment_name, seed=0) -> List:
    """
    Creates loggers for pytorch lightning. Seed will be used as model version.
    """
    logs = []

    # create tensorboard logger
    tensorboard = loggers.TensorBoardLogger(save_dir=save_dir, name=experiment_name, version=seed,
                                      default_hp_metric="Loss/val")
    # default_hp_metric={}
    log.info(f"Root is {os.path.abspath(tensorboard.log_dir)}")
    makedirs(os.path.abspath(tensorboard.save_dir), exist_ok=True)
    makedirs(os.path.abspath(tensorboard.root_dir), exist_ok=True)
    makedirs(os.path.abspath(tensorboard.log_dir), exist_ok=True)
    logs.append(tensorboard)

    # create csv logger
    csvlog = loggers.CSVLogger(save_dir=save_dir, name=experiment_name, version=seed)
    logs.append(csvlog)

    return logs


def _resolve_imports(config, cwd) -> Union[ListConfig, DictConfig]:
    """
    Recoursively load imported configuration files and merge
    """
    if "import" in config:
        imported = []
        for file in config["import"]:
            path = os.path.join(cwd, file)
            log.debug(f"Importing Config file from path {path}")

            try:
                sub_config = OmegaConf.load(path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not import missing configuration file: '{path}'")
            sub_cwd = os.path.dirname(path)
            sub_config = _resolve_imports(sub_config, sub_cwd)
            imported.append(sub_config)

        del config["import"]

        return OmegaConf.merge(*imported, config)
    else:
        return config


def load_effective_config(user_config_path, tool_config_path, overrides=[]) -> Union[ListConfig, DictConfig]:
    """
    Assemble the config that will ultimately be used in the experiment, with the exception of the
    random seed that might change later if multiple experiments should be conducted.
    """
    user_config = OmegaConf.load(user_config_path)
    tool_config = OmegaConf.load(tool_config_path)
    cli_overrides = OmegaConf.from_cli(overrides)

    log.debug("Using new config loading")
    cwd = os.path.dirname(user_config_path)
    user_config = _resolve_imports(user_config, cwd)

    cwd = os.path.dirname(tool_config_path)
    tool_config = _resolve_imports(tool_config, cwd)

    config = OmegaConf.merge(tool_config, user_config, cli_overrides)
    return config
