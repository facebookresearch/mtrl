# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import shutil
import time
from typing import List

import hydra
import torch

from mtrl.utils.types import ConfigType
from mtrl.utils.utils import set_seed


def prepare_and_run(config: ConfigType) -> None:
    """Prepare an experiment and run the experiment.

    Args:
        config (ConfigType): config of the experiment
    """

    set_seed(seed=config.setup.seed)
    print(f"Starting Experiment at {time.asctime(time.localtime(time.time()))}")
    print(f"torch version = {torch.__version__}")  # type: ignore
    experiment = hydra.utils.instantiate(
        config.experiment.builder, config
    )  # cant seem to pass as a kwargs
    experiment.run()


def clear(config: ConfigType) -> None:
    """Clear an experiment and delete all its data/metadata/logs
    given a config

    Args:
        config (ConfigType): config of the experiment to be cleared
    """

    for dir_to_del in get_dirs_to_delete_from_experiment(config):
        shutil.rmtree(dir_to_del)


def get_dirs_to_delete_from_experiment(config: ConfigType) -> List[str]:
    """Return a list of dirs that should be deleted when clearing an
        experiment

    Args:
        config (ConfigType): config of the experiment to be cleared

    Returns:
        List[str]: List of directories to be deleted
    """
    return [config.logbook.dir, config.experiment.save_dir]
