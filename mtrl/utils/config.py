# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Code to interface with the config."""
import datetime
import hashlib
import os
from copy import deepcopy
from typing import Any, Dict, cast

import hydra
from omegaconf import OmegaConf

from mtrl.utils import utils
from mtrl.utils.types import ConfigType


def dict_to_config(dictionary: Dict) -> ConfigType:
    """Convert the dictionary to a config.

    Args:
        dictionary (Dict): dictionary to convert.

    Returns:
        ConfigType: config made from the dictionary.
    """
    return OmegaConf.create(dictionary)


def make_config_mutable(config: ConfigType) -> ConfigType:
    """Set the config to be mutable.

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_readonly(config, False)
    return config


def make_config_immutable(config: ConfigType) -> ConfigType:
    """Set the config to be immutable.

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_readonly(config, True)
    return config


def set_struct(config: ConfigType) -> ConfigType:
    """Set the struct flag in the config.

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_struct(config, True)
    return config


def unset_struct(config: ConfigType) -> ConfigType:
    """Unset the struct flag in the config.

    Args:
        config (ConfigType):

    Returns:
        ConfigType:
    """
    OmegaConf.set_struct(config, False)
    return config


def to_dict(config: ConfigType) -> Dict[str, Any]:
    """Convert config to a dictionary.

    Args:
        config (ConfigType):

    Returns:
        Dict:
    """
    dict_config = cast(
        Dict[str, Any], OmegaConf.to_container(deepcopy(config), resolve=False)
    )
    return dict_config


def process_config(config: ConfigType, should_make_dir: bool = True) -> ConfigType:
    """Process the config.

    Args:
        config (ConfigType): config object to process.
        should_make_dir (bool, optional): should make dir for saving logs, models etc? Defaults to True.

    Returns:
        ConfigType: processed config.
    """
    config = _process_setup_config(config=config)
    config = _process_experiment_config(config=config, should_make_dir=should_make_dir)
    return set_struct(make_config_immutable(config))


def read_config_from_file(config_path: str) -> ConfigType:
    """Read the config from filesystem.

    Args:
        config_path (str): path to read config from.

    Returns:
        ConfigType:
    """
    config = OmegaConf.load(config_path)
    assert isinstance(config, ConfigType)
    return set_struct(make_config_immutable(config))


def _process_setup_config(config: ConfigType) -> ConfigType:
    """Process the `setup` node of the config.

    Args:
        config (ConfigType): config object.

    Returns:
        [ConfigType]: processed config.
    """

    setup_config = config.setup

    if setup_config.base_path is None:
        setup_config.base_path = hydra.utils.get_original_cwd()
    if not setup_config.debug.should_enable:
        setup_config.id = f"{hashlib.sha224(setup_config.description.encode()).hexdigest()}_issue_{setup_config.git.issue_id}_seed_{setup_config.seed}"

    current_commit_id = utils.get_current_commit_id()
    if not setup_config.git.commit_id:
        setup_config.git.commit_id = current_commit_id
    else:
        # if the commit id is already set, assert that the commit id (in the
        # config) is the same as the current commit id.
        if setup_config.git.commit_id != current_commit_id:
            raise RuntimeError(
                f"""The current commit id ({current_commit_id}) does
                 not match the commit id from the config
                 ({setup_config.git.commit_id})"""
            )
    if setup_config.git.has_uncommitted_changes == "":
        setup_config.git.has_uncommitted_changes = utils.has_uncommitted_changes()

    if not setup_config.date:
        setup_config.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    slurm_id = []
    env_var_names = ["SLURM_JOB_ID", "SLURM_STEP_ID"]
    for var_name in env_var_names:
        if var_name in os.environ:
            slurm_id.append(str(os.environ[var_name]))
    if slurm_id:
        setup_config.slurm_id = "-".join(slurm_id)
    else:
        setup_config.slurm_id = "-1"

    return config


def _process_experiment_config(config: ConfigType, should_make_dir: bool) -> ConfigType:
    """Process the `experiment` section of the config.

    Args:
        config (ConfigType): config object.
        should_make_dir (bool): should make dir.

    Returns:
        ConfigType: Processed config
    """
    if should_make_dir:
        utils.make_dir(path=config.experiment.save_dir)
    return config


def pretty_print(config, resolve: bool = True):
    """Prettyprint the config.

    Args:
        config ([type]):
        resolve (bool, optional): should resolve the config before printing. Defaults to True.
    """
    print(OmegaConf.to_yaml(config, resolve=resolve))


def get_env_params_from_config(config: ConfigType) -> ConfigType:
    """Get the params needed for building the environment from a config.

    Args:
        config (ConfigType):

    Returns:
        ConfigType: params for building the environment, encoded as a config.
    """
    env_params = deepcopy(config.env.builder)
    env_params = make_config_mutable(env_params)
    env_params = unset_struct(env_params)
    env_params.pop("_target_")
    return env_params
