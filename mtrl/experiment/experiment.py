# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a model."""

import json
import os
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import hydra
import torch

from mtrl.env.types import EnvType
from mtrl.logger import Logger
from mtrl.utils import checkpointable
from mtrl.utils import config as config_utils
from mtrl.utils import utils, video
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType


class Experiment(checkpointable.Checkpointable):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        self.id = experiment_id
        self.config = config
        self.device = torch.device(self.config.setup.device)

        self.get_env_metadata = get_env_metadata
        self.envs, self.env_metadata = self.build_envs()

        key = "ordered_task_list"
        if key in self.env_metadata and self.env_metadata[key]:
            ordered_task_dict = {
                task: index for index, task in enumerate(self.env_metadata[key])
            }
        else:
            ordered_task_dict = {}

        key = "envs_to_exclude_during_training"
        if key in self.config.experiment and self.config.experiment[key]:
            self.envs_to_exclude_during_training = {
                ordered_task_dict[task] for task in self.config.experiment[key]
            }
            print(
                f"Excluding the following environments: {self.envs_to_exclude_during_training}"
            )
        else:
            self.envs_to_exclude_during_training = set()

        self.action_space = self.env_metadata["action_space"]
        assert self.action_space.low.min() >= -1
        assert self.action_space.high.max() <= 1

        self.env_obs_space = self.env_metadata["env_obs_space"]

        env_obs_shape = self.env_obs_space.shape
        action_shape = self.action_space.shape

        self.config = prepare_config(config=self.config, env_metadata=self.env_metadata)
        self.agent = hydra.utils.instantiate(
            self.config.agent.builder,
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=[
                float(self.action_space.low.min()),
                float(self.action_space.high.max()),
            ],
            device=self.device,
        )

        self.video_dir = utils.make_dir(
            os.path.join(self.config.setup.save_dir, "video")
        )
        self.model_dir = utils.make_dir(
            os.path.join(self.config.setup.save_dir, "model")
        )
        self.buffer_dir = utils.make_dir(
            os.path.join(self.config.setup.save_dir, "buffer")
        )

        self.video = video.VideoRecorder(
            self.video_dir if self.config.experiment.save_video else None
        )

        self.replay_buffer = hydra.utils.instantiate(
            self.config.replay_buffer,
            device=self.device,
            env_obs_shape=env_obs_shape,
            task_obs_shape=(1,),
            action_shape=action_shape,
        )

        self.start_step = 0

        should_resume_experiment = self.config.experiment.should_resume

        if should_resume_experiment:
            self.start_step = self.agent.load_latest_step(model_dir=self.model_dir)
            self.replay_buffer.load(save_dir=self.buffer_dir)

        self.logger = Logger(
            self.config.setup.save_dir,
            config=self.config,
            retain_logs=should_resume_experiment,
        )
        self.max_episode_steps = self.env_metadata[
            "max_episode_steps"
        ]  # maximum steps that the agent can take in one environment.

        self.startup_logs()

    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Subclasses should implement this method to build the environments.

        Raises:
            NotImplementedError: this method should be implemented by the subclasses.

        Returns:
            Tuple[EnvsDictType, EnvMetaDataType]: Tuple of environment dictionary
            and environment metadata.
        """
        raise NotImplementedError(
            "`build_envs` is not defined for experiment.Experiment"
        )

    def startup_logs(self) -> None:
        """Write some logs at the start of the experiment."""
        config_file = f"{self.config.setup.save_dir}/config.json"
        with open(config_file, "w") as f:
            f.write(json.dumps(config_utils.to_dict(self.config)))

    def periodic_save(self, epoch: int) -> None:
        """Perioridically save the experiment.

        This is a utility method, built on top of the `save` method.
        It performs an extra check of wether the experiment is configured to
        be saved during the current epoch.
        Args:
            epoch (int): current epoch.
        """
        persist_frequency = self.config.experiment.persist_frequency
        if persist_frequency > 0 and epoch % persist_frequency == 0:
            self.save(epoch)

    def save(self, epoch: int) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def load(self, epoch: Optional[int]) -> Any:  # type: ignore[override]
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def run(self) -> None:
        """Run the experiment.

        Raises:
            NotImplementedError: This method should be implemented by the subclasses.
        """
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def close_envs(self):
        """Close all the environments."""
        for env in self.envs.values():
            env.close()


def prepare_config(config: ConfigType, env_metadata: EnvMetaDataType) -> ConfigType:
    """Infer some config attributes during runtime.

    Args:
        config (ConfigType): config to update.
        env_metadata (EnvMetaDataType): metadata of the environment.

    Returns:
        ConfigType: updated config.
    """
    config = config_utils.make_config_mutable(config_utils.unset_struct(config))
    key = "type_to_select"
    if key in config.agent.encoder:
        encoder_type_to_select = config.agent.encoder[key]
        # config.agent.encoder = config.agent.encoder[encoder_type_to_select]
    else:
        encoder_type_to_select = config.agent.encoder.type
    if encoder_type_to_select in ["identity"]:
        # if the encoder is an identity encoder infer the shape of the input dim.
        config.agent.encoder_feature_dim = env_metadata["env_obs_space"].shape[0]

    key = "ordered_task_list"
    if key in env_metadata and env_metadata[key]:
        config.env.ordered_task_list = deepcopy(env_metadata[key])
    config = config_utils.make_config_immutable(config)

    return config


def get_env_metadata(
    env: EnvType,
    max_episode_steps: Optional[int] = None,
    ordered_task_list: Optional[List[str]] = None,
) -> EnvMetaDataType:
    """Method to get the metadata from an environment"""
    dummy_env = env.env_fns[0]().env
    metadata: EnvMetaDataType = {
        "env_obs_space": dummy_env.observation_space,
        "action_space": dummy_env.action_space,
        "ordered_task_list": ordered_task_list,
    }
    if max_episode_steps is None:
        metadata["max_episode_steps"] = dummy_env._max_episode_steps
    else:
        metadata["max_episode_steps"] = max_episode_steps
    return metadata
