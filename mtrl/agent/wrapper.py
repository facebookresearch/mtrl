# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch

from mtrl.agent import abstract
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer
from mtrl.utils.types import ComponentType, ConfigType, ModelType, OptimizerType

ComponentOrOptimizerType = Union[ComponentType, OptimizerType]


class Agent(abstract.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        multitask_cfg: ConfigType,
        agent_cfg: ConfigType,
        device: torch.device,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """This wrapper agent wraps over the other agents. It is useful
        for alogorithms like PCGrad and GradNorm that can be used with
        may policies.

        Args:

            env_obs_shape (List[int]): shape of the environment observation
                that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor
                produces.
            action_range (Tuple[int, int]): min and max values for the action vector.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            agent_cfg (ConfigType): config for the agents that are wrapper over.
            device (torch.device): device for the agent.
            cfg_to_load_model (Optional[ConfigType], optional): config to
                load the model from filesystem. Defaults to None.
            should_complete_init (bool, optional): should call `complete_init`
                method. Defaults to True.
        """
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )
        self.agent = hydra.utils.instantiate(
            agent_cfg,
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
        )

    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        self.agent.complete_init(cfg_to_load_model=cfg_to_load_model)
        self.train()

    def train(self, training: bool = True):
        self.training = training
        self.agent.train(training=training)

    def select_action(self, multitask_obs: ObsType, modes: List[str]):
        return self.agent.select_action(multitask_obs=multitask_obs, modes=modes)

    def sample_action(self, multitask_obs: ObsType, modes: List[str]):
        return self.agent.sample_action(multitask_obs=multitask_obs, modes=modes)

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ):
        return self.agent.update(
            replay_buffer=replay_buffer,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=kwargs_to_compute_gradient,
            buffer_index_to_sample=buffer_index_to_sample,
        )

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:
        return self.agent.get_last_shared_layers(component_name=component_name)

    def get_component_name_list_for_checkpointing(
        self,
    ) -> List[Tuple[ComponentType, str]]:
        return self.agent.get_component_name_list_for_checkpointing()

    def get_optimizer_name_list_for_checkpointing(
        self,
    ) -> List[Tuple[OptimizerType, str]]:
        return self.agent.get_optimizer_name_list_for_checkpointing()

    def save(
        self,
        model_dir: str,
        step: int,
        retain_last_n: int,
        should_save_metadata: bool = True,
    ) -> None:
        return self.agent.save(
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            should_save_metadata=should_save_metadata,
        )

    def save_components(self, model_dir: str, step: int, retain_last_n: int) -> None:
        return self.agent.save_components(
            model_dir=model_dir, step=step, retain_last_n=retain_last_n
        )

    def save_optimizers(self, model_dir: str, step: int, retain_last_n: int) -> None:
        return self.save_optimizers(
            model_dir=model_dir, step=step, retain_last_n=retain_last_n
        )

    def load(self, model_dir: Optional[str], step: Optional[int]) -> None:
        return self.agent.load(model_dir=model_dir, step=step)
