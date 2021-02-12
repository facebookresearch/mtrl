# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from mtrl.agent import wrapper
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer
from mtrl.utils.types import ComponentType, ConfigType, OptimizerType, TensorType

ComponentOrOptimizerType = Union[ComponentType, OptimizerType]


@dataclass
class EnvMetadata:
    env_index: torch.Tensor
    unique_env_index: torch.Tensor
    env_index_count: torch.Tensor


class Agent(wrapper.Agent):
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
        """Base Class for Gradient Manipulation Algorithms."""
        agent_cfg_copy = deepcopy(agent_cfg)
        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"
        OmegaConf.set_struct(agent_cfg_copy, True)

        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg_copy,
            device=device,
        )
        self.agent._compute_gradient = self._compute_gradient
        self.component_name_sep = "-"
        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch = replay_buffer.sample()
        env_index = batch.task_obs
        unique_env_index, env_index_count = env_index.unique(dim=0, return_counts=True)
        env_metadata = EnvMetadata(
            env_index=env_index,
            unique_env_index=unique_env_index,
            env_index_count=env_index_count,
        )

        self.agent.update(
            replay_buffer=replay_buffer,
            logger=logger,
            step=step,
            buffer_index_to_sample=batch.buffer_index,
            kwargs_to_compute_gradient={
                "retain_graph": False,
                "allow_unused": False,
                "env_metadata": env_metadata,
            },
        )

        return batch.buffer_index

    def _convert_loss_into_task_loss(
        self, loss: TensorType, env_metadata: EnvMetadata
    ) -> TensorType:
        """Map the aggregated loss into per-task losses.

        Args:
            loss (TensorType): loss corresponding to all the tasks.
            env_metadata (EnvMetadata): environment metadata needed for
                computing the loss for different environments/tasks.

        Returns:
            TensorType: tensor of per-task loss.
        """
        # num_tasks = 50
        num_tasks = int(env_metadata.unique_env_index.max().item() + 1)
        # Adding a one since we start counting  with 0.
        task_loss = torch.zeros(
            (num_tasks, 1), dtype=torch.float, device=self.device
        ).scatter_add_(0, env_metadata.env_index, loss)
        task_loss = task_loss[env_metadata.unique_env_index.squeeze(1)]
        task_loss = task_loss / env_metadata.env_index_count.float().unsqueeze(1)
        return task_loss

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:
        """Method to override the gradient computation.

            Useful for algorithms like PCGrad and GradNorm.

        Args:
            loss (TensorType):
            parameters (List[ParameterType]):
            step (int): step for tracking the training of the agent.
            component_names (List[str]):
            env_metadata (EnvMetadata): environment metadata needed for
                computing the loss for different environments/tasks.
            retain_graph (bool, optional): if it should retain graph.
                Defaults to False.
            allow_unused (bool, optional): if False, specifying inputs
                that were not used when computing outputs (and therefore
                their grad is always zero) is an error. Defaults to False.
        """

        raise NotImplementedError(
            """The function is not implemented by `grad_manipulation agent`.
            It should be implemented by the agents subclassing it."""
        )

    def _join_componenet_names(self, component_names: List[str]) -> str:
        return self.component_name_sep.join(component_names)
