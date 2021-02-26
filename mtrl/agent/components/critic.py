# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Critic component for the agent."""

from typing import List, Tuple

import torch
from torch import nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.agent.components import encoder, moe_layer
from mtrl.agent.components.actor import (
    check_if_should_use_multi_head_policy,
    check_if_should_use_task_encoder,
)
from mtrl.agent.components.soft_modularization import SoftModularizedMLP
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import ConfigType, ModelType, TensorType


class QFunction(base_component.Component):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        multitask_cfg: ConfigType,
    ):
        """Q-function implemented as a MLP.

        Args:
            obs_dim (int): size of the observation.
            action_dim (int): size of the action vector.
            hidden_dim (int): size of the hidden layer of the model.
            num_layers (int): number of layers in the model.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        super().__init__()

        self.should_condition_model_on_task_info = False
        self.should_condition_encoder_on_task_info = True

        if "critic_cfg" in multitask_cfg and multitask_cfg.critic_cfg:
            self.should_condition_model_on_task_info = (
                multitask_cfg.critic_cfg.should_condition_model_on_task_info
            )
            self.should_condition_encoder_on_task_info = (
                multitask_cfg.critic_cfg.should_condition_encoder_on_task_info
            )

        self.should_use_multi_head_policy = check_if_should_use_multi_head_policy(
            multitask_cfg=multitask_cfg
        )

        self.model = self.build_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            multitask_cfg=multitask_cfg,
        )

        if self.should_condition_model_on_task_info:
            # layer to project obs_action to shape of obs_dim
            self.obs_action_projection_layer = nn.Linear(
                in_features=obs_dim + action_dim, out_features=obs_dim
            )

    def _make_head(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        multitask_cfg: ConfigType,
    ) -> ModelType:
        """Make the heads for the Q-function.

        Args:
            input_dim (int): size of the input.
            hidden_dim (int): size of the hidden layer of the head.
            num_layers (int): number of layers in the model.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        Returns:
            ModelType:
        """
        return moe_layer.FeedForward(
            num_experts=multitask_cfg.num_envs,
            in_features=input_dim,
            out_features=1,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            bias=True,
        )

    def _make_trunk(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        multitask_cfg: ConfigType,
    ) -> ModelType:
        """Make the tunk for the Q-function.

        Args:
            obs_dim (int): size of the observation.
            action_dim (int): size of the action vector.
            hidden_dim (int): size of the hidden layer of the trunk.
            output_dim (int): size of the output.
            num_layers (int): number of layers in the model.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        Returns:
            ModelType:
        """
        if (
            "critic_cfg" in multitask_cfg
            and multitask_cfg.critic_cfg
            and "moe_cfg" in multitask_cfg.critic_cfg
            and multitask_cfg.critic_cfg.moe_cfg.should_use
        ):
            moe_cfg = multitask_cfg.critic_cfg.moe_cfg
            if moe_cfg.mode == "soft_modularization":
                trunk = SoftModularizedMLP(
                    num_experts=moe_cfg.num_experts,
                    in_features=obs_dim,
                    out_features=output_dim,
                    num_layers=2,
                    hidden_features=hidden_dim,
                    bias=True,
                )
            else:
                raise NotImplementedError(
                    f"""`moe_cfg.mode` = {moe_cfg.mode} is not implemented."""
                )

        else:
            trunk = agent_utils.build_mlp(  # type: ignore[assignment]
                input_dim=obs_dim + action_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
            )
            # This seems to be a false alarm since both nn.Module and
            # SoftModularizedMLP are subtypes of ModelType.
        return trunk

    def build_model(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        multitask_cfg: ConfigType,
    ) -> ModelType:
        """Build the Q-Function.

        Args:
            obs_dim (int): size of the observation.
            action_dim (int): size of the action vector.
            hidden_dim (int): size of the hidden layer of the trunk.
            num_layers (int): number of layers in the model.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.

        Returns:
            ModelType:
        """
        if self.should_use_multi_head_policy:
            if multitask_cfg.should_use_disjoint_policy:
                heads = self._make_head(
                    input_dim=obs_dim + action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    multitask_cfg=multitask_cfg,
                )
                return heads
            else:
                heads = self._make_head(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=2,
                    multitask_cfg=multitask_cfg,
                )
                trunk = self._make_trunk(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_layers=num_layers,
                    multitask_cfg=multitask_cfg,
                )
                return nn.Sequential(trunk, nn.ReLU(), heads)
        else:
            trunk = self._make_trunk(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_layers,
                multitask_cfg=multitask_cfg,
            )
            return trunk

    def get_last_shared_layers(self) -> List[ModelType]:
        if self.should_use_multi_head_policy:
            # the trunk is the first element in `self.model` and is also the last
            # shared component.
            return [self.model[0][-1]]  # type: ignore[index]
        else:
            return [self.model[-1]]  # type: ignore[index]

    def forward(self, mtobs: MTObs) -> TensorType:
        obs_action = mtobs.env_obs
        if self.should_condition_model_on_task_info:
            obs_action = self.obs_action_projection_layer(obs_action)
            new_mtobs = MTObs(
                env_obs=obs_action, task_obs=mtobs.task_obs, task_info=mtobs.task_info
            )
            return self.model(new_mtobs)
        else:
            return self.model(obs_action)


class Critic(base_component.Component):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        hidden_dim: int,
        num_layers: int,
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
    ):
        """Critic component for the agent.

        Args:
            env_obs_shape (List[int]): shape of the environment observation that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor produces.
            hidden_dim (int): hidden dimensionality of the actor.
            num_layers (int): number of layers in the actor.
            encoder_cfg (ConfigType): config for the encoder.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        key = "type_to_select"
        if key in encoder_cfg:
            encoder_type_to_select = encoder_cfg[key]
            encoder_cfg = encoder_cfg[encoder_type_to_select]

        super().__init__()

        if check_if_should_use_task_encoder(multitask_cfg):
            self.should_condition_model_on_task_info = False
            self.should_condition_encoder_on_task_info = True
            self.should_concatenate_task_info_with_encoder = True
            if "critic_cfg" in multitask_cfg and multitask_cfg.critic_cfg:
                self.should_condition_model_on_task_info = (
                    multitask_cfg.critic_cfg.should_condition_model_on_task_info
                )
                self.should_condition_encoder_on_task_info = (
                    multitask_cfg.critic_cfg.should_condition_encoder_on_task_info
                )
                self.should_concatenate_task_info_with_encoder = (
                    multitask_cfg.critic_cfg.should_concatenate_task_info_with_encoder
                )

        else:
            self.should_condition_model_on_task_info = False
            self.should_condition_encoder_on_task_info = False
            self.should_concatenate_task_info_with_encoder = False

        self.encoder = self._make_encoder(
            env_obs_shape=env_obs_shape,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
            # **kwargs
        )

        self.should_use_multi_head_policy = check_if_should_use_multi_head_policy(
            multitask_cfg=multitask_cfg
        )

        if self.should_use_multi_head_policy:
            task_index_to_mask = torch.eye(multitask_cfg.num_envs)
            self.moe_masks = moe_layer.MaskCache(
                task_index_to_mask=task_index_to_mask,
                **multitask_cfg.multi_head_policy_cfg.mask_cfg,
            )

        self.Q1 = self._make_qfunction(
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
        )

        self.Q2 = self._make_qfunction(
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
        )

        self.apply(agent_utils.weight_init)

    def _make_encoder(
        self,
        env_obs_shape: List[int],
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
        **kwargs: ConfigType,
    ) -> encoder.Encoder:
        """Make the encoder.

        Args:
            env_obs_shape (List[int]):
            encoder_cfg (ConfigType):
            multitask_cfg (ConfigType):

        Returns:
            encoder.Encoder: encoder
        """
        return encoder.make_encoder(
            env_obs_shape=env_obs_shape,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
        )

    def _make_qfunction(
        self,
        action_shape: List[int],
        hidden_dim: int,
        num_layers: int,
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
    ) -> QFunction:
        """Make the QFunction.

        Args:
            action_shape (List[int]):
            hidden_dim (int):
            num_layers (int):
            encoder_cfg (ConfigType):
            multitask_cfg (ConfigType):

        Returns:
            QFunction:
        """
        key = "type_to_select"
        if key in encoder_cfg:
            encoder_type_to_select = encoder_cfg[key]
            encoder_cfg = encoder_cfg[encoder_type_to_select]
        if encoder_cfg.type in ["moe", "fmoe"]:
            obs_dim = encoder_cfg.encoder_cfg.feature_dim
        else:
            obs_dim = encoder_cfg.feature_dim
        if (
            multitask_cfg.should_use_task_encoder
            and self.should_condition_encoder_on_task_info
        ):
            obs_dim += multitask_cfg.task_encoder_cfg.model_cfg.output_dim

        return QFunction(
            obs_dim=obs_dim,
            action_dim=action_shape[0],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            multitask_cfg=multitask_cfg,
        )

    def encode(
        self,
        mtobs: MTObs,
        detach: bool = False,
    ) -> TensorType:
        """Encode the input observation.

        Args:
            mtobs (MTObs): multi-task observation.
            detach (bool, optional): should detach the observation encoding
                from the computation graph. Defaults to False.

        Returns:
            TensorType: encoding of the observation.
        """
        encoding = self.encoder(mtobs=mtobs, detach=detach)
        task_info = mtobs.task_info
        if self.should_concatenate_task_info_with_encoder:
            return torch.cat((encoding, task_info.encoding), dim=1)  # type: ignore[arg-type, union-attr]
            # mypy is raising a false alarm. task_info is not None
        return encoding

    def get_last_shared_layers(self) -> List[ModelType]:
        last_shared_layers: List[ModelType] = []
        for q in [self.Q1, self.Q2]:
            last_shared_layers += q.get_last_shared_layers()
        return last_shared_layers

    def forward(
        self,
        mtobs: MTObs,
        action: TensorType,
        detach_encoder: bool = False,
    ) -> Tuple[TensorType, TensorType]:

        task_info = mtobs.task_info
        assert task_info is not None
        # detach_encoder allows to stop gradient propogation to encoder
        if self.should_condition_encoder_on_task_info:
            obs = self.encode(mtobs=mtobs, detach=detach_encoder)
        else:
            # making a new task_info since we do not want to condition on
            # # the task encoding.
            temp_task_info = TaskInfo(
                encoding=None,
                compute_grad=task_info.compute_grad,
                env_index=task_info.env_index,
            )
            temp_mtobs = MTObs(
                env_obs=mtobs.env_obs, task_obs=mtobs.task_obs, task_info=temp_task_info
            )
            obs = self.encode(mtobs=temp_mtobs, detach=detach_encoder)

        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        mtobs_for_q = MTObs(
            env_obs=obs_action,
            task_obs=mtobs.task_obs,
            task_info=mtobs.task_info,
        )
        q1 = self.Q1(mtobs=mtobs_for_q)
        q2 = self.Q2(mtobs=mtobs_for_q)
        if self.should_use_multi_head_policy:
            q_mask = self.moe_masks.get_mask(task_info=task_info)
            sum_of_q_count = q_mask.sum(dim=0)
            sum_of_q1 = (q1 * q_mask).sum(dim=0)
            q1 = sum_of_q1 / sum_of_q_count
            sum_of_q2 = (q2 * q_mask).sum(dim=0)
            q2 = sum_of_q2 / sum_of_q_count

        return q1, q2
