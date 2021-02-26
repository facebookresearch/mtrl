# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Actor component for the agent."""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.agent.components import encoder, moe_layer
from mtrl.agent.components.soft_modularization import SoftModularizedMLP
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.utils.types import ConfigType, ModelType, TensorType


def check_if_should_use_multi_head_policy(multitask_cfg: ConfigType) -> bool:
    if "should_use_multi_head_policy" in multitask_cfg:
        return multitask_cfg.should_use_multi_head_policy
    return False


def check_if_should_use_task_encoder(multitask_cfg: ConfigType) -> bool:
    if "should_use_task_encoder" in multitask_cfg:
        return multitask_cfg.should_use_task_encoder
    return False


def _gaussian_logprob(noise: TensorType, log_std: TensorType) -> TensorType:
    """Compute the gaussian log probability.

    Args:
        noise (TensorType):
        log_std (TensorType): [description]

    Returns:
        TensorType: log-probaility of the sample.
    """
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def _squash(
    mu: TensorType, pi: TensorType, log_pi: TensorType
) -> Tuple[TensorType, TensorType, TensorType]:
    """Apply squashing function.
        See appendix C from https://arxiv.org/pdf/1812.05905.pdf.

    Args:
        mu ([TensorType]): mean of the gaussian distribution.
        pi ([TensorType]): sample from the gaussian distribution.
        log_pi ([TensorType]): log probability.

    Returns:
        Tuple[TensorType, TensorType, TensorType]: tuple of
            (squashed mean of the gaussian, squashed sample from the gaussian,
                squashed  log-probability of the sample)
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class BaseActor(base_component.Component):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
        *args,
        **kwargs,
    ):
        """Interface for the actor component for the agent.

        Args:
            env_obs_shape (List[int]): shape of the environment observation that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor produces.
            encoder_cfg (ConfigType): config for the encoder.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        super().__init__()
        self.multitask_cfg = multitask_cfg

    def encode(self, mtobs: MTObs, detach: bool = False) -> TensorType:
        """Encode the input observation.

        Args:
            mtobs (MTObs): multi-task observation.
            detach (bool, optional): should detach the observation encoding
                from the computation graph. Defaults to False.

        Raises:
            NotImplementedError:

        Returns:
            TensorType: encoding of the observation.
        """
        raise NotImplementedError

    def forward(
        self,
        mtobs: MTObs,
        detach_encoder: bool = False,
    ) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        """Compute the predictions from the actor.

        Args:
            mtobs (MTObs): multi-task observation.
            detach_encoder (bool, optional): should detach the observation encoding
                from the computation graph. Defaults to False.

        Raises:
            NotImplementedError:

        Returns:
            Tuple[TensorType, TensorType, TensorType, TensorType]: tuple of
            (mean of the gaussian, sample from the gaussian,
                log-probability of the sample, log of standard deviation of the gaussian).
        """

        raise NotImplementedError


class Actor(BaseActor):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        hidden_dim: int,
        num_layers: int,
        log_std_bounds: Tuple[float, float],
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
    ):
        """Actor component for the agent.

        Args:
            env_obs_shape (List[int]): shape of the environment observation that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor produces.
            hidden_dim (int): hidden dimensionality of the actor.
            num_layers (int): number of layers in the actor.
            log_std_bounds (Tuple[float, float]): bounds to clip log of standard deviation.
            encoder_cfg (ConfigType): config for the encoder.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        key = "type_to_select"
        if key in encoder_cfg:
            encoder_type_to_select = encoder_cfg[key]
            encoder_cfg = encoder_cfg[encoder_type_to_select]
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
        )

        self.log_std_bounds = log_std_bounds

        self.should_use_multi_head_policy = check_if_should_use_multi_head_policy(
            multitask_cfg=multitask_cfg
        )

        if self.should_use_multi_head_policy:
            task_index_to_mask = torch.eye(multitask_cfg.num_envs)
            self.moe_masks = moe_layer.MaskCache(
                task_index_to_mask=task_index_to_mask,
                **multitask_cfg.multi_head_policy_cfg.mask_cfg,
            )

        if check_if_should_use_task_encoder(multitask_cfg):
            self.should_condition_model_on_task_info = False
            self.should_condition_encoder_on_task_info = True
            self.should_concatenate_task_info_with_encoder = True
            if "actor_cfg" in multitask_cfg and multitask_cfg.actor_cfg:
                self.should_condition_model_on_task_info = (
                    multitask_cfg.actor_cfg.should_condition_model_on_task_info
                )
                self.should_condition_encoder_on_task_info = (
                    multitask_cfg.actor_cfg.should_condition_encoder_on_task_info
                )
                self.should_concatenate_task_info_with_encoder = (
                    multitask_cfg.actor_cfg.should_concatenate_task_info_with_encoder
                )

        else:
            self.should_condition_model_on_task_info = False
            self.should_condition_encoder_on_task_info = False
            self.should_concatenate_task_info_with_encoder = False

        self.encoder = self._make_encoder(
            env_obs_shape=env_obs_shape,
            encoder_cfg=encoder_cfg,
            multitask_cfg=multitask_cfg,
        )
        self.model = self.make_model(
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

    def _make_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        multitask_cfg: ConfigType,
    ) -> ModelType:
        """Make the head of the actor.

        Args:
            input_dim (int):
            hidden_dim (int):
            output_dim (int):
            num_layers (int):
            multitask_cfg (ConfigType):

        Returns:
            ModelType: head
        """
        return moe_layer.FeedForward(
            num_experts=multitask_cfg.num_envs,
            in_features=input_dim,
            out_features=output_dim,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            bias=True,
        )

    def _make_trunk(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        multitask_cfg: ConfigType,
    ) -> ModelType:
        if (
            "actor_cfg" in multitask_cfg
            and multitask_cfg.actor_cfg
            and "moe_cfg" in multitask_cfg.actor_cfg
            and multitask_cfg.actor_cfg.moe_cfg.should_use
        ):
            moe_cfg = multitask_cfg.actor_cfg.moe_cfg
            if moe_cfg.mode == "soft_modularization":
                trunk = SoftModularizedMLP(
                    num_experts=moe_cfg.num_experts,
                    in_features=input_dim,
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
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
            )
            # This seems to be a false alarm since both nn.Module and
            # SoftModularizedMLP are subtypes of ModelType.
        return trunk

    def make_model(
        self,
        action_shape: List[int],
        hidden_dim: int,
        num_layers: int,
        encoder_cfg: ConfigType,
        multitask_cfg: ConfigType,
    ) -> ModelType:
        """Make the model for the actor.

        Args:
            action_shape (List[int]):
            hidden_dim (int):
            num_layers (int):
            encoder_cfg (ConfigType):
            multitask_cfg (ConfigType):

        Returns:
            ModelType: model for the actor.
        """
        model_output_dim = 2 * action_shape[0]
        if encoder_cfg.type in ["moe", "fmoe"]:
            model_input_dim = encoder_cfg.encoder_cfg.feature_dim
        else:
            model_input_dim = encoder_cfg.feature_dim
        if (
            "should_use_task_encoder" in multitask_cfg
            and multitask_cfg.should_use_task_encoder
            and self.should_condition_encoder_on_task_info
            and self.should_concatenate_task_info_with_encoder
        ):
            model_input_dim += multitask_cfg.task_encoder_cfg.model_cfg.output_dim

        if self.should_use_multi_head_policy:
            if multitask_cfg.should_use_disjoint_policy:
                heads = self._make_head(
                    input_dim=model_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=model_output_dim,
                    num_layers=num_layers,
                    multitask_cfg=multitask_cfg,
                )
                return heads
            else:
                heads = self._make_head(
                    input_dim=hidden_dim,
                    output_dim=model_output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=2,
                    multitask_cfg=multitask_cfg,
                )
                trunk = self._make_trunk(
                    input_dim=model_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_layers=num_layers,
                    multitask_cfg=multitask_cfg,
                )
                return nn.Sequential(trunk, nn.ReLU(), heads)
        else:
            trunk = self._make_trunk(
                input_dim=model_input_dim,
                hidden_dim=hidden_dim,
                output_dim=model_output_dim,
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

    def encode(self, mtobs: MTObs, detach: bool = False) -> TensorType:
        encoding = self.encoder(mtobs=mtobs, detach=detach)
        task_info = mtobs.task_info
        if self.should_concatenate_task_info_with_encoder:
            return torch.cat((encoding, task_info.encoding), dim=1)  # type: ignore[arg-type, union-attr]
            # mypy is raising a false alarm. task_info is not None
        return encoding

    def forward(
        self,
        mtobs: MTObs,
        detach_encoder: bool = False,
    ) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
        task_info = mtobs.task_info
        assert task_info is not None
        if self.should_condition_encoder_on_task_info:
            obs = self.encode(mtobs=mtobs, detach=detach_encoder)
        else:
            # making a new task_info since we do not want to condition on
            # the task encoding.
            temp_task_info = TaskInfo(
                encoding=None,
                compute_grad=task_info.compute_grad,
                env_index=task_info.env_index,
            )
            temp_mtobs = MTObs(
                env_obs=mtobs.env_obs, task_obs=mtobs.task_obs, task_info=temp_task_info
            )
            obs = self.encode(temp_mtobs, detach=detach_encoder)
        if self.should_condition_model_on_task_info:
            new_mtobs = MTObs(
                env_obs=obs, task_obs=mtobs.task_obs, task_info=mtobs.task_info
            )
            mu_and_log_std = self.model(new_mtobs)
        else:
            mu_and_log_std = self.model(obs)
        if self.should_use_multi_head_policy:
            policy_mask = self.moe_masks.get_mask(task_info=task_info)
            sum_of_masked_mu_and_log_std = (mu_and_log_std * policy_mask).sum(dim=0)
            sum_of_policy_count = policy_mask.sum(dim=0)
            mu_and_log_std = sum_of_masked_mu_and_log_std / sum_of_policy_count
        mu, log_std = mu_and_log_std.chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        log_pi = _gaussian_logprob(noise, log_std)

        mu, pi, log_pi = _squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std
