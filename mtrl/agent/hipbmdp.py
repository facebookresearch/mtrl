# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from mtrl.agent import deepmdp
from mtrl.agent.components import hipbmdp_theta
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.replay_buffer import ReplayBufferSample
from mtrl.utils.types import ConfigType, ParameterType, TensorType

LOG_FREQ = 10000


class Agent(deepmdp.Agent):
    """HiPBMDP Agent"""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        decoder_cfg: ConfigType,
        reward_decoder_cfg: ConfigType,
        transition_model_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        decoder_optimizer_cfg: ConfigType,
        encoder_optimizer_cfg: ConfigType,
        reward_decoder_optimizer_cfg: ConfigType,
        transition_model_optimizer_cfg: ConfigType,
        discount: float = 0.99,
        init_temperature: float = 0.01,
        actor_update_freq: int = 2,
        critic_tau: float = 0.005,
        critic_target_update_freq: int = 2,
        encoder_tau: float = 0.005,
        loss_reduction: str = "mean",
        decoder_update_freq: int = 1,
        decoder_latent_lambda: float = 0.0,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
            actor_cfg=actor_cfg,
            critic_cfg=critic_cfg,
            decoder_cfg=decoder_cfg,
            transition_model_cfg=transition_model_cfg,
            reward_decoder_cfg=reward_decoder_cfg,
            discount=discount,
            init_temperature=init_temperature,
            alpha_optimizer_cfg=alpha_optimizer_cfg,
            actor_optimizer_cfg=actor_optimizer_cfg,
            critic_optimizer_cfg=critic_optimizer_cfg,
            multitask_cfg=multitask_cfg,
            decoder_optimizer_cfg=decoder_optimizer_cfg,
            encoder_optimizer_cfg=encoder_optimizer_cfg,
            reward_decoder_optimizer_cfg=reward_decoder_optimizer_cfg,
            transition_model_optimizer_cfg=transition_model_optimizer_cfg,
            actor_update_freq=actor_update_freq,
            critic_tau=critic_tau,
            critic_target_update_freq=critic_target_update_freq,
            encoder_tau=encoder_tau,
            loss_reduction=loss_reduction,
            decoder_update_freq=decoder_update_freq,
            decoder_latent_lambda=decoder_latent_lambda,
            cfg_to_load_model=None,
            should_complete_init=False,
        )
        self._cache_theta_sampling_strategy: Dict[
            str, List[hipbmdp_theta.ThetaSamplingStrategy]
        ] = {}
        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _get_theta_sampling_strategy(self, modes: List[str]):
        if modes[0] not in self._cache_theta_sampling_strategy:
            if modes[0] == "train":
                strategy = self.multitask_cfg.task_encoder_cfg.sampling_strategy[
                    "train"
                ]
                theta_sampling_strategy = [
                    hipbmdp_theta.ThetaSamplingStrategy(strategy) for _ in modes
                ]
            elif modes[0] == "base":
                theta_sampling_strategy = [
                    hipbmdp_theta.ThetaSamplingStrategy(
                        self.multitask_cfg.task_encoder_cfg.sampling_strategy["eval"][
                            submode
                        ]
                    )
                    for submode in modes
                ]
            else:
                raise ValueError(f"`mode`={modes[0]} is not supported")
            assert len(modes) == len(theta_sampling_strategy)
            self._cache_theta_sampling_strategy[modes[0]] = theta_sampling_strategy
        return self._cache_theta_sampling_strategy[modes[0]]

    def get_task_encoding(
        self, env_index: TensorType, modes: List[str], disable_grad: bool
    ):

        theta_sampling_strategy = self._get_theta_sampling_strategy(modes=modes)

        if disable_grad:
            with torch.no_grad():
                return self.task_encoder(
                    env_index=env_index.to(self.device),
                    theta_sampling_strategy=theta_sampling_strategy,
                    modes=modes,
                )
        return self.task_encoder(
            env_index=env_index.to(self.device),
            theta_sampling_strategy=theta_sampling_strategy,
            modes=modes,
        )

    def update_task_encoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger,
        step,
        kwargs_to_compute_gradient: Dict[str, Any],
    ):
        if not self.multitask_cfg.contrastive.should_use:
            return
        h = self.critic.encoder(batch.env_obs, task_info=task_info)
        flipped_task_encoding = torch.flip(task_info.encoding, [0])  # type: ignore[arg-type]

        h_1 = torch.cat([h, task_info.encoding], dim=1)  # type: ignore[list-item]
        h_2 = torch.cat([h, flipped_task_encoding], dim=1)

        pred_next_latent_mu_1, pred_next_latent_sigma_1 = self.transition_model(
            torch.cat([h_1, batch.action], dim=1)
        )

        pred_next_latent_mu_2, pred_next_latent_sigma_2 = self.transition_model(
            torch.cat([h_2, batch.action], dim=1)
        )

        loss = F.mse_loss(
            torch.norm(task_info.encoding - flipped_task_encoding),
            torch.norm(pred_next_latent_mu_1.detach() - pred_next_latent_mu_2.detach()),
        )

        logger.log("train/contrastive_loss", loss, step)
        loss *= self.multitask_cfg.contrastive.alpha

        # optimize the task_encoder
        component_names = ["task_encoder"]
        parameters: List[ParameterType] = []
        for name in component_names:
            parameters += self.get_parameters(name)
        self._compute_gradient(
            loss=loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        self.task_encoder_optimizer.step()
