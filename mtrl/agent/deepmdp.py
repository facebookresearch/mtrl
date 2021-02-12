# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F

from mtrl.agent import sac_ae
from mtrl.agent import utils as agent_utils
from mtrl.agent.components import reward_decoder
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBufferSample
from mtrl.utils.types import ConfigType, ParameterType

LOG_FREQ = 10000


class Agent(sac_ae.Agent):
    """DeepMDP Agent"""

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
            alpha_optimizer_cfg=alpha_optimizer_cfg,
            actor_optimizer_cfg=actor_optimizer_cfg,
            critic_optimizer_cfg=critic_optimizer_cfg,
            multitask_cfg=multitask_cfg,
            decoder_optimizer_cfg=decoder_optimizer_cfg,
            encoder_optimizer_cfg=encoder_optimizer_cfg,
            discount=discount,
            init_temperature=init_temperature,
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
        self.is_encoder_identity = actor_cfg.encoder_cfg.type == "identity"
        if not self.is_encoder_identity:
            self.encoder_optimizer = hydra.utils.instantiate(
                encoder_optimizer_cfg, params=self.get_parameters(name="encoder")
            )
            self._optimizers["encoder"] = self.encoder_optimizer

        self.transition_model = hydra.utils.call(
            transition_model_cfg, action_shape=action_shape, multitask_cfg=multitask_cfg
        ).to(device)

        # in future, we should move it to a spearate function
        reward_feature_dim = reward_decoder_cfg.feature_dim
        if multitask_cfg.should_use_task_encoder:
            reward_feature_dim += multitask_cfg.task_encoder_cfg.model_cfg.output_dim

        self.reward_decoder = reward_decoder.RewardDecoder(
            feature_dim=reward_feature_dim,
        ).to(device)

        self._components.update(
            {
                "reward_decoder": self.reward_decoder,
                "transition_model": self.transition_model,
            }
        )

        self.reward_decoder_optimizer = hydra.utils.instantiate(
            reward_decoder_optimizer_cfg,
            params=self.get_parameters(name="reward_decoder"),
        )

        self.transition_model_optimizer = hydra.utils.instantiate(
            transition_model_optimizer_cfg,
            params=self.get_parameters(name="transition_model"),
        )

        self._optimizers.update(
            {
                "reward_decoder": self.reward_decoder_optimizer,
                "transition_model": self.transition_model_optimizer,
            }
        )

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def update_transition_reward_model(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ):
        obs = batch.env_obs
        action = batch.action
        next_obs = batch.next_env_obs
        reward = batch.reward
        mtobs = MTObs(env_obs=obs, task_obs=None, task_info=task_info)
        h = self.critic.encode(mtobs=mtobs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(
            torch.cat([h, action], dim=1)
        )
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        mtobs = MTObs(env_obs=next_obs, task_obs=None, task_info=task_info)

        next_h = self.critic.encode(mtobs=mtobs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        if self.loss_reduction == "mean":
            loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
            loss_to_log = loss
        elif self.loss_reduction == "none":
            loss = (0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma)).mean(
                dim=1, keepdim=True
            )
            loss_to_log = loss.mean()
        logger.log("train/ae_transition_loss", loss_to_log, step)

        pred_next_latent = self.transition_model.sample_prediction(
            torch.cat([h, action], dim=1)
        )
        pred_next_reward = self.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(
            pred_next_reward, reward, reduction=self.loss_reduction
        )
        total_loss = loss + reward_loss

        parameters: List[ParameterType] = []
        component_names = [
            "transition_model",
            "reward_decoder",
        ]
        if not self.is_encoder_identity:
            component_names.append("encoder")
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name=name)
        component_names_to_pass = deepcopy(component_names)
        if task_info.compute_grad:
            component_names_to_pass.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=total_loss,
            # oooh this order is very important
            parameters=parameters,
            step=step,
            component_names=component_names_to_pass,
            **kwargs_to_compute_gradient,
        )
        for name in component_names:
            self._optimizers[name].step()

    def update_decoder(  # type: ignore[override]
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ):
        obs = batch.env_obs
        action = batch.action
        target_obs = batch.next_env_obs
        #  uses transition model
        mtobs = MTObs(env_obs=obs, task_obs=None, task_info=task_info)
        h = self.critic.encode(mtobs=mtobs)
        next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = agent_utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(next_h)  # type: ignore[misc]
        if self.loss_reduction == "mean":
            rec_loss = F.mse_loss(target_obs, rec_obs).mean()
        elif self.loss_reduction == "none":
            rec_loss = F.mse_loss(target_obs, rec_obs, reduction="none")
            rec_loss = rec_loss.view(rec_loss.shape[0], -1).mean(dim=1, keepdim=True)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf

        if self.loss_reduction == "mean":
            latent_loss = (0.5 * h.pow(2).sum(1)).mean()
        elif self.loss_reduction == "none":
            latent_loss = 0.5 * h.pow(2).sum(1, keepdim=True)

        loss = rec_loss + self.decoder_latent_lambda * latent_loss

        component_names = ["transition_model", "decoder"]
        if not self.is_encoder_identity:
            component_names.append("encoder")
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name=name)
        component_names_to_pass = deepcopy(component_names)
        if task_info.compute_grad:
            component_names_to_pass.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss,
            parameters=parameters,
            step=step,
            component_names=component_names_to_pass,
            **kwargs_to_compute_gradient,
        )

        for name in component_names:
            self._optimizers[name].step()

        if self.loss_reduction == "mean":
            loss_to_log = loss
        elif self.loss_reduction == "none":
            loss_to_log = loss.mean()
        logger.log("train/ae_loss", loss_to_log, step)
