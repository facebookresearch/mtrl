# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F

from mtrl.agent import sac
from mtrl.agent import utils as agent_utils
from mtrl.agent.components.decoder import make_decoder
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBufferSample
from mtrl.utils.types import ConfigType, ParameterType


class Agent(sac.Agent):
    """SAC+AE algorithm."""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        decoder_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        decoder_optimizer_cfg: ConfigType,
        encoder_optimizer_cfg: ConfigType,
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
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
            alpha_optimizer_cfg=alpha_optimizer_cfg,
            actor_optimizer_cfg=actor_optimizer_cfg,
            critic_optimizer_cfg=critic_optimizer_cfg,
            multitask_cfg=multitask_cfg,
            discount=discount,
            init_temperature=init_temperature,
            actor_update_freq=actor_update_freq,
            critic_tau=critic_tau,
            critic_target_update_freq=critic_target_update_freq,
            encoder_tau=encoder_tau,
            loss_reduction=loss_reduction,
            cfg_to_load_model=None,
            should_complete_init=False,
        )

        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        decoder_type = decoder_cfg.type
        if decoder_type != "identity":
            # create decoder
            self.decoder = make_decoder(
                env_obs_shape=env_obs_shape,
                decoder_cfg=decoder_cfg,
                multitask_cfg=multitask_cfg,
            ).to(device)
            self.decoder.apply(agent_utils.weight_init)

            self._components.update({"decoder": self.decoder})
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = hydra.utils.instantiate(
                encoder_optimizer_cfg, params=self.get_parameters(name="encoder")
            )

            self.decoder_optimizer = hydra.utils.instantiate(
                decoder_optimizer_cfg, params=self.get_parameters(name="decoder")
            )

            self._optimizers.update(
                {"decoder": self.decoder_optimizer, "encoder": self.encoder_optimizer}
            )

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def update_decoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        obs = batch.env_obs
        target_obs = batch.env_obs
        mtobs = MTObs(env_obs=obs, task_obs=None, task_info=task_info)
        h = self.critic.encode(mtobs=mtobs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = agent_utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
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
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        component_names = ["encoder", "decoder"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient
        )

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if self.loss_reduction == "mean":
            loss_to_log = loss
        elif self.loss_reduction == "none":
            loss_to_log = loss.mean()
        logger.log("train/ae_loss", loss_to_log, step)
