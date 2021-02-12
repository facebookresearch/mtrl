# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf as OC

from mtrl.agent import abstract, wrapper
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import NoneTaskInfo, TaskInfo
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer, ReplayBufferSample
from mtrl.utils.types import (
    ComponentType,
    ConfigType,
    ModelType,
    OptimizerType,
    ParameterType,
    TensorType,
)

ComponentOrOptimizerType = Union[ComponentType, OptimizerType]


def gaussian_kld(
    mean1: TensorType, logvar1: TensorType, mean2: TensorType, logvar2: TensorType
) -> TensorType:
    """Compute KL divergence between a bunch of univariate Gaussian
        distributions with the given means and log-variances.
        ie `KL(N(mean1, logvar1) || N(mean2, logvar2))`

    Args:
        mean1 (TensorType):
        logvar1 (TensorType):
        mean2 (TensorType):
        logvar2 (TensorType):

    Returns:
        TensorType: [description]
    """

    gauss_klds = 0.5 * (
        (logvar2 - logvar1)
        + ((torch.exp(logvar1) + (mean1 - mean2) ** 2.0) / torch.exp(logvar2))
        - 1.0
    )
    assert len(gauss_klds.size()) == 2
    return gauss_klds


class Agent(abstract.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        multitask_cfg: ConfigType,
        device: torch.device,
        distral_alpha: float,
        distral_beta: float,
        agent_index_to_task_index: List[str],
        distilled_agent_cfg: ConfigType,
        task_agent_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Distral algorithm."""
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )
        self.distral_alpha = distral_alpha
        self.distral_beta = distral_beta
        self.agent_index_to_task_index = agent_index_to_task_index
        # eventually, this will be done via OC.
        self.num_task_agents = len(self.agent_index_to_task_index)
        self.task_index_to_agent_index = {
            task_index: agent_index
            for agent_index, task_index in enumerate(self.agent_index_to_task_index)
        }

        self.distilled_agent = hydra.utils.instantiate(
            distilled_agent_cfg,
            env_obs_shape=self.env_obs_shape,
            action_shape=self.action_shape,
            action_range=action_range,
            multitask_cfg=OC.create({"num_envs": 1}),
            device=self.device,
            cfg_to_load_model=cfg_to_load_model,
            should_complete_init=True,
        )

        self.task_agents = [
            hydra.utils.instantiate(
                task_agent_cfg,
                env_obs_shape=self.env_obs_shape,
                action_shape=self.action_shape,
                action_range=action_range,
                multitask_cfg=OC.create({"num_envs": 1}),
                device=self.device,
                index=agent_index,
                env_index=task_index,
                distilled_agent=self.distilled_agent,
                cfg_to_load_model=cfg_to_load_model,
                should_complete_init=True,
            )
            for agent_index, task_index in enumerate(self.agent_index_to_task_index)
        ]

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def complete_init(self, cfg_to_load_model: Optional[ConfigType]) -> None:
        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        self.distilled_agent.train(training)
        [_task_agent.train(training) for _task_agent in self.task_agents]

    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        """Used during testing"""
        return self.distilled_agent.select_action(
            multitask_obs=multitask_obs, modes=modes
        )

    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        """Used during training"""
        obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        actions = [
            self.task_agents[self.task_index_to_agent_index[index]].sample_action(
                multitask_obs={
                    "env_obs": obs[self.task_index_to_agent_index[index]],
                    "task_obs": torch.LongTensor(
                        [
                            [index],
                        ]
                    ),  # not used in the actor.
                },
                modes=modes,
            )
            for index in env_index.numpy()
        ]
        actions = np.concatenate(actions, axis=0)
        return actions

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        for _task_agent in self.task_agents:
            _task_agent.update(
                replay_buffer=replay_buffer,
                logger=logger,
                step=step,
            )

    def save(
        self,
        model_dir: str,
        step: int,
        retain_last_n: int,
        should_save_metadata: bool = True,
    ) -> None:
        self.distilled_agent.save(
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            should_save_metadata=False,
        )
        for agent in self.task_agents:
            agent.save(
                model_dir=model_dir,
                step=step,
                retain_last_n=retain_last_n,
                should_save_metadata=False,
            )
        if should_save_metadata:
            self.save_metadata(model_dir, step)

    def load(self, model_dir: Optional[str], step: Optional[int]) -> None:
        self.distilled_agent.load(model_dir, step)
        for agent in self.task_agents:
            agent.load(model_dir, step)

    def load_latest_step(self, model_dir: str) -> int:
        latest_step = -1
        metadata = self.load_metadata(model_dir=model_dir)
        if metadata is None:
            return latest_step + 1
        latest_step = metadata["step"]
        self.distilled_agent.load(model_dir, step=latest_step)
        for agent in self.task_agents:
            agent.load(model_dir, step=latest_step)
        return latest_step + 1


class DistilledAgent(abstract.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        multitask_cfg: ConfigType,
        device: torch.device,
        actor_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Centroid policy for distral"""
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            # num_envs=1,
            device=device,
        )
        self._name = "distilled_agent"
        self.actor: ModelType = hydra.utils.instantiate(
            actor_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        self._components: Dict[str, ModelType] = {
            "actor": self.actor,
        }
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, self.actor.parameters()
        )
        self._optimizers: Dict[str, OptimizerType] = {
            "actor": self.actor_optimizer,
        }
        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def train(self, training=True) -> None:
        self.training = training
        self.actor.train(training)

    def complete_init(self, cfg_to_load_model: Optional[ConfigType]) -> None:
        self.train()

    def select_action(self, multitask_obs: ObsType, modes: List[str]):
        with torch.no_grad():
            env_obs = multitask_obs["env_obs"].float().to(self.device)
            if len(env_obs.shape) == 3:
                env_obs = env_obs.unsqueeze(0)  # Make a batch

            mtobs = MTObs(env_obs=env_obs, task_obs=None, task_info=NoneTaskInfo)
            mu, _, _, _ = self.actor(mtobs=mtobs)
            return mu.cpu().numpy()

    def sample_action(self, multitask_obs: ObsType, modes: List[str]):
        with torch.no_grad():
            env_obs = multitask_obs["env_obs"].float().to(self.device)
            if len(env_obs.shape) == 3:
                env_obs = env_obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=env_obs, task_obs=None, task_info=NoneTaskInfo)
            mu, pi, _, _ = self.actor(mtobs=mtobs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def save(
        self,
        model_dir: str,
        step: int,
        retain_last_n: int,
        should_save_metadata: bool = True,
    ) -> None:
        return super().save(
            model_dir=os.path.join(model_dir, self._name),
            step=step,
            retain_last_n=retain_last_n,
            should_save_metadata=should_save_metadata,
        )

    def load(self, model_dir: Optional[str], step: Optional[int]) -> None:
        if model_dir is not None:
            return super().load(
                model_dir=os.path.join(model_dir, self._name), step=step
            )
        return

    def load_latest_step(self, model_dir: str) -> int:
        latest_step = -1
        metadata = self.load_metadata(model_dir=model_dir)
        if metadata is None:
            return latest_step + 1
        latest_step = metadata["step"]
        self.load(model_dir=os.path.join(model_dir, self._name), step=latest_step)
        return latest_step + 1

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ):
        raise NotImplementedError(
            "`update` method is not implemented for distral algorithm."
        )


class TaskAgent(wrapper.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        multitask_cfg: ConfigType,
        device: torch.device,
        agent_cfg: ConfigType,
        index: int,
        env_index: int,
        distral_alpha: float,
        distral_beta: float,
        distilled_agent: DistilledAgent,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Wrapper class for the task specific agent"""
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg,
            device=device,
            cfg_to_load_model=cfg_to_load_model,
            should_complete_init=should_complete_init,
        )
        self.index = index
        self.env_index = env_index
        self.distral_alpha = distral_alpha
        self.distral_beta = distral_beta
        self._name = f"task_agent_{self.index}"
        self.patch_agent()
        self.distilled_agent = distilled_agent
        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def patch_agent(self) -> None:
        """Change some function definitions at runtime."""
        self.agent.update_actor_and_alpha = self.update_actor_and_alpha
        self.agent._get_target_V = self._get_target_V

    def _get_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
        _, policy_action, log_pi, _ = self.agent.actor(mtobs=mtobs)
        _, _, distral_log_pi, _ = self.distilled_agent.actor(mtobs=mtobs)
        target_Q1, target_Q2 = self.agent.critic_target(
            mtobs=mtobs, action=policy_action
        )
        agent_alpha = self.agent.get_alpha(batch.task_obs).detach()
        alpha_from_paper = self.distral_alpha / (self.distral_alpha + agent_alpha)
        beta_from_paper = 1.0 / (self.distral_alpha + agent_alpha)
        return (
            torch.min(target_Q1, target_Q2)
            + (alpha_from_paper * distral_log_pi - log_pi) / beta_from_paper
        )

    def update_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the actor and alpha component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        # detach encoder, so we don't update it with the actor loss
        suffix = f"_agent_index_{self.index}"
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
        mu, pi, log_pi, log_std = self.agent.actor(mtobs=mtobs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.agent.critic(
            mtobs=mtobs, action=pi, detach_encoder=True
        )

        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.agent.loss_reduction == "mean":
            actor_loss = (
                self.agent.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            ).mean()
            logger.log(f"train/actor_loss{suffix}", actor_loss, step)

        elif self.agent.loss_reduction == "none":
            actor_loss = (
                self.agent.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            )
            logger.log(f"train/actor_loss{suffix}", actor_loss.mean(), step)

        logger.log(
            f"train/actor_target_entropy{suffix}", self.agent.target_entropy, step
        )

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )

        logger.log(f"train/actor_entropy{suffix}", entropy.mean(), step)

        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=NoneTaskInfo)
        distral_mu, _, distral_log_pi, distral_log_std = self.distilled_agent.actor(
            mtobs=mtobs, detach_encoder=False
        )
        distilled_agent_loss = gaussian_kld(
            mean1=distral_mu,
            logvar1=2 * distral_log_std,
            mean2=mu.detach(),
            logvar2=2 * log_std.detach(),
        )
        batch_size = distilled_agent_loss.shape[0]
        distilled_agent_loss = torch.sum(distilled_agent_loss) / batch_size
        logger.log(
            f"train/actor_distilled_agent_loss{suffix}",
            distilled_agent_loss.mean(),
            step,
        )
        distilled_agent_loss = distilled_agent_loss * self.distral_alpha

        # optimize the actor
        component_names = ["actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self.agent._optimizers[name].zero_grad()
            parameters += self.agent.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.agent.get_parameters("task_encoder")

        self.agent._compute_gradient(
            loss=actor_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )
        self.agent.actor_optimizer.step()
        self.agent.log_alpha_optimizer.zero_grad()
        if self.agent.loss_reduction == "mean":
            alpha_loss = (
                self.agent.get_alpha(batch.task_obs)
                * (-log_pi - self.agent.target_entropy).detach()
            ).mean()
            logger.log(f"train/alpha_loss{suffix}", alpha_loss, step)
        elif self.agent.loss_reduction == "none":
            alpha_loss = (
                self.agent.get_alpha(batch.task_obs)
                * (-log_pi - self.agent.target_entropy).detach()
            )
            logger.log(f"train/alpha_loss{suffix}", alpha_loss.mean(), step)
        # logger.log("train/alpha_value", self.get_alpha(batch.task_obs), step)
        self.agent._compute_gradient(
            loss=alpha_loss,
            parameters=self.agent.get_parameters(name="log_alpha"),
            step=step,
            component_names=["log_alpha"],
            **kwargs_to_compute_gradient,
        )
        self.agent.log_alpha_optimizer.step()
        self.distilled_agent._optimizers["actor"].zero_grad()
        distilled_agent_loss.backward()
        self.distilled_agent._optimizers["actor"].step()

    def save(
        self,
        model_dir: str,
        step: int,
        retain_last_n: int,
        should_save_metadata: bool = True,
    ) -> None:
        return super().save(
            model_dir=os.path.join(model_dir, self._name),
            step=step,
            retain_last_n=retain_last_n,
            should_save_metadata=should_save_metadata,
        )

    def load(self, model_dir: Optional[str], step: Optional[int]) -> None:
        if model_dir is None or step is None:
            return
        return super().load(model_dir=os.path.join(model_dir, self._name), step=step)

    def load_latest_step(self, model_dir: str) -> int:
        latest_step = -1
        metadata = self.load_metadata(model_dir=model_dir)
        if metadata is None:
            return latest_step + 1
        latest_step = metadata["step"]
        super().load(model_dir=os.path.join(model_dir, self._name), step=latest_step)
        return latest_step + 1
