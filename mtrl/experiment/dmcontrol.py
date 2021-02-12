# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import List

import numpy as np
from mtenv.utils.types import ObsType

from mtrl.agent import utils as agent_utils
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import multitask
from mtrl.utils.types import ConfigType, TensorType


class Experiment(multitask.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a multi-task model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(config, experiment_id)

    def get_action_when_evaluating_vec_env_of_tasks(
        self, multitask_obs: ObsType, modes: List[str]
    ) -> TensorType:
        agent = self.agent
        with agent_utils.eval_mode(agent):
            action = agent.select_action(multitask_obs=multitask_obs, modes=modes)
        return action

    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode", episode, step)

        episode_reward, mask, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False]
        ]  # (num_envs, 1)
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        offset = self.config.experiment.num_eval_episodes
        while not np.all(done):
            action = self.get_action_when_evaluating_vec_env_of_tasks(
                multitask_obs=multitask_obs, modes=vec_env.mode
            )
            multitask_obs, reward, done, _ = vec_env.step(action)
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask

        start_index = 0
        for mode in self.eval_modes_to_env_ids:
            num_envs = len(self.eval_modes_to_env_ids[mode])
            self.logger.log(
                f"{mode}/episode_reward",
                episode_reward[start_index : start_index + offset * num_envs].mean(),
                step,
            )
            for _current_env_index, _current_env_id in enumerate(
                self.eval_modes_to_env_ids[mode]
            ):
                self.logger.log(
                    f"{mode}/episode_reward_env_index_{_current_env_index}",
                    episode_reward[
                        start_index
                        + _current_env_index * offset : start_index
                        + (_current_env_index + 1) * offset
                    ].mean(),
                    step,
                )
                self.logger.log(
                    f"{mode}/env_index_{_current_env_index}", _current_env_id, step
                )
            start_index += offset * num_envs
        self.logger.dump(step)
