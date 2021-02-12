# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import Dict

import hydra
import numpy as np

from mtrl.agent import utils as agent_utils
from mtrl.env import builder as env_builder
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import multitask
from mtrl.utils.types import ConfigType


class Experiment(multitask.Experiment):
    """Experiment Class"""

    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        super().__init__(config, experiment_id)
        self.should_reset_env_manually = True

    def create_eval_modes_to_env_ids(self):
        eval_modes_to_env_ids = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            if self.config.env.benchmark._target_ in [
                "metaworld.ML1",
                "metaworld.MT1",
                "metaworld.MT10",
                "metaworld.MT50",
            ]:
                eval_modes_to_env_ids[mode] = list(range(self.config.env.num_envs))
            else:
                raise ValueError(
                    f"`{self.config.env.benchmark._target_}` env is not supported by metaworld experiment."
                )
        return eval_modes_to_env_ids

    def build_envs(self):
        benchmark = hydra.utils.instantiate(self.config.env.benchmark)

        envs = {}
        mode = "train"
        envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
            config=self.config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )
        mode = "eval"
        envs[mode], env_id_to_task_map = env_builder.build_metaworld_vec_env(
            config=self.config,
            benchmark=benchmark,
            mode="train",
            env_id_to_task_map=env_id_to_task_map,
        )
        # In MT10 and MT50, the tasks are always sampled in the train mode.
        # For more details, refer https://github.com/rlworkgroup/metaworld

        max_episode_steps = 150
        # hardcoding the steps as different environments return different
        # values for max_path_length. MetaWorld uses 150 as the max length.
        metadata = self.get_env_metadata(
            env=envs["train"],
            max_episode_steps=max_episode_steps,
            ordered_task_list=list(env_id_to_task_map.keys()),
        )
        return envs, metadata

    def create_env_id_to_index_map(self) -> Dict[str, int]:
        env_id_to_index_map: Dict[str, int] = {}
        current_id = 0
        for env in self.envs.values():
            assert isinstance(env, VecEnv)
            for env_name in env.ids:
                if env_name not in env_id_to_index_map:
                    env_id_to_index_map[env_name] = current_id
                    current_id += 1
        return env_id_to_index_map

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
        episode_step = 0
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode", episode, step)

        episode_reward, mask, done, success = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False, 0.0]
        ]  # (num_envs, 1)
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        agent = self.agent
        offset = self.config.experiment.num_eval_episodes

        while episode_step < self.max_episode_steps:
            with agent_utils.eval_mode(agent):
                action = agent.select_action(
                    multitask_obs=multitask_obs, modes=["eval"]
                )
            multitask_obs, reward, done, info = vec_env.step(action)
            success += np.asarray([x["success"] for x in info])
            mask = mask * (1 - done.astype(int))
            episode_reward += reward * mask
            episode_step += 1
        start_index = 0
        success = (success > 0).astype("float")
        for mode in self.eval_modes_to_env_ids:
            num_envs = len(self.eval_modes_to_env_ids[mode])
            self.logger.log(
                f"{mode}/episode_reward",
                episode_reward[start_index : start_index + offset * num_envs].mean(),
                step,
            )
            self.logger.log(
                f"{mode}/success",
                success[start_index : start_index + offset * num_envs].mean(),
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
                    f"{mode}/success_env_index_{_current_env_index}",
                    success[
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

    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        env_indices = multitask_obs["task_obs"]
        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        for _ in range(num_steps):
            with agent_utils.eval_mode(self.agent):
                action = self.agent.sample_action(
                    multitask_obs=multitask_obs, mode="train"
                )  # (num_envs, action_dim)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()
            episode_reward += reward

            # allow infinite bootstrap
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                self.replay_buffer.add(
                    multitask_obs["env_obs"][index],
                    action[index],
                    reward[index],
                    next_multitask_obs["env_obs"][index],
                    done_bool,
                    env_index=env_index,
                )

            multitask_obs = next_multitask_obs
            episode_step += 1
