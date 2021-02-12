# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""`Experiment` class manages the lifecycle of a multi-task model."""

import time
from typing import Dict, List, Tuple

import hydra
import numpy as np

from mtrl.agent import utils as agent_utils
from mtrl.env.types import EnvType
from mtrl.env.vec_env import VecEnv  # type: ignore
from mtrl.experiment import experiment
from mtrl.utils.types import ConfigType, EnvMetaDataType, EnvsDictType, ListConfigType


class Experiment(experiment.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a multi-task model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(config, experiment_id)
        self.eval_modes_to_env_ids = self.create_eval_modes_to_env_ids()
        self.should_reset_env_manually = False
        self.metrics_to_track = {
            x[0] for x in self.config.metrics["train"] if not x[0].endswith("_")
        }

    def build_envs(self) -> Tuple[EnvsDictType, EnvMetaDataType]:
        """Build environments and return env-related metadata"""
        if "dmcontrol" not in self.config.env.name:
            raise NotImplementedError
        envs: EnvsDictType = {}
        mode = "train"
        env_id_list = self.config.env[mode]
        num_envs = len(env_id_list)
        seed_list = list(range(1, num_envs + 1))
        mode_list = [mode for _ in range(num_envs)]

        envs[mode] = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )
        envs["eval"] = self._create_dmcontrol_vec_envs_for_eval()
        metadata = self.get_env_metadata(env=envs["train"])
        return envs, metadata

    def create_eval_modes_to_env_ids(self) -> Dict[str, List[int]]:
        """Map each eval mode to a list of environment index.

            The eval modes are of the form `eval_xyz` where `xyz` specifies
            the specific type of evaluation. For example. `eval_interpolation`
            means that we are using interpolation environments for evaluation.
            The eval moe can also be set to just `eval`.

        Returns:
            Dict[str, List[int]]: dictionary with different eval modes as
                keys and list of environment index as values.
        """
        eval_modes_to_env_ids: Dict[str, List[int]] = {}
        eval_modes = [
            key for key in self.config.metrics.keys() if not key.startswith("train")
        ]
        for mode in eval_modes:
            # todo: add support for mode == "eval"
            if "_" in mode:
                _mode, _submode = mode.split("_")
                env_ids = self.config.env[_mode][_submode]
                eval_modes_to_env_ids[mode] = env_ids
            elif mode != "eval":
                raise ValueError(f"eval mode = `{mode}`` is not supported.")
        return eval_modes_to_env_ids

    def _create_dmcontrol_vec_envs_for_eval(self) -> EnvType:
        """Method to create the vec env with multiple copies of the same
        environment. It is useful when evaluating the agent multiple times
        in the same env.

        The vec env is organized as follows - number of modes x number of tasks per mode x number of episodes per task

        """

        env_id_list: List[str] = []
        seed_list: List[int] = []
        mode_list: List[str] = []
        num_episodes_per_env = self.config.experiment.num_eval_episodes
        for mode in self.config.metrics.keys():
            if mode == "train":
                continue

            if "_" in mode:
                _mode, _submode = mode.split("_")
                if _mode != "eval":
                    raise ValueError("`mode` does not start with `eval_`")
                if not isinstance(self.config.env.eval, ConfigType):
                    raise ValueError(
                        f"""`self.config.env.eval` should either be a DictConfig.
                        Detected type is {type(self.config.env.eval)}"""
                    )
                if _submode in self.config.env.eval:
                    for _id in self.config.env[_mode][_submode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [_submode for _ in range(num_episodes_per_env)]
            elif mode == "eval":
                if isinstance(self.config.env.eval, ListConfigType):
                    for _id in self.config.env[mode]:
                        env_id_list += [_id for _ in range(num_episodes_per_env)]
                        seed_list += list(range(1, num_episodes_per_env + 1))
                        mode_list += [mode for _ in range(num_episodes_per_env)]
            else:
                raise ValueError(f"eval mode = `{mode}` is not supported.")
        env = hydra.utils.instantiate(
            self.config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )

        return env

    def run(self):
        """Run the experiment."""
        exp_config = self.config.experiment

        vec_env = self.envs["train"]

        episode_reward, episode_step, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 0, True]
        ]  # (num_envs, 1)

        if "success" in self.metrics_to_track:
            success = np.full(shape=vec_env.num_envs, fill_value=0.0)

        info = {}

        assert self.start_step >= 0
        episode = self.start_step // self.max_episode_steps

        start_time = time.time()

        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        env_indices = multitask_obs["task_obs"]

        train_mode = ["train" for _ in range(vec_env.num_envs)]

        for step in range(self.start_step, exp_config.num_train_steps):

            if step % self.max_episode_steps == 0:  # todo
                if step > 0:
                    if "success" in self.metrics_to_track:
                        success = (success > 0).astype("float")
                        for index, _ in enumerate(env_indices):
                            self.logger.log(
                                f"train/success_env_index_{index}",
                                success[index],
                                step,
                            )
                        self.logger.log("train/success", success.mean(), step)
                    for index, env_index in enumerate(env_indices):
                        self.logger.log(
                            f"train/episode_reward_env_index_{index}",
                            episode_reward[index],
                            step,
                        )
                        self.logger.log(f"train/env_index_{index}", env_index, step)

                    self.logger.log("train/duration", time.time() - start_time, step)
                    start_time = time.time()
                    self.logger.dump(step)

                # evaluate agent periodically
                if step % exp_config.eval_freq == 0:
                    self.evaluate_vec_env_of_tasks(
                        vec_env=self.envs["eval"], step=step, episode=episode
                    )
                    if exp_config.save.model:
                        self.agent.save(
                            self.model_dir,
                            step=step,
                            retain_last_n=exp_config.save.model.retain_last_n,
                        )
                    if exp_config.save.buffer.should_save:
                        self.replay_buffer.save(
                            self.buffer_dir,
                            size_per_chunk=exp_config.save.buffer.size_per_chunk,
                            num_samples_to_save=exp_config.save.buffer.num_samples_to_save,
                        )
                episode += 1
                episode_reward = np.full(shape=vec_env.num_envs, fill_value=0.0)
                if "success" in self.metrics_to_track:
                    success = np.full(shape=vec_env.num_envs, fill_value=0.0)

                self.logger.log("train/episode", episode, step)

            if step < exp_config.init_steps:
                action = np.asarray(
                    [self.action_space.sample() for _ in range(vec_env.num_envs)]
                )  # (num_envs, action_dim)

            else:
                with agent_utils.eval_mode(self.agent):
                    # multitask_obs = {"env_obs": obs, "task_obs": env_indices}
                    action = self.agent.sample_action(
                        multitask_obs=multitask_obs,
                        modes=[
                            train_mode,
                        ],
                    )  # (num_envs, action_dim)

            # run training update
            if step >= exp_config.init_steps:
                num_updates = (
                    exp_config.init_steps if step == exp_config.init_steps else 1
                )
                for _ in range(num_updates):
                    self.agent.update(self.replay_buffer, self.logger, step)
            next_multitask_obs, reward, done, info = vec_env.step(action)
            if self.should_reset_env_manually:
                if (episode_step[0] + 1) % self.max_episode_steps == 0:
                    # we do a +2 because we started the counting from 0 and episode_step is incremented after updating the buffer
                    next_multitask_obs = vec_env.reset()

            episode_reward += reward
            if "success" in self.metrics_to_track:
                success += np.asarray([x["success"] for x in info])

            # allow infinite bootstrap
            for index, env_index in enumerate(env_indices):
                done_bool = (
                    0
                    if episode_step[index] + 1 == self.max_episode_steps
                    else float(done[index])
                )
                if index not in self.envs_to_exclude_during_training:
                    self.replay_buffer.add(
                        multitask_obs["env_obs"][index],
                        action[index],
                        reward[index],
                        next_multitask_obs["env_obs"][index],
                        done_bool,
                        task_obs=env_index,
                    )

            multitask_obs = next_multitask_obs
            episode_step += 1
        self.replay_buffer.delete_from_filesystem(self.buffer_dir)
        self.close_envs()

    def collect_trajectory(self, vec_env: VecEnv, num_steps: int) -> None:
        """Collect some trajectories, by unrolling the policy (in train mode),
        and update the replay buffer.
        Args:
            vec_env (VecEnv): environment to collect data from.
            num_steps (int): number of steps to collect data for.

        """
        raise NotImplementedError
