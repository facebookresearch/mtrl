# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict

import torch
from gym.vector.async_vector_env import AsyncVectorEnv


class VecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            observation_space=observation_space,
            action_space=action_space,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def reset(self):
        multitask_obs = super().reset()
        return _cast_multitask_obs(multitask_obs=multitask_obs)

    def step(self, actions):
        multitask_obs, reward, done, info = super().step(actions)
        return _cast_multitask_obs(multitask_obs=multitask_obs), reward, done, info


def _cast_multitask_obs(multitask_obs):
    return {key: torch.tensor(value) for key, value in multitask_obs.items()}


class MetaWorldVecEnv(AsyncVectorEnv):
    def __init__(
        self,
        env_metadata: Dict[str, Any],
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        """Return only every `skip`-th frame"""
        super().__init__(
            env_fns=env_fns,
            observation_space=observation_space,
            action_space=action_space,
            shared_memory=shared_memory,
            copy=copy,
            context=context,
            daemon=daemon,
            worker=worker,
        )
        self.num_envs = len(env_fns)
        self.task_obs = torch.arange(self.num_envs)
        assert "mode" in env_metadata
        assert "ids" in env_metadata
        self._metadata = env_metadata

    @property
    def mode(self):
        return self._metadata["mode"]

    @property
    def ids(self):
        return self._metadata["ids"]

    def _check_observation_spaces(self):
        return

    def reset(self):
        env_obs = super().reset()
        return self.create_multitask_obs(env_obs=env_obs)

    def step(self, actions):
        env_obs, reward, done, info = super().step(actions)
        return self.create_multitask_obs(env_obs=env_obs), reward, done, info

    def create_multitask_obs(self, env_obs):
        return {"env_obs": torch.tensor(env_obs), "task_obs": self.task_obs}
