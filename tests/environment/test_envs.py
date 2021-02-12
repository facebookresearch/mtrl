# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
import numpy as np
import pytest
from hydra.experimental import compose, initialize

from mtrl.env import builder as env_builder


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "env, mode",
    [
        ("dmcontrol-finger-spin-distribution-v0", "train"),
        ("dmcontrol-finger-spin-distribution-v0", "eval.interpolation"),
    ],
)
def test_dmcontrol_vec_env(env, mode) -> None:
    with initialize(config_path="../../config"):
        # config is relative to a modules
        config = compose(
            config_name="config",
            overrides=[
                f"env={env}",
                "env.builder._target_=mtrl.env.builder.build_dmcontrol_vec_env",
            ],
        )
        if "." in mode:
            submodes = mode.split(".")
            _current_config = config.env
            for _submode in submodes:
                _current_config = _current_config[_submode]
            env_id_list = list(_current_config)
        else:
            env_id_list = list(config.env[mode])
        num_envs = len(env_id_list)
        seed_list = list(range(1, num_envs + 1))
        mode_list = [mode for _ in range(num_envs)]
        env = hydra.utils.call(
            config.env.builder,
            env_id_list=env_id_list,
            seed_list=seed_list,
            mode_list=mode_list,
        )
        env.reset()

        action = np.asarray(env.action_space.sample())
        obs, reward, done, info = env.step(action)
        assert action.shape == (num_envs, 2)
        assert isinstance(obs, dict)
        for key in ["env_obs", "task_obs"]:
            assert key in obs
        assert obs["env_obs"].shape == (num_envs, 9, 84, 84)
        assert obs["task_obs"].shape == (num_envs,)
        env.close()


def get_test_params_for_metaworld_env_MT():
    test_params = []
    for num_tasks in [1, 10, 50]:
        env = f"metaworld-mt{num_tasks}"
        for mode in ["train"]:
            test_params.append((env, mode))
    return test_params


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("env, mode", get_test_params_for_metaworld_env_MT())
def test_metaworld_env_MT(env, mode) -> None:
    with initialize(config_path="../../config"):
        # config is relative to a modules
        config = compose(
            config_name="config",
            overrides=[f"env={env}", "experiment.num_eval_episodes=2"],
        )
        benchmark = hydra.utils.instantiate(config.env.benchmark)
        env, env_id_to_task_map = env_builder.build_metaworld_vec_env(
            config=config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )
        _, new_env_id_to_task_map = env_builder.build_metaworld_vec_env(
            config=config,
            benchmark=benchmark,
            mode=mode,
            env_id_to_task_map=env_id_to_task_map,
        )
        assert new_env_id_to_task_map is env_id_to_task_map
        env.reset()
        num_envs = len(env.ids)
        action = np.concatenate(
            [np.expand_dims(x, 0) for x in env.action_space.sample()]
        )
        mtobs, reward, done, info = env.step(action)
        assert mtobs["env_obs"].shape == (num_envs, 12)
        assert action.shape == (num_envs, 4)
        assert "success" in info[0]
        env.close()
