# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hydra
from hydra.experimental import compose, initialize

from mtrl.env import builder as env_builder
from mtrl.experiment import experiment


def get_metaworld_env_and_metadata(env_name: str, mode: str):
    with initialize(config_path="../../config"):
        config = compose(
            config_name="config",
            overrides=[
                "agent=state_sac",
                f"env={env_name}",
                "experiment.num_eval_episodes=1",
                "metrics=metaworld",
            ],
        )
        benchmark = hydra.utils.instantiate(config.env.benchmark)
        env, env_id_to_task_map = env_builder.build_metaworld_vec_env(
            config=config, benchmark=benchmark, mode=mode, env_id_to_task_map=None
        )
        env_metadata = experiment.get_env_metadata(
            env=env,
            max_episode_steps=150,
            ordered_task_list=list(env_id_to_task_map.keys()),
        )
    return env, env_metadata


def get_hip_env_and_metadata(env_name: str, mode: str):
    with initialize(config_path="../../config"):
        config = compose(
            config_name="config",
            overrides=[
                "agent=sac",
                f"env={env_name}",
                "experiment.num_eval_episodes=1",
                "metrics=hipbmdp",
            ],
        )

        env = hydra.utils.instantiate(
            config.env.builder,
            mode_list=[mode for _ in config.env["train"]],
            env_id_list=config.env["train"],
            seed_list=[config.setup.seed for _ in config.env["train"]],
        )

        metadata = experiment.get_env_metadata(env=env)
        return env, metadata
