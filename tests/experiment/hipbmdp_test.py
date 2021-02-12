# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

import pytest

from tests.experiment.utils import (
    check_output_from_cmd,
    get_configs_to_test,
    get_test_id,
)

OverridesType = Dict[str, str]


def get_config_for_dmcontrol_multi_task():
    params_to_test = {
        "agent": [
            "sac",
            "sac_ae",
            "deepmdp",
            "pcgrad_sac",
            "pcgrad_sac_ae",
            "pcgrad_deepmdp",
            "gradnorm_deepmdp",
            "gradnorm_sac",
            "hipbmdp",
        ],
        "setup": ["hipbmdp"],
        "env": ["dmcontrol-finger-spin-distribution-v0"],
        "num_envs": [10],
        "should_use_disentangled_alpha": [True, False],
        "should_use_task_encoder": [False],
        "should_use_multi_head_policy": [True, False],
        # "env.builder.make_kwargs.sticky_observation_cfg.should_use": [True, False],
        "should_use_soft_modularization": [False],
    }
    return get_configs_to_test(params_to_test=params_to_test)


@pytest.mark.parametrize(
    "config",
    get_config_for_dmcontrol_multi_task(),
)
def test_dmcontrol_multi_task(config) -> None:

    num_train_steps = 11
    frame_skip = 100
    num_steps_per_episode = 1000
    experiment_name = "dmcontrol"
    _id = get_test_id(experiment_name=experiment_name, config=config)
    check_output_from_cmd(
        experiment_name=experiment_name,
        agent=config.pop("agent"),
        env=config.pop("env"),
        _id=_id,
        num_train_steps=num_train_steps,
        frame_skip=frame_skip,
        num_steps_per_episode=num_steps_per_episode,
        num_envs=config.pop("num_envs"),
        should_use_disentangled_alpha=config.pop("should_use_disentangled_alpha"),
        should_use_task_encoder=config.pop("should_use_task_encoder"),
        should_use_multi_head_policy=config.pop("should_use_multi_head_policy"),
        should_use_soft_modularization=config.pop("should_use_soft_modularization"),
        overrides=config,
    )
