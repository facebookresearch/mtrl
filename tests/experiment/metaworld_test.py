# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

import pytest

from tests.experiment.utils import (
    check_output_from_cmd,
    get_configs_to_test,
    get_test_id,
)

OverridesType = Dict[str, str]


def get_config_for_metaworld_single_agent():

    params_to_test = {
        "agent": [
            "state_sac",
            "state_deepmdp",
            "pcgrad_state_sac",
            "pcgrad_state_deepmdp",
            "gradnorm_state_sac",
            "gradnorm_state_deepmdp",
        ],
        "agent.encoder.type_to_select": [
            "moe",
            "feedforward",
            # "identity"
            # "${agent.encoder.moe}",
            # "${agent.encoder.feedforward}",
            # "${agent.encoder.identity}",
        ],
        "agent.encoder.moe.task_id_to_encoder_id_cfg.mode": [
            "gate",
            "attention",
            "identity",
        ],
        "agent.encoder.moe.num_experts": [10],
        "env": ["metaworld-mt10"],
        "num_envs": [50],
        "should_use_disentangled_alpha": [True, False],
        "should_use_task_encoder": [True],
        "should_use_multi_head_policy": [True, False],
        "metrics": ["metaworld"],
        "should_use_soft_modularization": [False],
    }
    return get_configs_to_test(params_to_test=params_to_test)


@pytest.mark.parametrize(
    "config",
    get_config_for_metaworld_single_agent(),
)
def test_metaworld_single_agent(
    config,
) -> None:
    if config["should_use_soft_modularization"]:
        if (
            config["agent.encoder.type_to_select"] == "moe"
            or config["should_use_multi_head_policy"]
        ):
            return

    num_train_steps = 11
    frame_skip = None
    num_steps_per_episode = 150
    experiment_name = "metaworld"

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
