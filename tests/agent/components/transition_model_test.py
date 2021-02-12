# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

import hydra
import pytest
import torch
from hydra.experimental import compose, initialize

setup = "hipbmdp"


def get_overrides_for_testing_decoder():
    overrides = [
        [
            "agent=deepmdp",
            "env= dmcontrol-finger-spin-distribution-v0",
            f"metrics={setup}",
        ],
    ]
    return overrides


@pytest.mark.parametrize("overrides", get_overrides_for_testing_decoder())
def test_reward_decoder(overrides: List[str]) -> None:
    with initialize(config_path="../../../config"):
        config = compose(
            config_name="config",
            overrides=overrides,
        )
        batch_size = 128
        action_shape = (4,)
        transition_model = hydra.utils.call(
            config.agent.transition_model,
            action_shape=action_shape,
            multitask_cfg=config.agent.multitask,
        )

        batch = torch.rand((batch_size, 50 + action_shape[0]))
        output = transition_model(batch)
        assert len(output) == 2
        assert output[0].shape == (batch_size, 50)
