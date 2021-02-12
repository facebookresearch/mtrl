# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import pytest
import torch
from hydra.experimental import compose, initialize

from mtrl.agent.components.reward_decoder import RewardDecoder

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
        reward_feature_dim = config.agent.reward_decoder.feature_dim
        reward_decoder = RewardDecoder(
            feature_dim=reward_feature_dim,
        )
        batch = torch.rand((batch_size, reward_feature_dim))
        assert reward_decoder(batch).shape == (batch_size, 1)
