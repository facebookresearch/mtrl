# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import pytest
import torch
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf as OC

from mtrl.agent.components.decoder import make_decoder

setup = "hipbmdp"


def get_overrides_for_testing_decoder():
    overrides = [
        [
            "env=dmcontrol-finger-spin-distribution-v0",
            "agent=sac_ae",
            f"metrics={setup}",
        ],
    ]
    return overrides


@pytest.mark.parametrize("overrides", get_overrides_for_testing_decoder())
def test_decoder(overrides: List[str]) -> None:
    with initialize(config_path="../../../config"):
        config = compose(
            config_name="config",
            overrides=overrides,
        )
        env_obs_shape = [3, 84, 84]
        batch_size = 128
        decoder = make_decoder(
            env_obs_shape=env_obs_shape,
            decoder_cfg=config.agent.decoder,
            multitask_cfg=OC.create({}),
        )
        batch = torch.rand((batch_size, config.agent.decoder.feature_dim))
        assert decoder(batch).shape == (batch_size, *env_obs_shape)
