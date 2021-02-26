# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Reward decoder component for the agent."""

from typing import List

import torch.nn as nn

from mtrl.agent.components import base as base_component
from mtrl.utils.types import ModelType, TensorType


class RewardDecoder(base_component.Component):
    def __init__(
        self,
        feature_dim: int,
    ):
        """Predict reward using the observations.

        Args:
            feature_dim (int): dimension of the feature used to predict
                the reward.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: TensorType) -> TensorType:
        return self.trunk(x)

    def get_last_shared_layers(self) -> List[ModelType]:
        return [self.trunk[-1]]
