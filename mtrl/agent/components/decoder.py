# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Decoder component for the agent."""

from typing import List

import torch
import torch.nn as nn

from mtrl.agent.components import base as base_component
from mtrl.utils.types import ConfigType, ModelType, TensorType


class PixelDecoder(base_component.Component):
    def __init__(
        self,
        env_obs_shape: List[int],
        multitask_cfg: ConfigType,
        feature_dim: int,
        num_layers: int = 2,
        num_filters: int = 32,
    ):
        """Convolutional decoder for pixels observations.

        Args:
            env_obs_shape (List[int]): shape of the observation that the actor gets.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            feature_dim (int): feature dimension.
            num_layers (int, optional): number of layers. Defaults to 2.
            num_filters (int, optional): number of conv filters per layer. Defaults to 32.
        """

        super().__init__()
        self.multitask_cfg = multitask_cfg
        self.num_layers = num_layers
        self.num_filters = num_filters
        layer_to_dim_mapping = {2: 39, 4: 35, 6: 31}
        self.out_dim = layer_to_dim_mapping[num_layers]

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, env_obs_shape[0], 3, stride=2, output_padding=1
            )
        )

    def forward(self, h: TensorType) -> TensorType:
        h = torch.relu(self.fc(h))

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        obs = self.deconvs[-1](deconv)

        return obs

    def get_last_shared_layers(self) -> List[ModelType]:
        return [self.deconvs[-1]]


_AVAILABLE_DECODERS = {"pixel": PixelDecoder}


def make_decoder(
    env_obs_shape: List[int], decoder_cfg: ConfigType, multitask_cfg: ConfigType
):
    assert decoder_cfg.type in _AVAILABLE_DECODERS
    feature_dim = decoder_cfg.feature_dim
    if multitask_cfg.should_use_task_encoder:
        feature_dim += multitask_cfg.task_encoder_cfg.model_cfg.output_dim
    return _AVAILABLE_DECODERS[decoder_cfg.type](
        env_obs_shape=env_obs_shape,
        multitask_cfg=multitask_cfg,
        feature_dim=feature_dim,
        num_layers=decoder_cfg.num_layers,
        num_filters=decoder_cfg.num_filters,
    )
