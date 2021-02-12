# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Transition dynamics for the agent."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.functional import Tensor

from mtrl.agent.components import base as base_component
from mtrl.utils.types import ConfigType, ModelType, TensorType


class TransitionModel(base_component.Component):
    def __init__(
        self,
        encoder_feature_dim: int,
        action_shape: List[int],
        layer_width: int,
        multitask_cfg: ConfigType,
    ):
        """Model for predicting the transition dynamics.

        Args:
            encoder_feature_dim (int): size of the input feature.
            action_shape (List[int]): size of the action vector.
            layer_width (int): width for each layer.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.

        """
        super().__init__()
        self.multitask_cfg = multitask_cfg

    def forward(self, x: TensorType) -> Tuple[TensorType, TensorType]:
        """Return the mean and standard deviation of the
            gaussian distribution that the model predicts for the
            next state.

        Args:
            x (TensorType): input.

        Returns:
            Tuple[TensorType, TensorType]: [mean of gaussian distribution, sigma of gaussian distribution]
        """
        raise NotImplementedError

    def sample_prediction(self, x: TensorType) -> TensorType:
        """Sample a possible value of next state from the model.

        Args:
            x (TensorType): input.

        Returns:
            TensorType: predicted next state.
        """
        raise NotImplementedError


class DeterministicTransitionModel(TransitionModel):
    def __init__(
        self,
        encoder_feature_dim: int,
        action_shape: List[int],
        layer_width: int,
        multitask_cfg: ConfigType,
    ):
        """Determinisitc model for predicting the transition dynamics.

        Args:
            encoder_feature_dim (int): size of the input feature.
            action_shape (List[int]): size of the action vector.
            layer_width (int): width for each layer.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
        """
        super().__init__(
            encoder_feature_dim=encoder_feature_dim,
            action_shape=action_shape,
            layer_width=layer_width,
            multitask_cfg=multitask_cfg,
        )
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)

    def forward(  # type: ignore[override]
        self, x: TensorType
    ) -> Tuple[TensorType, Optional[Tensor]]:
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x: TensorType) -> TensorType:
        mu, sigma = self(x)
        return mu

    def get_last_shared_layers(self) -> List[ModelType]:
        return [self.fc_mu]


class ProbabilisticTransitionModel(TransitionModel):
    def __init__(
        self,
        encoder_feature_dim: int,
        action_shape: List[int],
        layer_width: int,
        multitask_cfg: ConfigType,
        max_sigma: float = 1e1,
        min_sigma: float = 1e-4,
    ):
        """Probabilistic model for predicting the transition dynamics.

        Args:
            encoder_feature_dim (int): size of the input feature.
            action_shape (List[int]): size of the action vector.
            layer_width (int): width for each layer.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            max_sigma (float, optional): maximum value of sigma (of the learned
                gaussian distribution). Larger values are clipped to this value.
                Defaults to 1e1.
            min_sigma (float, optional): minimum value of sigma (of the learned
                gaussian distribution). Smaller values are clipped to this value.
                Defaults to 1e-4.
        """
        super().__init__(
            encoder_feature_dim=encoder_feature_dim,
            action_shape=action_shape,
            layer_width=layer_width,
            multitask_cfg=multitask_cfg,
        )
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = (
            self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        )  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def get_last_shared_layers(self) -> List[ModelType]:
        breakpoint()
        return [self.fc]


_AVAILABLE_TRANSITION_MODELS = {
    "": DeterministicTransitionModel,
    "deterministic": DeterministicTransitionModel,
    "probabilistic": ProbabilisticTransitionModel,
    # "ensemble": EnsembleOfProbabilisticTransitionModels,
}


def make_transition_model(
    action_shape: List[int], transition_cfg: ConfigType, multitask_cfg: ConfigType
):
    assert transition_cfg.type in _AVAILABLE_TRANSITION_MODELS

    encoder_feature_dim = transition_cfg.feature_dim
    if multitask_cfg.should_use_task_encoder:
        encoder_feature_dim += multitask_cfg.task_encoder_cfg.model_cfg.output_dim

    return _AVAILABLE_TRANSITION_MODELS[transition_cfg.type](
        action_shape=action_shape,
        encoder_feature_dim=encoder_feature_dim,
        layer_width=transition_cfg.layer_width,
        multitask_cfg=multitask_cfg,
    )
