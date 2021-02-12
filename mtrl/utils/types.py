# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Collection of types used in the code."""

from typing import TYPE_CHECKING, Dict, Tuple

import gym
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]

ListConfigType = ListConfig
ConfigType = DictConfig
TensorType = torch.Tensor
ReplayBufferSampleType = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]

if TYPE_CHECKING:
    OptimizerType = torch.optim.optimizer.Optimizer
else:
    OptimizerType = torch.optim.Optimizer
ModelType = torch.nn.Module
ParameterType = torch.nn.Parameter

ComponentType = ModelType
EnvsDictType = Dict[str, VecEnv]
EnvMetaDataType = Dict[str, gym.spaces.box.Box]
