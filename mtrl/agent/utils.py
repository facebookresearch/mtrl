# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

from mtrl.agent.components import moe_layer
from mtrl.utils.types import ModelType, TensorType


class eval_mode(object):
    def __init__(self, *models):
        """Put the agent in the eval mode"""
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net: ModelType, target_net: ModelType, tau: float) -> None:
    """Perform soft udpate on the net using target net.

    Args:
        net ([ModelType]): model to update.
        target_net (ModelType): model to update with.
        tau (float): control the extent of update.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed: int) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int): seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocess_obs(obs: TensorType, bits=5) -> TensorType:
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def weight_init_linear(m: ModelType):
    assert isinstance(m.weight, TensorType)
    nn.init.xavier_uniform_(m.weight)
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init_conv(m: ModelType):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert isinstance(m.weight, TensorType)
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)  # type: ignore[operator]
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain("relu")
    assert isinstance(m.weight, TensorType)
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init_moe_layer(m: ModelType):
    assert isinstance(m.weight, TensorType)
    for i in range(m.weight.shape[0]):
        nn.init.xavier_uniform_(m.weight[i])
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init(m: ModelType):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        weight_init_linear(m)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        weight_init_conv(m)
    elif isinstance(m, moe_layer.Linear):
        weight_init_moe_layer(m)


def _get_list_of_layers(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> List[nn.Module]:
    """Utility function to get a list of layers. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module]
    if num_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return mods


def build_mlp_as_module_list(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> ModelType:
    """Utility function to build a module list of layers. This assumes all
    the hidden layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    sequential_layers = []
    new_layer = []
    for index, current_layer in enumerate(mods):
        if index % 2 == 0:
            new_layer = [current_layer]
        else:
            new_layer.append(current_layer)
            sequential_layers.append(nn.Sequential(*new_layer))
    sequential_layers.append(nn.Sequential(*new_layer))
    return nn.ModuleList(sequential_layers)


def build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> ModelType:
    """Utility function to build a mlp model. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    return nn.Sequential(*mods)
