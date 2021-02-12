# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Implementation of the soft routing network and MLP described in
"Multi-Task Reinforcement Learning with Soft Modularization"
Link: https://arxiv.org/abs/2003.13661
"""


from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from mtrl.agent.components.base import Component as BaseComponent
from mtrl.agent.components.moe_layer import Linear
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.utils.types import TensorType


class SoftModularizedMLP(BaseComponent):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """Class to implement the actor/critic in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        It is similar to layers.FeedForward but allows selection of expert
        at each layer.
        """
        super().__init__()
        layers: List[nn.Module] = []
        current_in_features = in_features
        for _ in range(num_layers - 1):
            linear = Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layers.append(nn.Sequential(linear, nn.ReLU()))
            # Each layer is a combination of a moe layer and ReLU.
            current_in_features = hidden_features
        linear = Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )
        layers.append(linear)
        self.layers = nn.ModuleList(layers)
        self.routing_network = RoutingNetwork(
            in_features=in_features,
            hidden_features=hidden_features,
            num_layers=num_layers - 1,
            num_experts_per_layer=num_experts,
        )

    def forward(self, mtobs: MTObs) -> TensorType:
        probs = self.routing_network(mtobs=mtobs)
        # (num_layers, batch, num_experts, num_experts)
        probs = probs.permute(0, 2, 3, 1)
        # (num_layers, num_experts, num_experts, batch)
        num_experts = probs.shape[1]
        inp = mtobs.env_obs
        # (batch, dim1)
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index]
            # (num_experts, num_experts, batch)
            inp = layer(inp)
            # (num_experts, batch, dim2)
            _out = p.unsqueeze(-1) * inp.unsqueeze(0).repeat(num_experts, 1, 1, 1)
            # (num_experts, num_experts, batch, dim2)
            inp = _out.sum(dim=1)
            # (num_experts, batch, dim2)
        out = self.layers[-1](inp).mean(dim=0)
        # (batch, out_dim)
        return out


class RoutingNetwork(BaseComponent):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_experts_per_layer: int,
        num_layers: int,
    ) -> None:
        """Class to implement the routing network in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        """
        super().__init__()

        self.num_experts_per_layer = num_experts_per_layer

        self.projection_before_routing = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
        )

        self.W_d = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_features,
                    out_features=self.num_experts_per_layer ** 2,
                )
                for _ in range(num_layers)
            ]
        )

        self.W_u = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.num_experts_per_layer ** 2,
                    out_features=hidden_features,
                )
                for _ in range(num_layers - 1)
            ]
        )  # the first layer does not need W_u

        self._softmax = nn.Softmax(dim=2)

    def _process_logprob(self, logprob: TensorType) -> TensorType:
        logprob_shape = logprob.shape
        logprob = logprob.reshape(
            logprob_shape[0], self.num_experts_per_layer, self.num_experts_per_layer
        )
        # logprob[:][i][j] == weight (probability )of the ith module (in current layer)
        # for contributing to the jth module in the next layer.
        # Since the ith module has to divide its weight among all modules in the
        # next layer, logprob[batch_index][i][:] to 1
        prob = self._softmax(logprob)
        return prob

    def forward(self, mtobs: MTObs) -> TensorType:
        obs = mtobs.env_obs
        task_info = mtobs.task_info
        assert task_info is not None
        assert task_info.encoding is not None
        assert obs.shape == task_info.encoding.shape
        inp = self.projection_before_routing(obs * task_info.encoding)
        p = self.W_d[0](F.relu(inp))  # batch x num_experts ** 2
        prob = [p]
        for W_u, W_d in zip(self.W_u, self.W_d[1:]):
            p = W_d(F.relu((W_u(prob[-1]) * inp)))
            prob.append(p)
        prob_tensor = torch.cat(
            [self._process_logprob(logprob=logprob).unsqueeze(0) for logprob in prob],
            dim=0,
        )
        return prob_tensor
