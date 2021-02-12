# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Component to encode the task."""

import json

import torch
import torch.nn as nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.utils.types import ConfigType, TensorType


class TaskEncoder(base_component.Component):
    def __init__(
        self,
        pretrained_embedding_cfg: ConfigType,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        """Encode the task into a vector.

        Args:
            pretrained_embedding_cfg (ConfigType): config for using pretrained
                embeddings.
            num_embeddings (int): number of elements in the embedding table. This is
                used if pretrained embedding is not used.
            embedding_dim (int): dimension for the embedding. This is
                used if pretrained embedding is not used.
            hidden_dim (int): dimension of the hidden layer of the trunk.
            num_layers (int): number of layers in the trunk.
            output_dim (int): output dimension of the task encoder.
        """
        super().__init__()
        if pretrained_embedding_cfg.should_use:
            with open(pretrained_embedding_cfg.path_to_load_from) as f:
                metadata = json.load(f)
            ordered_task_list = pretrained_embedding_cfg.ordered_task_list
            pretrained_embedding = torch.Tensor(
                [metadata[task] for task in ordered_task_list]
            )
            assert num_embeddings == pretrained_embedding.shape[0]
            pretrained_embedding_dim = pretrained_embedding.shape[1]
            pretrained_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding,
                freeze=True,
            )
            projection_layer = nn.Sequential(
                nn.Linear(
                    in_features=pretrained_embedding_dim, out_features=2 * embedding_dim
                ),
                nn.ReLU(),
                nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim),
                nn.ReLU(),
            )
            projection_layer.apply(agent_utils.weight_init)
            self.embedding = nn.Sequential(  # type: ignore [call-overload]
                pretrained_embedding,
                nn.ReLU(),
                projection_layer,
            )

        else:
            self.embedding = nn.Sequential(
                nn.Embedding(
                    num_embeddings=num_embeddings, embedding_dim=embedding_dim
                ),
                nn.ReLU(),
            )
            self.embedding.apply(agent_utils.weight_init)
        self.trunk = agent_utils.build_mlp(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        self.trunk.apply(agent_utils.weight_init)

    def forward(self, env_index: TensorType) -> TensorType:
        return self.trunk(self.embedding(env_index))
