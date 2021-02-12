# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Implementation of the theta component described in
"Multi-Task Reinforcement Learning as a Hidden-Parameter Block MDP"
Link: https://arxiv.org/abs/2007.07206
"""
from enum import Enum
from typing import List

import torch
from torch import nn

from mtrl.agent.components import base as base_component
from mtrl.utils.types import TensorType


class ThetaSamplingStrategy(Enum):
    """Different strategies for sampling theta values.

    * `embedding` - use an embedding layer and index into it using
        task index.

    * `zero` - set theta as tensor of zeros.

    * `mean` - use an embedding layer and set theta as the mean of
        all the embeddings.

    * `mean_train` - use an embedding layer and set theta as the mean of
        all the embeddings that were trained.

    """

    EMBEDDING = "embedding"
    ZERO = "zero"
    MEAN = "mean"
    MEAN_TRAIN = "mean_train"


class ThetaModel(base_component.Component):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        num_envs: int,
        train_env_id: List[str],
    ):
        """Implementation of the theta component described in
            "Multi-Task Reinforcement Learning as a Hidden-Parameter Block MDP"
            Link: https://arxiv.org/abs/2007.07206
        Args:
            dim (int): input dimension.
            output_dim (int): output dimension.
            num_envs (int): number of environments.
            train_env_id (List[str]): index of environments corresponding
                to training tasks. Some strategies (for sampling theta)
                need this information.
        """
        super().__init__()
        self.num_envs = num_envs
        self.embedding = nn.Embedding(
            num_embeddings=self.num_envs,
            embedding_dim=dim,
            padding_idx=0,
        )
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, output_dim),
        )
        self.train_env_index = train_env_id

    def forward(
        self, env_index: TensorType, theta_sampling_strategy: str, modes: List[str]
    ) -> TensorType:
        """Sample theta.

            Following strategies are supported:

            * `embedding` - use an embedding layer and index into it using
                task index. This is the default strategy and used during
                training and testing on in-distribution environments.

            * `zero` - set theta as tensor of zeros.

            * `mean` - use an embedding layer and set theta as the mean of
                all the embeddings.

            * `mean_train` - use an embedding layer and set theta as the mean of
                all the embeddings that were trained.

        Args:
            env_index (TensorType):
            theta_sampling_strategy (str): strategy to sample theta.
            modes (List[str]): List of train/eval/... modes.

        Returns:
            TensorType: sampled theta.
        """
        if modes[0] == "train":
            # All the modes are same
            return self._compute_theta_for_one_mode(
                env_index=env_index,
                theta_sampling_strategy=theta_sampling_strategy[0],
                mode=modes[0],
            )
        elif modes[0] == "base":
            theta_groups = []
            start_index = 0
            for current_index in range(1, len(env_index)):
                if modes[current_index] != modes[start_index]:
                    # start a new group
                    theta_groups.append(
                        self._compute_theta_for_one_mode(
                            env_index=env_index[start_index:current_index],
                            theta_sampling_strategy=theta_sampling_strategy[
                                start_index
                            ],
                            mode=modes[start_index],
                        )
                    )
                    start_index = current_index
            theta_groups.append(
                self._compute_theta_for_one_mode(
                    env_index=env_index[start_index : current_index + 1],
                    theta_sampling_strategy=theta_sampling_strategy[start_index],
                    mode=modes[start_index],
                )
            )
            return torch.cat(theta_groups, dim=0)
        raise ValueError(f"`mode`={modes[0]} is not supported.")

    def _compute_theta_for_one_mode(
        self, env_index, theta_sampling_strategy: str, mode: str
    ):
        """Sample theta for one mode.

            Following strategies are supported:

            * `embedding` - use an embedding layer and index into it using
                task index. This is the default strategy and used during
                training and testing on in-distribution environments.

            * `zero` - set theta as tensor of zeros.

            * `mean` - use an embedding layer and set theta as the mean of
                all the embeddings.

            * `mean_train` - use an embedding layer and set theta as the mean of
                all the embeddings that were trained.

        Args:
            env_index (TensorType):
            theta_sampling_strategy (str): strategy to sample theta.
            mode (str): train/eval/... mode

        Returns:
            TensorType: sampled theta.
        """

        device = env_index.device
        if theta_sampling_strategy == ThetaSamplingStrategy.EMBEDDING:
            theta = self.model(self.embedding(env_index))
        elif theta_sampling_strategy == ThetaSamplingStrategy.ZERO:
            batch_dimension = len(env_index)
            env_index = [0]
            env_index = torch.LongTensor(env_index).to(device)  # type: ignore
            emb = self.embedding(env_index) * 0
            theta = self.model(emb).repeat(batch_dimension, 1)
        elif theta_sampling_strategy == ThetaSamplingStrategy.MEAN:
            batch_dimension = len(env_index)
            env_index = list(range(1, self.num_envs))
            env_index = torch.LongTensor(env_index).to(device)  # type: ignore
            theta = (
                self.model(self.embedding(env_index))
                .mean(dim=0, keepdim=True)
                .repeat(batch_dimension, 1)
            )
        elif theta_sampling_strategy == ThetaSamplingStrategy.MEAN_TRAIN:
            batch_dimension = len(env_index)
            env_index = list(self.train_env_index)
            env_index = torch.LongTensor(env_index).to(device)  # type: ignore
            theta = (
                self.model(self.embedding(env_index))
                .mean(dim=0, keepdim=True)
                .repeat(batch_dimension, 1)
            )
        if mode == "eval":
            theta = theta.detach()
        return theta
