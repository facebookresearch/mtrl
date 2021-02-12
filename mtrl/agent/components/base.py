# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Interface for the agent components."""
from typing import List

from torch import nn

from mtrl.utils.types import ModelType


class Component(nn.Module):
    """Basic component (for building the agent) that every other component should extend.

    It inherits `torch.nn.Module`.

    """

    def __init__(self):
        super().__init__()

    def get_last_shared_layers(self) -> List[ModelType]:
        """Get the list of last layers (for different sub-components) that are shared
        across tasks.

        This method should be implemented by the subclasses if the component is to be
        trained with gradnorm algorithm.

        Returns:
            List[ModelType]: list of layers.
        """
        raise NotImplementedError(
            """Implement the `get_last_shared_layers` method
                if you want to train the agent with gradnorm algorithm."""
        )
