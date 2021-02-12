# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Interface for the objects that can be checkpointed on the filesystem."""
from abc import ABC, abstractmethod
from typing import Any


class Checkpointable(ABC):
    """All classes that want to support checkpointing should extend this class."""

    @abstractmethod
    def save(self, *args, **kwargs) -> Any:
        """Save the object to a checkpoint.

        Returns:
            Any
        """
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """Load the object from a checkpoint.

        Returns:
            Any
        """
        pass
