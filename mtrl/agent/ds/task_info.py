# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Datastructure to encapsulate the task-related information."""
from dataclasses import dataclass
from typing import Optional

import torch

from mtrl.utils.types import TensorType


@dataclass
class TaskInfo:
    __slots__ = ["encoding", "compute_grad", "env_index"]
    encoding: Optional[TensorType]
    compute_grad: bool
    env_index: TensorType


NoneTaskInfo = TaskInfo(encoding=None, compute_grad=False, env_index=torch.zeros(1))
# This is a special variable. It is used when we want `TaskInfo` to be None.
