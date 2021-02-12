# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.utils.types import ConfigType, TensorType


def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device


def apply_vector_grad_to_parameters(
    vec: TensorType, parameters: Iterable[TensorType], accumulate: bool = False
):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (
                param.grad + vec[pointer : pointer + num_param].view_as(param).data
            )
        else:
            param.grad = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


class Agent(grad_manipulation_agent.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        agent_cfg: ConfigType,
        multitask_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """PCGrad algorithm."""
        agent_cfg_copy = deepcopy(agent_cfg)
        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"
        OmegaConf.set_struct(agent_cfg_copy, True)

        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg_copy,
            device=device,
        )
        self.agent._compute_gradient = self._compute_gradient
        self._rng = np.random.default_rng()
        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:
        task_loss = self._convert_loss_into_task_loss(
            loss=loss, env_metadata=env_metadata
        )
        num_tasks = task_loss.shape[0]
        grad = []

        for index in range(num_tasks):
            grad.append(
                tuple(
                    _grad.contiguous()
                    for _grad in torch.autograd.grad(
                        task_loss[index],
                        parameters,
                        retain_graph=(retain_graph or index != num_tasks - 1),
                        allow_unused=allow_unused,
                    )
                )
            )

        grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )  # num_tasks x dim

        normalized_grad_vec = grad_vec / (
            grad_vec.norm(dim=1, keepdim=True) + 1e-8
        )  # num_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        shuffled_task_indices = self.get_shuffled_task_indices(num_tasks=num_tasks)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[
                task_indices
            ]  # num_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(
                dim=1, keepdim=True
            )  # num_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        pcgrad_vec = modified_grad_vec.mean(dim=0)
        apply_vector_grad_to_parameters(pcgrad_vec, parameters)

    def get_shuffled_task_indices(self, num_tasks: int) -> np.ndarray:
        shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
        for i in range(num_tasks):
            task_indices = np.arange(num_tasks)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            self._rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T
        return shuffled_task_indices
