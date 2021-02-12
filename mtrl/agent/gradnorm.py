# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import OmegaConf

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.utils.types import ConfigType, OptimizerType, ParameterType, TensorType
from mtrl.utils.utils import flatten_list


class Agent(grad_manipulation_agent.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        multitask_cfg: ConfigType,
        agent_cfg: ConfigType,
        device: torch.device,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """GradNorm algorithm."""
        agent_cfg_copy = deepcopy(agent_cfg)
        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"

        self.gradnorm_optimizer_cfg = agent_cfg_copy.pop("gradnorm_optimizer_cfg")
        gradnorm_cfg = agent_cfg_copy.pop("gradnorm_cfg")

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
        self.gradnorm_alpha = gradnorm_cfg.alpha

        # change this
        self.task_weights: Dict[str, ParameterType] = {}
        self.task_weight_optimizer: Dict[str, OptimizerType] = {}
        self.initial_task_loss: Dict[str, TensorType] = {}

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _get_initial_task_weights_and_optimizer(self, num_tasks: int):
        task_weights = torch.nn.Parameter(
            torch.ones((num_tasks, 1), dtype=torch.float, device=self.device)
        )
        task_weight_optimizer = hydra.utils.instantiate(
            self.gradnorm_optimizer_cfg, params=[task_weights], lr=1e-6
        )

        return task_weights, task_weight_optimizer

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
        eps = 1e-8

        last_shared_layers = [
            self.get_last_shared_layers(component_name=comp_name)
            for comp_name in component_names
        ]
        last_shared_layers = [
            layer for layer in last_shared_layers if layer is not None
        ]
        if not last_shared_layers:
            loss.mean().backward()
        else:
            name = self._join_componenet_names(component_names=component_names)
            task_loss = self._convert_loss_into_task_loss(
                loss=loss, env_metadata=env_metadata
            )  # (num_tasks, 1)
            num_tasks = task_loss.shape[0]

            if name not in self.task_weights:
                # This is a new component
                (
                    self.task_weights[name],
                    self.task_weight_optimizer[name],
                ) = self._get_initial_task_weights_and_optimizer(num_tasks)
                self.initial_task_loss[name] = task_loss.clone().detach()
            task_weights = self.task_weights[name]  # (num_tasks, 1)
            weighted_task_loss = torch.mul(task_weights, task_loss)  # (num_tasks, 1)
            initial_task_loss = self.initial_task_loss[name]  # (num_tasks, 1)

            self.initial_task_loss[name] = task_loss.clone().detach()

            sum_weighted_task_loss = weighted_task_loss.sum()
            # we take the sum because the task_weights are normalized to sum to 1
            sum_weighted_task_loss.backward(retain_graph=True)

            task_weights.grad.zero_()
            # these grads will be computed here.
            last_shared_layer_parameters = flatten_list(
                [list(layer.parameters()) for layer in flatten_list(last_shared_layers)]  # type: ignore[arg-type]
            )

            grad_norm_list = []

            for index in range(num_tasks):
                task_grad = torch.autograd.grad(
                    task_loss[index],
                    last_shared_layer_parameters,
                    retain_graph=(index != num_tasks - 1),
                )
                vec_task_grad = torch.nn.utils.parameters_to_vector(task_grad)
                grad_norm_list.append(
                    torch.norm(torch.mul(task_weights[index], vec_task_grad))
                )
            grad_norms = torch.stack(grad_norm_list).unsqueeze(1)  # (num_tasks, 1)

            loss_ratio = (
                task_loss.clone().detach() / initial_task_loss
            )  # (num_tasks, 1)

            inverse_training_rate = loss_ratio / (
                loss_ratio.mean() + eps
            )  # (num_tasks, 1)

            mean_grad_norm = grad_norms.mean().detach()  # ()

            # detach

            target_grad_norm = mean_grad_norm * (
                inverse_training_rate ** self.gradnorm_alpha
            )  # (num_tasks, 1)

            grad_norm_loss = torch.nn.functional.l1_loss(
                grad_norms, target_grad_norm
            )  # ()
            # we take the sum because the task_weights are normalized to sum to 1
            task_weights.grad = torch.autograd.grad(grad_norm_loss, task_weights)[0]

            self.task_weight_optimizer[name].step()
            task_weights = task_weights / (task_weights.sum().detach() + eps)  # type: ignore[assignment]
            if any(weight[0].item() <= 0 for weight in task_weights):
                print(name, task_weights)
