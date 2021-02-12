# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Interface for the agent."""
import abc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy as np
import torch

from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer
from mtrl.utils.types import ComponentType, ConfigType, ModelType, OptimizerType
from mtrl.utils.utils import is_integer, make_dir

ComponentOrOptimizerType = Union[ComponentType, OptimizerType]


class Agent(abc.ABC):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        multitask_cfg: ConfigType,
        device: torch.device,
    ):
        """Abstract agent class that every other agent should extend.

        Args:

            env_obs_shape (List[int]): shape of the environment observation that the actor gets.
            action_shape (List[int]): shape of the action vector that the actor produces.
            action_range (Tuple[int, int]): min and max values for the action vector.
            multitask_cfg (ConfigType): config for encoding the multitask knowledge.
            device (torch.device): device for the agent.
        """
        self.env_obs_shape = env_obs_shape
        self.action_shape = action_shape
        self.action_range = action_range
        self.multitask_cfg = multitask_cfg
        self.num_envs = multitask_cfg.num_envs
        self.device = device
        self._opimizer_suffix = "_optimizer"
        self._components: Dict[str, ModelType] = {}
        self._optimizers: Dict[str, OptimizerType] = {}

    @abc.abstractmethod
    def complete_init(self, cfg_to_load_model: ConfigType) -> None:
        """Complete the init process.

            The derived classes should implement this to perform different post-processing steps.

        Args:
            cfg_to_load_model (ConfigType): config to load the model.
        """
        pass

    @abc.abstractmethod
    def train(self, training: bool = True) -> None:
        """Set the agent in training/evaluation mode

        Args:
            training (bool, optional): should set in training mode. Defaults to True.
        """
        pass

    @abc.abstractmethod
    # def select_action(self, obs, env_index: TensorType, mode: List[str]):
    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        """Select the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            modes (List[str]): modes for selecting the action.

        Returns:
            np.ndarray: selected action.
        """
        pass

    @abc.abstractmethod
    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        """Sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            modes (List[str]): modes for sampling the action.

        Returns:
            np.ndarray: sampled action.
        """
        pass

    @abc.abstractmethod
    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update the agent.

        Args:
            replay_buffer (ReplayBuffer): replay buffer to sample the data.
            logger (Logger): logger for logging.
            step (int): step for tracking the training progress.
            kwargs_to_compute_gradient (Optional[Dict[str, Any]], optional): Defaults
                to None.
            buffer_index_to_sample (Optional[np.ndarray], optional): if this parameter
                is specified, use these indices instead of sampling from the replay
                buffer. If this is set to `None`, sample from the replay buffer.
                buffer_index_to_sample Defaults to None.

        Returns:
            np.ndarray: index sampled (from the replay buffer) to train the model. If
                buffer_index_to_sample is not set to None, return buffer_index_to_sample.

        """

        pass

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:
        """Get the last shared layer for any given component.

        Args:
            component_name (str): given component.

        Returns:
            List[ModelType]: list of layers.
        """
        raise NotImplementedError(
            """Implement the `get_last_shared_layers` method
                if you want to train the agent with grad_norm algorithm."""
        )

    def get_component_name_list_for_checkpointing(self) -> List[Tuple[ModelType, str]]:
        """Get the list of tuples of (model, name) from the agent to checkpoint.

        Returns:
            List[Tuple[ModelType, str]]: list of tuples of (model, name).
        """
        return [(value, key) for key, value in self._components.items()]

    def get_optimizer_name_list_for_checkpointing(
        self,
    ) -> List[Tuple[OptimizerType, str]]:
        """Get the list of tuples of (optimizer, name) from the agent to checkpoint.

        Returns:
            List[Tuple[OptimizerType, str]]: list of tuples of (optimizer, name).
        """
        return [(value, key) for key, value in self._optimizers.items()]

    def save(
        self,
        model_dir: str,
        step: int,
        retain_last_n: int,
        should_save_metadata: bool = True,
    ) -> None:
        """Save the agent.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.
            should_save_metadata (bool, optional): should training metadata be
                saved. Defaults to True.
        """
        if retain_last_n == 0:
            print("Not saving the models as retain_last_n = 0")
            return
        make_dir(model_dir)
        # write a test case for save/load

        self.save_components(model_dir, step, retain_last_n)

        self.save_optimizers(model_dir, step, retain_last_n)

        if should_save_metadata:
            self.save_metadata(model_dir, step)

    def save_components(self, model_dir: str, step: int, retain_last_n: int) -> None:
        """Save the different components of the agent.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.

        """
        return self.save_components_or_optimizers(
            component_or_optimizer_list=self.get_component_name_list_for_checkpointing(),
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            suffix="",
        )

    def save_optimizers(self, model_dir: str, step: int, retain_last_n: int) -> None:
        """Save the different optimizers of the agent.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.

        """

        return self.save_components_or_optimizers(
            component_or_optimizer_list=self.get_optimizer_name_list_for_checkpointing(),
            model_dir=model_dir,
            step=step,
            retain_last_n=retain_last_n,
            suffix=self._opimizer_suffix,
        )

    def save_components_or_optimizers(
        self,
        component_or_optimizer_list: Union[
            List[Tuple[ComponentType, str]], List[Tuple[OptimizerType, str]]
        ],
        model_dir: str,
        step: int,
        retain_last_n: int,
        suffix: str = "",
    ) -> None:
        """Save the components and optimizers from the given list.

        Args:
            component_or_optimizer_list
                (Union[ List[Tuple[ComponentType, str]], List[Tuple[OptimizerType, str]] ]):
                list of components and optimizers to save.
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.
            retain_last_n (int): number of models to retain.
            suffix (str, optional): suffix to add at the name of the model before
                checkpointing. Defaults to "".
        """
        model_dir_path = Path(model_dir)

        for component_or_optimizer, name in component_or_optimizer_list:
            if component_or_optimizer is not None:
                name = name + suffix
                path_to_save_at = f"{model_dir}/{name}_{step}.pt"
                if name == "log_alpha":
                    torch.save(component_or_optimizer, path_to_save_at)
                else:
                    torch.save(component_or_optimizer.state_dict(), path_to_save_at)
                print(f"Saved {path_to_save_at}")
                if retain_last_n == -1:
                    continue
                reverse_sorted_existing_versions = (
                    _get_reverse_sorted_existing_versions(model_dir_path, name)
                )
                if len(reverse_sorted_existing_versions) > retain_last_n:
                    # assert len(reverse_sorted_existing_versions) == retain_last_n + 1
                    for path_to_del in reverse_sorted_existing_versions[retain_last_n:]:
                        if os.path.lexists(path_to_del):
                            os.remove(path_to_del)
                            print(f"Deleted {path_to_del}")

    def save_metadata(self, model_dir: str, step: int) -> None:
        """Save the metadata.

        Args:
            model_dir (str): directory to save.
            step (int): step for tracking the training of the agent.

        """
        metadata = {"step": step}
        path_to_save_at = f"{model_dir}/metadata.pt"
        torch.save(metadata, path_to_save_at)
        print(f"Saved {path_to_save_at}")

    def load(self, model_dir: Optional[str], step: Optional[int]) -> None:
        """Load the agent.

        Args:
            model_dir (Optional[str]): directory to load the model from.
            step (Optional[int]): step for tracking the training of the agent.
        """

        if model_dir is None or step is None:
            return
        for component, name in self.get_component_name_list_for_checkpointing():
            component = _load_component_or_optimizer(
                component,
                model_dir=model_dir,
                name=name,
                step=step,
                num_envs=self.num_envs,
            )
            if isinstance(component, ComponentType):
                component = component.to(self.device)
        for optimizer, name in self.get_optimizer_name_list_for_checkpointing():
            optimizer = _load_component_or_optimizer(
                component_or_optimizer=optimizer,
                model_dir=model_dir,
                name=name + self._opimizer_suffix,
                step=step,
                num_envs=self.num_envs,
            )

    def load_latest_step(self, model_dir: str) -> int:
        """Load the agent using the latest training step.

        Args:
            model_dir (Optional[str]): directory to load the model from.

        Returns:
            int: step for tracking the training of the agent.
        """
        latest_step = -1
        if model_dir is None:
            print("model_dir is None.")
            return latest_step
        metadata = self.load_metadata(model_dir=model_dir)
        if metadata is None:
            return latest_step + 1
        latest_step = metadata["step"]
        self.load(model_dir, step=latest_step)
        return latest_step + 1

    def load_metadata(self, model_dir: str) -> Optional[Dict[Any, Any]]:
        """Load the metadata of the agent.

        Args:
            model_dir (str): directory to load the model from.

        Returns:
            Optional[Dict[Any, Any]]: metadata.
        """
        metadata_path = f"{model_dir}/metadata.pt"
        if not os.path.exists(metadata_path):
            print(f"{metadata_path} does not exist.")
            metadata = None
        else:
            metadata = torch.load(metadata_path)
        return metadata


def _get_reverse_sorted_existing_versions(model_dir_path: Path, name: str) -> List[str]:
    """List of model components in reverse sorted order.

    Args:
        model_dir_path (Path): directory to find components in.
        name (str): name of the component.

    Returns:
        List[str]: list of model components in reverse sorted order.
    """
    existing_versions: List[str] = [str(x) for x in model_dir_path.glob(f"{name}_*.pt")]
    existing_versions = [
        x
        for x in existing_versions
        if is_integer(x.rsplit("/", 1)[-1].replace(f"{name}_", "").replace(".pt", ""))
    ]
    existing_versions.sort(reverse=True, key=_get_step_from_model_path)
    return existing_versions


def _get_step_from_model_path(_path: str) -> int:
    """Parse the model path to obtain the

    Args:
        _path (str): path to the model.

    Returns:
        int: step for tracking the training of the agent.
    """
    return int(_path.rsplit("/", 1)[-1].replace(".pt", "").rsplit("_", 1)[-1])


@overload
def _load_component_or_optimizer(
    component_or_optimizer: ComponentType,
    model_dir: str,
    name: str,
    step: int,
    num_envs: int,
) -> ComponentType:
    ...


@overload
def _load_component_or_optimizer(
    component_or_optimizer: OptimizerType,
    model_dir: str,
    name: str,
    step: int,
    num_envs: int,
) -> OptimizerType:
    ...


def _load_component_or_optimizer(
    component_or_optimizer: ComponentOrOptimizerType,
    model_dir: str,
    name: str,
    step: int,
    num_envs: int,
) -> ComponentOrOptimizerType:
    """Load a component/optimizer for the agent.

    Args:
        component_or_optimizer (ComponentOrOptimizerType): component or
            optimizer to load.
        model_dir (str): directory to load from.
        name (str): name of the component.
        step (int): step for tracking the training of the agent.
        num_envs (int): number of environments in the task.

    Returns:
        ComponentOrOptimizerType: loaded component or
            optimizer.
    """

    assert component_or_optimizer is not None
    # if component_or_optimizer is not None:
    path_to_load_from = f"{model_dir}/{name}_{step}.pt"
    print(f"path_to_load_from: {path_to_load_from}")
    if os.path.exists(path_to_load_from):
        if name == "log_alpha":
            component_or_optimizer = torch.load(path_to_load_from)
        else:
            component_or_optimizer.load_state_dict(torch.load(path_to_load_from))
    else:
        print(f"No component to load from {path_to_load_from}")
    return component_or_optimizer
