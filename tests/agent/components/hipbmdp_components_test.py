# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Tuple

import hydra
import pytest
import torch
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf as OC

from mtrl.agent.components.encoder import make_encoder
from mtrl.agent.components.hipbmdp_theta import ThetaSamplingStrategy
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.experiment import experiment
from mtrl.utils.types import EnvMetaDataType
from tests.agent.utils import get_hip_env_and_metadata

agent_name = "sac"
env_name = "dmcontrol-finger-spin-distribution-v0"
setup = "hipbmdp"


def get_config(overrides: List[str], env_metadata: EnvMetaDataType):
    with initialize(config_path="../../../config"):
        config = compose(
            config_name="config",
            overrides=[
                f"agent={agent_name}",
                f"env={env_name}",
                "experiment.num_eval_episodes=1",
                f"setup={setup}",
            ]
            + overrides,
        )
        config = experiment.prepare_config(config=config, env_metadata=env_metadata)
    return config


@pytest.fixture(scope="session")
def get_hip_train_env_and_metadata():
    env, env_metadata = get_hip_env_and_metadata(env_name=env_name, mode="train")
    yield env, env_metadata
    env.close()


def get_overrides_using_encoders(encoders: List[str]) -> List[Tuple[List[str], str]]:
    modes = ["train", "eval"]
    overrides = []
    for mode in modes:
        for enc in encoders:
            enc_override_str = f"agent.encoder.type_to_select={enc}"
            overrides.append(([enc_override_str], mode))
    return overrides


def get_overrides_for_testing_encoder():
    overrides = []

    for _override in get_overrides_using_encoders(encoders=["pixel"]):
        overrides.append((*_override, 50))

    return overrides


@pytest.mark.parametrize(
    "overrides, mode, encoding_dim", get_overrides_for_testing_encoder()
)
def test_encoder_actor_and_critic(
    overrides: List[str], mode: str, encoding_dim: int, get_hip_train_env_and_metadata
) -> None:

    env, env_metadata = get_hip_train_env_and_metadata
    config = get_config(overrides=overrides, env_metadata=env_metadata)
    encoder_cfg = config.agent.encoder
    key = "type_to_select"
    if key in encoder_cfg:
        encoder_type_to_select = encoder_cfg[key]
        encoder_cfg = encoder_cfg[encoder_type_to_select]
    encoder = make_encoder(
        env_obs_shape=env_metadata["env_obs_space"].shape,
        encoder_cfg=encoder_cfg,
        multitask_cfg=OC.create({}),
    )
    actor = hydra.utils.instantiate(
        config.agent.actor,
        env_obs_shape=env_metadata["env_obs_space"].shape,
        action_shape=env_metadata["action_space"].shape,
    )
    critic = hydra.utils.instantiate(
        config.agent.critic,
        env_obs_shape=env_metadata["env_obs_space"].shape,
        action_shape=env_metadata["action_space"].shape,
    )

    num_envs = len(env.ids)
    multitask_obs = env.reset()
    task_info = TaskInfo(
        encoding=None,
        compute_grad=False,
        env_index=multitask_obs["task_obs"],
    )
    mtobs = MTObs(
        env_obs=multitask_obs["env_obs"],
        task_obs=None,
        task_info=task_info,
    )

    with torch.no_grad():
        z = encoder(mtobs)
        action_mu, _, _, _ = actor(mtobs)
        q1, q2 = critic(mtobs=mtobs, action=action_mu)
    assert z.shape == (num_envs, encoding_dim)
    assert action_mu.shape == (num_envs, *env_metadata["action_space"].shape)
    assert q1.shape == (num_envs, 1)

    # action = np.asarray([env.action_space.sample() for _ in range(num_envs)])
    multitask_obs, reward, done, info = env.step(action_mu.numpy())
    mtobs = MTObs(
        env_obs=multitask_obs["env_obs"],
        task_obs=None,
        task_info=task_info,
    )
    with torch.no_grad():
        z = encoder(mtobs)
        action_mu, _, _, _ = actor(mtobs)
        q1, q2 = critic(mtobs=mtobs, action=action_mu)
    assert z.shape == (num_envs, encoding_dim)
    assert action_mu.shape == (num_envs, *env_metadata["action_space"].shape)
    assert q1.shape == (num_envs, 1)


def get_overrides_for_testing_task_encoder():
    overrides = []
    for sampling_strategy in ["embedding", "mean_train"]:
        overrides.append(
            [
                f"agent.multitask.task_encoder_cfg.sampling_strategy.train={sampling_strategy}"
            ]
        )
    return overrides


@pytest.mark.parametrize(
    "overrides",
    get_overrides_for_testing_task_encoder(),
)
def test_task_encoder(overrides: List[str], get_hip_train_env_and_metadata) -> None:
    env, env_metadata = get_hip_train_env_and_metadata
    config = get_config(overrides=overrides, env_metadata=env_metadata)
    task_encoder = hydra.utils.instantiate(
        config.agent.multitask.task_encoder_cfg.model_cfg,
    )
    num_envs = len(env.ids)
    env_index = torch.randint(1, num_envs, (num_envs,))
    output = task_encoder(
        env_index=env_index,
        theta_sampling_strategy=[ThetaSamplingStrategy("embedding")],
        modes=["train"],
    )
    assert output.shape == (num_envs, 50)
