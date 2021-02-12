# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Tuple

import hydra
import pytest
import torch
from hydra.experimental import compose, initialize

from mtrl.experiment import experiment
from mtrl.utils.types import EnvMetaDataType
from tests.agent.utils import get_metaworld_env_and_metadata

env_name = "metaworld-mt10"
setup = "metaworld"


def get_config(overrides: List[str], env_metadata: EnvMetaDataType):
    with initialize(config_path="../../config"):
        config = compose(
            config_name="config",
            overrides=[
                f"env={env_name}",
                "experiment.num_eval_episodes=1",
                f"setup={setup}",
                "agent.multitask.should_use_task_encoder=True",
                "agent.multitask.actor_cfg.should_condition_encoder_on_task_info=True",
                "agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True",
            ]
            + overrides,
        )
        config = experiment.prepare_config(config=config, env_metadata=env_metadata)
    return config


@pytest.fixture(scope="session")
def get_mt10_train_env_and_metadata():
    env, env_metadata = get_metaworld_env_and_metadata(env_name=env_name, mode="train")
    yield env, env_metadata
    env.close()


def get_overrides_using_encoders_and_meta_encoders(
    encoders: List[str], meta_encoders: List[str]
) -> List[Tuple[List[str], str]]:
    modes = ["train", "eval"]
    overrides = []
    for agent_name in [
        "state_sac",
        "state_deepmdp",
        "pcgrad_state_sac",
        "pcgrad_state_deepmdp",
        "gradnorm_state_sac",
        "gradnorm_state_deepmdp",
    ]:
        agent_override_str = f"agent={agent_name}"
        for mode in modes:
            for meta_enc in meta_encoders:
                meta_enc_override_str = f"agent.encoder.type_to_select={meta_enc}"
                for task_id_to_encoder_id_mode in [
                    # "identity",
                    "cluster",
                    "ensemble",
                    # "gate",
                    # "attention",
                ]:
                    task_id_to_encoder_id_override_str = f"agent.encoder.moe.task_id_to_encoder_id_cfg.mode={task_id_to_encoder_id_mode}"
                    overrides.append(
                        (
                            [
                                meta_enc_override_str,
                                task_id_to_encoder_id_override_str,
                                agent_override_str,
                            ],
                            mode,
                        )
                    )
                for enc in encoders:
                    sub_enc_override_str = (
                        f"agent.encoder.{meta_enc}.encoder_cfg=${{agent.encoder.{enc}}}"
                    )
                    overrides.append(
                        (
                            [
                                meta_enc_override_str,
                                sub_enc_override_str,
                                agent_override_str,
                            ],
                            mode,
                        )
                    )
            for enc in encoders:
                enc_override_str = f"agent.encoder.type_to_select={enc}"
                overrides.append(([enc_override_str, agent_override_str], mode))
    return overrides


def get_overrides_for_testing_agent():
    overrides = []
    for _override in get_overrides_using_encoders_and_meta_encoders(
        encoders=["feedforward"], meta_encoders=["moe"]
    ):
        overrides.append((*_override, 50))

    for _override in get_overrides_using_encoders_and_meta_encoders(
        encoders=["feedforward"], meta_encoders=[]
    ):
        overrides.append((*_override, 12))

    return overrides


@pytest.mark.parametrize(
    "overrides, mode, encoding_dim", get_overrides_for_testing_agent()
)
def test_agent(
    overrides: List[str], mode: str, encoding_dim: int, get_mt10_train_env_and_metadata
) -> None:

    env, env_metadata = get_mt10_train_env_and_metadata
    config = get_config(overrides=overrides, env_metadata=env_metadata)
    agent = hydra.utils.instantiate(
        config.agent.builder,
        env_obs_shape=env_metadata["env_obs_space"].shape,
        action_shape=env_metadata["action_space"].shape,
        action_range=[
            float(env_metadata["action_space"].low.min()),
            float(env_metadata["action_space"].high.max()),
        ],
        device=torch.device("cpu"),
    )

    num_envs = len(env.ids)

    multitask_obs = env.reset()
    action = agent.select_action(multitask_obs, modes=["train"])
    assert action.shape == (num_envs, *env_metadata["action_space"].shape)
    multitask_obs, reward, done, info = env.step(action)
    action = agent.sample_action(multitask_obs, modes=["train"])
    assert action.shape == (num_envs, *env_metadata["action_space"].shape)
