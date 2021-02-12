# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hashlib
import itertools
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

OverridesType = Dict[str, str]


def get_command(
    experiment_name: str,
    agent: str,
    env: str,
    _id: str,
    num_train_steps: int,
    frame_skip: Optional[int],
    num_envs: int,
    should_use_disentangled_alpha: bool,
    should_use_task_encoder: bool,
    should_use_multi_head_policy: bool,
    should_use_soft_modularization: bool,
    overrides: OverridesType,
) -> List[str]:
    seed = 1
    description = "pytest"
    num_eval_episodes = 2
    retain_last_n = 1
    init_steps = 10
    eval_freq = 10
    parameters = {
        "experiment.name": experiment_name,
        "agent": agent,
        "env": env,
        "setup.id": _id,
        "setup.seed": seed,
        "setup.debug.should_enable": True,
        "setup.description": description,
        "experiment.num_eval_episodes": num_eval_episodes,
        "experiment.save.model.retain_last_n": retain_last_n,
        "experiment.init_steps": init_steps,
        "experiment.num_train_steps": num_train_steps,
        "experiment.eval_freq": eval_freq,
        "metrics": "hipbmdp",
        "agent.multitask.num_envs": num_envs,
        "agent.multitask.should_use_disentangled_alpha": should_use_disentangled_alpha,
        "agent.multitask.should_use_task_encoder": should_use_task_encoder,
        "agent.multitask.should_use_multi_head_policy": should_use_multi_head_policy,
    }
    # parameters[
    #     "agent.multitask.metaworld.actor_cfg.moe_cfg.should_use"
    # ] = should_use_soft_modularization

    if should_use_soft_modularization:
        parameters[
            "agent.multitask.metaworld.actor_cfg.should_condition_model_on_task_info"
        ] = True
        parameters[
            "agent.multitask.metaworld.actor_cfg.should_condition_encoder_on_task_info"
        ] = False

    if frame_skip is not None:
        parameters["env.builder.make_kwargs.frame_skip"] = frame_skip

    parameters.update(overrides)
    cmd = [sys.executable, "main.py"]
    for key, value in parameters.items():
        if "$" in str(key) or "$" in str(value):
            cmd.append(f"{key}={value}")
        else:
            cmd.append(f"{key}={value}")
    return cmd


def check_output_from_cmd(
    experiment_name: str,
    agent: str,
    env: str,
    _id: str,
    num_train_steps: int,
    frame_skip: Optional[int],
    num_steps_per_episode: int,
    num_envs: int,
    should_use_disentangled_alpha: bool,
    should_use_task_encoder: bool,
    should_use_multi_head_policy: bool,
    should_use_soft_modularization: bool,
    overrides: OverridesType,
):

    cmd = get_command(
        experiment_name=experiment_name,
        agent=agent,
        env=env,
        _id=_id,
        num_train_steps=num_train_steps,
        frame_skip=frame_skip,
        num_envs=num_envs,
        should_use_disentangled_alpha=should_use_disentangled_alpha,
        should_use_task_encoder=should_use_task_encoder,
        should_use_multi_head_policy=should_use_multi_head_policy,
        should_use_soft_modularization=should_use_soft_modularization,
        overrides=overrides,
    )
    cwd = os.getcwd()
    assert os.path.exists(f"{cwd}/logs/{_id}") is False
    print(" ".join(cmd))
    result = subprocess.check_output(cmd)
    actual_logs = set(result.decode("utf-8").split("\n"))
    cwd = os.getcwd()
    shutil.rmtree(f"{cwd}/logs/{_id}")
    assert os.path.exists(f"{cwd}/logs/{_id}") is False
    component_names = [
        "actor",
        "critic",
        "critic_target",
        "log_alpha",
    ]
    if "sac_ae" in agent and ("state_" not in agent):
        component_names.append("decoder")
    elif "deepmdp" in agent:
        component_names.append("transition_model")
        component_names.append("reward_decoder")
        if "state_" not in agent:
            component_names.append("decoder")
    if frame_skip is None:
        frame_skip_num = 1
    else:
        frame_skip_num = frame_skip
    steps = list(range(0, num_train_steps, int(num_steps_per_episode / frame_skip_num)))
    separators = ["_", "_optimizer_"]
    prefixs = ["Saved", "Deleted"]
    expected_logs = []
    for component, step, separator, prefix in itertools.product(
        component_names, steps, separators, prefixs
    ):
        if component in ["metadata", "critic_target"] and separator == "_optimizer_":
            continue
        if step == steps[-1] and prefix == "Deleted":
            continue
        expected_logs.append(
            f"{prefix} {cwd}/logs/{_id}/model/{component}{separator}{step}.pt"
        )
    for log in expected_logs:
        if log not in actual_logs:
            breakpoint()
        assert log in actual_logs


def map_combination_of_params_to_config(
    combination: List[Any], keys: List[str]
) -> Dict[str, Any]:
    config = {}
    for index, key in enumerate(keys):
        config[key] = combination[index]
    return config


def get_configs_to_test(params_to_test: Dict[str, List[Any]]):
    combinations = list(itertools.product(*params_to_test.values()))
    configs_to_test = [
        map_combination_of_params_to_config(
            combination=combination, keys=list(params_to_test.keys())
        )
        for combination in combinations
    ]
    return configs_to_test


def map_config_to_string(config: Dict[str, Any]) -> str:
    config_str = "_".join(
        [
            f"{key}-{value}"
            for (key, value) in config.items()
            if "$" not in str(key) and "$" not in str(value)
        ]
    )
    return hashlib.sha224(config_str.encode()).hexdigest()


def get_test_id(experiment_name: str, config: Dict[str, Any]) -> str:

    return f"pytest_{experiment_name}_" + map_config_to_string(config=config)
