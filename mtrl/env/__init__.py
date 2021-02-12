# type: ignore

import gym
from gym.envs.registration import register


def register_once(id, entry_point, **kwargs):
    if id not in gym.envs.registry.env_specs:
        register(id=id, entry_point=entry_point, **kwargs)


register_once(
    id="cartpole-distribution-v0",
    entry_point="codes.env.gym.cartpole:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)


register_once(
    id="cartpole-distribution-v1",
    entry_point="codes.env.gym.cartpole:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register_once(
    id="halfcheetah-distribution-v0",
    entry_point="codes.env.gym.half_cheetah:HalfCheetahEnvDistribution",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
