# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 12:51:14 2025

@author: thoma
"""

import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

def sync_env_state_eval(src_env, dst_env, obs_dict):
    """
    Synchronize the state of dst_env with src_env.

        src_env: source VecEnv (or single env)
        dst_env: destination VecEnv (or single env)
        obs_dict: dict of observations from src_env
        src_last_obs: the last obs dict from src_env
        src_last_episode_starts: the episode starts flag from src_env

    Returns:
        new_last_obs, new_last_episode_starts for dst_env
        """
    for j in range(src_env.num_envs):
        obs_j = {k: v[j] for k, v in obs_dict.items()}  # slice per env
        # call the underlying set_state
        dst_env.env_method("SetState", obs_j, indices=j)

def hevaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env1: Union[gym.Env, VecEnv],
    env2: Union[gym.Env, VecEnv],
    p_interval: int,
    n_eval_episodes: int = 10,
    
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped1 = False
    is_monitor_wrapped2 = False

    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env1, VecEnv):
        env1 = DummyVecEnv([lambda: env1])  # type: ignore[list-item, return-value]

    is_monitor_wrapped1 = is_vecenv_wrapped(env1, VecMonitor) or env1.env_is_wrapped(Monitor)[0]

    if not isinstance(env2, VecEnv):
        env2 = DummyVecEnv([lambda: env2])  # type: ignore[list-item, return-value]

    is_monitor_wrapped2 = is_vecenv_wrapped(env2, VecMonitor) or env2.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped1 and warn:
        warnings.warn(
            "Evaluation environment 1 is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
        
    if not is_monitor_wrapped2 and warn:
        warnings.warn(
            "Evaluation environment 2 is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env1.num_envs

    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env1.reset()
    observations = env2.reset()
    states = None

    episode_starts = np.ones((env2.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        if (current_lengths % p_interval).any() == 0:
    
            actions1, states = model.predict1(
                observations,  # type: ignore[arg-type]
                state1=states,
                episode_start1=episode_starts,
                deterministic=deterministic,
                )
            new_observations, rewards1, dones, infos = env1.step(actions1)

            observations = new_observations
            sync_env_state_eval(env1, env2, new_observations)  
            
        actions2, states = model.predict2(
            observations,  # type: ignore[arg-type]
            state2=states,
            episode_start2=episode_starts,
            deterministic=deterministic,
            )
        new_observations, rewards2, dones, infos = env2.step(actions2)

        current_rewards += rewards2
        current_lengths += 1
        observations = new_observations
        if (current_lengths % p_interval).any() == 0:
            sync_env_state_eval(env2, env1, new_observations)  
        
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards2[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped1 and is_monitor_wrapped2:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0


        if render:
            env1.render()
            env2.render()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward