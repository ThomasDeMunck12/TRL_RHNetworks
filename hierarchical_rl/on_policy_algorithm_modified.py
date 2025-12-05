# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:47:31 2025

@author: thoma
"""

import sys
import time
import warnings
from typing import Callable, Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from base_class_modified import HierarchicalBaseAlgorithm
from buffers_modified import HDictRolloutBuffer, HRolloutBuffer
from callbacks_modified import HBaseCallback
from policies_modified import ActorCriticPolicy
#from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

MaybeCallback = Union[None, Callable, list["HBaseCallback"], "HBaseCallback"]
SelfHOnPolicyAlgorithm = TypeVar("SelfHOnPolicyAlgorithm", bound="HierarchicalOnPolicyAlgorithm")

class HierarchicalOnPolicyAlgorithm(HierarchicalBaseAlgorithm):
    """
    The base for Hierarchical On-Policy algorithms (ex: Hierarchical PPO).

    :param policy1: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param policy2: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env1: The environment to learn from (if registered in Gym, can be str)
    :param env2: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for environment 2 per update. Equivalently, n_steps/p_interval is the number of steps to 
    run for environment 1 per update.
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
    :param p_interval: The number of steps to run for env2 per step of env1
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs1: additional arguments to be passed to the policy on creation
    :param policy_kwargs2: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: HRolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy1: Union[str, type[ActorCriticPolicy]],
        policy2: Union[str, type[ActorCriticPolicy]],
        env1: Union[GymEnv, str],
        env2: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        p_interval: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[type[HRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs1: Optional[dict[str, Any]] = None,
        policy_kwargs2: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy1=policy1,
            policy2=policy2,
            env1=env1,
            env2=env2,
            learning_rate=learning_rate,
            policy_kwargs1=policy_kwargs1,
            policy_kwargs2=policy_kwargs2,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.p_interval = p_interval
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class

        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space2, spaces.Dict):
                self.rollout_buffer_class = HDictRolloutBuffer
            else:
                self.rollout_buffer_class = HRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.p_interval,
            self.observation_space1,  # type: ignore[arg-type]
            self.observation_space2,  # type: ignore[arg-type]
            self.action_space1,
            self.action_space2,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        
        self.policy1 = self.policy_class1(  # type: ignore[assignment]
            self.observation_space1, self.action_space1, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs1
        )
        self.policy1 = self.policy1.to(self.device)
        
        self.policy2 = self.policy_class2(  # type: ignore[assignment]
            self.observation_space2, self.action_space2, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs2
        )
        self.policy2 = self.policy2.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name1 = self.policy_class1.__name__
        if self.device != th.device("cpu") and policy_class_name1 == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name1} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )
        policy_class_name2 = self.policy_class2.__name__
        if self.device != th.device("cpu") and policy_class_name2 == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name2} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )
            
    def sync_env_state(self, src_env, dst_env, obs_dict, src_last_obs, src_last_episode_starts):
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

        # Sync obs + episode starts (deep copies to avoid shared references)
        new_last_obs = {k: np.array(v, copy=True) for k, v in src_last_obs.items()}
        new_last_episode_starts = np.copy(src_last_episode_starts)

        return new_last_obs, new_last_episode_starts
   
    def collect_rollouts(
        self,
        env1: VecEnv,
        env2: VecEnv,
        callback: HBaseCallback,
        rollout_buffer: HRolloutBuffer,
        n_rollout_steps: int,
        p_interval: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs1 is not None, "No previous observation was provided"
        assert self._last_obs2 is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy1.set_training_mode(False)
        self.policy2.set_training_mode(False)

        n_step_total = 0
        rollout_buffer.reset()
        
        # Sample new weights for the state dependent exploration
        
        callback.on_rollout_start()

        while n_step_total < n_rollout_steps:
            if n_step_total % p_interval == 0:
                with th.no_grad():
                    #print(n_step_total)
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs1, self.device)
                    actions, values, log_probs = self.policy1(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                
                if isinstance(self.action_space1, spaces.Box):
                    if self.policy1.squash_output:
                        # Unscale the actions to match env bounds
                        # if they were previously squashed (scaled in [-1, 1])
                        clipped_actions = self.policy1.unscale_action(clipped_actions)
                    else:
                        # Otherwise, clip the actions to avoid out of bound error
                        # as we are sampling from an unbounded Gaussian distribution
                        clipped_actions = np.clip(actions, self.action_space1.low, self.action_space1.high)
                
                new_obs, rewards, dones, infos = env1.step(clipped_actions)

                # Give access to local variables
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._update_info_buffer(infos, dones)
                #n_step_total += 1

                if isinstance(self.action_space1, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstrapping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                            done
                            and infos[idx].get("terminal_observation") is not None
                            and infos[idx].get("TimeLimit.truncated", False)
                            ):
                        terminal_obs = self.policy1.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy1.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add1(
                    self._last_obs1,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts1,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    )
                self._last_episode_starts1 = dones
                self._last_obs1 = new_obs
                self._last_obs2, self._last_episode_starts2 = self.sync_env_state(env1, env2, new_obs, self._last_obs1, self._last_episode_starts1)
                #for j in range(env2.num_envs):
                #    obs_j = {k: v[j] for k, v in new_obs.items()}  # slice per env
                #    env2.env_method("SetState", obs_j, indices=j)

                    # Keep your last-obs in sync (copy to avoid shared views)
                #   self._last_obs2 = {k: np.array(v, copy=True) for k, v in new_obs.items()}
                    # Usually episode_starts should be False right after a manual “teleport”
                #    self._last_episode_starts2 = np.copy(self._last_episode_starts1)

                        
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs2, self.device)
                
                actions, values, log_probs = self.policy2(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
            
            if isinstance(self.action_space2, spaces.Box):
                if self.policy2.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy2.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space2.low, self.action_space2.high)

            new_obs, rewards, dones, infos = env2.step(clipped_actions)

            self.num_timesteps += env2.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_step_total += 1

            if isinstance(self.action_space2, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                        ):
                    terminal_obs = self.policy2.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy2.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add2(
                self._last_obs2,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts2,  # type: ignore[arg-type]
                values,
                log_probs,
                )
                
            self._last_obs2 = new_obs  # type: ignore[assignment]
            self._last_episode_starts2 = dones

            if n_step_total % p_interval == 0 :
                self._last_obs1, self._last_episode_starts1 = self.sync_env_state(env2, env1, new_obs, self._last_obs2, self._last_episode_starts2)
                #for j in range(env1.num_envs):
                #   obs_j = {k: v[j] for k, v in new_obs.items()}  # slice per env
                #    env1.env_method("SetState", obs_j, indices=j)

                #    # Keep your last-obs in sync (copy to avoid shared views)
                #    self._last_obs1 = {k: np.array(v, copy=True) for k, v in new_obs.items()}
                #     # Usually episode_starts should be False right after a manual “teleport”
                #     self._last_episode_starts1 = np.copy(self._last_episode_starts2)

        with th.no_grad():
            # Compute value for the last timestep
            values1 = self.policy1.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            
        with th.no_grad():
            # Compute value for the last timestep
            values2 = self.policy2.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values2, dones=dones)
        
        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError
        
    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfHOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HierarchicalOnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env1 is not None
        assert self.env2 is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env1, self.env2, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, p_interval=self.p_interval)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)
                
            self.train()

        callback.on_training_end()

        return self
    
    def _get_torch_save_params(self) -> tuple[list[str], list[str], list[str], list[str]]:
        state_dicts = ["policy1", "policy1.optimizer", "policy2", "policy2.optimizer"]

        return state_dicts, []