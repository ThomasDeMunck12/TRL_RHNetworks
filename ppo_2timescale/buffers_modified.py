# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 11:39:21 2025

@author: thoma
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    DictRolloutBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class HBaseBuffer(ABC):
    """
    Base class that represent a buffer used for hierarchical learning (rollout only)

    :param buffer_size: Number of element in the buffer for env 2. 
    :param observation_space1: Observation space in env 1
    :param action_space1: Action space in env 1
    :param observation_space2: Observation space in env 2
    :param action_space2: Action space in env 2
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space1: spaces.Space
    observation_space2: spaces.Space

    obs_shape1: tuple[int, ...]
    obs_shape2: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        p_interval: int,
        observation_space1: spaces.Space,
        observation_space2: spaces.Space,
        action_space1: spaces.Space,
        action_space2: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.p_interval = p_interval

        self.observation_space1 = observation_space1
        self.observation_space2 = observation_space2

        self.action_space1 = action_space1
        self.action_space2 = action_space2

        self.obs_shape1 = get_obs_shape(observation_space1)  # type: ignore[assignment]
        self.obs_shape2 = get_obs_shape(observation_space2)  # type: ignore[assignment]

        self.action_dim1 = get_action_dim(action_space1)
        self.action_dim2 = get_action_dim(action_space2)

        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add1(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()
        
    def add2(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend1(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add1(*data)
            
    def extend2(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add2(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples1(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()
    
    def _get_samples2(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward

class HRolloutBuffer(HBaseBuffer):
    """
    HRollout buffer used in hierarchical on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be confused with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space1: Observation space in env 1
    :param action_space1: Action space in env 1
    :param observation_space2: Observation space in env 2
    :param action_space2: Action space in env 2
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs. variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations1: np.ndarray
    observations2: np.ndarray

    actions1: np.ndarray
    actions2: np.ndarray

    rewards1: np.ndarray
    rewards2: np.ndarray

    advantages1: np.ndarray
    advantages2: np.ndarray

    returns1: np.ndarray
    returns2: np.ndarray

    episode_starts1: np.ndarray
    episode_starts2: np.ndarray

    log_probs1: np.ndarray
    log_probs2: np.ndarray

    values1: np.ndarray
    values2: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        p_interval: int,
        observation_space1: spaces.Space,
        observation_space2: spaces.Space,

        action_space1: spaces.Space,
        action_space2: spaces.Space,

        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, p_interval, observation_space1, observation_space2, action_space1, action_space2, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready1 = False
        self.generator_ready2 = False
        self.reset()

    def reset(self) -> None:
        buffer_size1 = int(self.buffer_size/self.p_interval)
        self.observations1 = np.zeros((buffer_size1, self.n_envs, *self.obs_shape1), dtype=self.observation_space1.dtype)
        self.observations2 = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape2), dtype=self.observation_space2.dtype)
        self.actions1 = np.zeros((buffer_size1, self.n_envs, self.action_dim1), dtype=self.action_space1.dtype)
        self.actions2 = np.zeros((self.buffer_size, self.n_envs, self.action_dim2), dtype=self.action_space2.dtype)
        self.rewards1 = np.zeros((buffer_size1, self.n_envs), dtype=np.float32)
        self.rewards2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns1 = np.zeros((buffer_size1, self.n_envs), dtype=np.float32)
        self.returns2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts1 = np.zeros((buffer_size1, self.n_envs), dtype=np.float32)
        self.episode_starts2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values1 = np.zeros((buffer_size1, self.n_envs), dtype=np.float32)
        self.values2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs1 = np.zeros((buffer_size1, self.n_envs), dtype=np.float32)
        self.log_probs2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages1 = np.zeros((buffer_size1, self.n_envs), dtype=np.float32)
        self.advantages2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready1 = False
        self.generator_ready2 = False

        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam1 = 0
        last_gae_lam2 = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values2 = last_values
                delta2 = self.rewards2[step] + self.gamma * next_values2 * next_non_terminal - self.values2[step]
                last_gae_lam2 = delta2 + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam2
                self.advantages2[step] = last_gae_lam2
                
            else:
                if step % self.p_interval != 0:
                   if (step + 1) % self.p_interval == 0:
                       next_non_terminal = 1.0 - self.episode_starts1[(step + 1)//self.p_interval]
                   else:
                       next_non_terminal = 1.0
                   next_values2 = self.values2[step + 1]
                   delta2 = self.rewards2[step] + self.gamma * next_values2 * next_non_terminal - self.values2[step]
                   last_gae_lam2 = delta2 + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam2
                   self.advantages2[step] = last_gae_lam2
                else: 
                   next_non_terminal = 1.0 
                   next_values1 = self.values2[step + 1]
                   next_values2 = self.values2[step + 1]
                   delta1 = self.rewards2[step] + self.gamma * next_values1 * next_non_terminal - self.values1[step//self.p_interval]
                   delta2 = self.rewards2[step] + self.gamma * next_values2 * next_non_terminal - self.values2[step]

                   last_gae_lam1 = delta1 + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam2
                   last_gae_lam2 = delta2 + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam2

                   self.advantages1[step//self.p_interval] = last_gae_lam1
                   self.advantages2[step] = last_gae_lam2
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA

        self.returns1 = self.advantages1 + self.values1
        self.returns2 = self.advantages2 + self.values2
        np.set_printoptions(threshold=np.inf)

    def add1(
        self,
        obs1: np.ndarray,
        action1: np.ndarray,
        reward1: np.ndarray,
        episode_start1: np.ndarray,
        value1: th.Tensor,
        log_prob1: th.Tensor,
    ) -> None:
        """
        :param obs1: Observation
        :param action1: Action
        :param reward1:
        :param episode_start1: Start of episode signal.
        :param value1: estimated value of the current state
            following the current policy.
        :param log_prob1: log probability of the action
            following the current policy.
        """
        pos1 = int(self.pos/self.p_interval)
        if len(log_prob1.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob1 = log_prob1.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space1, spaces.Discrete):
            obs1 = obs1.reshape((self.n_envs, *self.obs_shape1))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action1 = action1.reshape((self.n_envs, self.action_dim1))

        self.observations1[pos1] = np.array(obs1)
        self.actions1[pos1] = np.array(action1)
        self.rewards1[pos1] = np.array(reward1)
        self.episode_starts1[pos1] = np.array(episode_start1)
        self.values1[pos1] = value1.clone().cpu().numpy().flatten()
        self.log_probs1[pos1] = log_prob1.clone().cpu().numpy()
        
    def add2(
        self,
        obs2: np.ndarray,
        action2: np.ndarray,
        reward2: np.ndarray,
        episode_start2: np.ndarray,
        value2: th.Tensor,
        log_prob2: th.Tensor,
    ) -> None:
        """
        :param obs2: Observation
        :param action2: Action
        :param reward2:
        :param episode_start2: Start of episode signal.
        :param value2: estimated value of the current state
            following the current policy.
        :param log_prob2: log probability of the action
            following the current policy.
        """
        if len(log_prob2.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob2 = log_prob2.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space2, spaces.Discrete):
            obs2 = obs2.reshape((self.n_envs, *self.obs_shape2))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action2 = action2.reshape((self.n_envs, self.action_dim2))

        self.observations2[self.pos] = np.array(obs2)
        self.actions2[self.pos] = np.array(action2)
        self.rewards2[self.pos] = np.array(reward2)
        self.episode_starts2[self.pos] = np.array(episode_start2)
        self.values2[self.pos] = value2.clone().cpu().numpy().flatten()
        self.log_probs2[self.pos] = log_prob2.clone().cpu().numpy()
        self.pos += 1
       
        if self.pos == self.buffer_size:
            self.full = True
    
    def get1(self, batch_size1: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size / self.p_interval * self.n_envs)
        # Prepare the data
        if not self.generator_ready1:
            _tensor_names = [
                "observations1",
                "actions1",
                "values1",
                "log_probs1",
                "advantages1",
                "returns1",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready1 = True

        # Return everything, don't create minibatches
        if batch_size1 is None:
            batch_size1 = self.buffer_size / self.p_interval * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size // self.p_interval * self.n_envs:
            yield self._get_samples1(indices[start_idx : start_idx + batch_size1])
            start_idx += batch_size1

    def _get_samples1(
        self,
        batch_inds: np.ndarray,
        env1: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations1[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions1[batch_inds].astype(np.float32, copy=False),
            self.values1[batch_inds].flatten(),
            self.log_probs1[batch_inds].flatten(),
            self.advantages1[batch_inds].flatten(),
            self.returns1[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get2(self, batch_size2: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready1:
            _tensor_names = [
                "observations2",
                "actions2",
                "values2",
                "log_probs2",
                "advantages2",
                "returns2",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready1 = True

        # Return everything, don't create minibatches
        if batch_size2 is None:
            batch_size2 = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples2(indices[start_idx : start_idx + batch_size2])
            start_idx += batch_size2

    def _get_samples2(
        self,
        batch_inds: np.ndarray,
        env2: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations2[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions2[batch_inds].astype(np.float32, copy=False),
            self.values2[batch_inds].flatten(),
            self.log_probs2[batch_inds].flatten(),
            self.advantages2[batch_inds].flatten(),
            self.returns2[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class HDictRolloutBuffer(HRolloutBuffer):
    """
    HDict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the HRolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space1: Observation space 1
    :param observation_space2: Observation space 2
    :param action_space1: Action space 1
    :param action_space1: Action space 2
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observation_space1: spaces.Dict
    observation_space2: spaces.Dict

    obs_shape1: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    obs_shape2: dict[str, tuple[int, ...]]  # type: ignore[assignment]

    observations1: dict[str, np.ndarray]  # type: ignore[assignment]
    observations2: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        p_interval: int,
        observation_space1: spaces.Dict,
        observation_space2: spaces.Dict,
        action_space1: spaces.Space,
        action_space2: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(HRolloutBuffer, self).__init__(buffer_size, p_interval, observation_space1, observation_space2, action_space1, action_space2, device, n_envs=n_envs)

        assert isinstance(self.obs_shape1, dict), "HDictRolloutBuffer must be used with Dict obs space only"
        assert isinstance(self.obs_shape2, dict), "HDictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready1 = False
        self.generator_ready2 = False

        self.reset()

    def reset(self) -> None:
        self.observations1 = {}
        self.observations2 = {}
        
        size_buffer1 = int(self.buffer_size/ self.p_interval)
        for key, obs_input_shape1 in self.obs_shape1.items():
            self.observations1[key] = np.zeros(
                (size_buffer1, self.n_envs, *obs_input_shape1), dtype=self.observation_space1[key].dtype
            )
        for key, obs_input_shape2 in self.obs_shape2.items():
            self.observations2[key] = np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape2), dtype=self.observation_space2[key].dtype
            )
        self.actions1 = np.zeros((size_buffer1, self.n_envs, self.action_dim1), dtype=self.action_space1.dtype)
        self.actions2 = np.zeros((self.buffer_size, self.n_envs, self.action_dim2), dtype=self.action_space2.dtype)
        self.rewards1 = np.zeros((size_buffer1, self.n_envs), dtype=np.float32)
        self.rewards2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns1 = np.zeros((size_buffer1, self.n_envs), dtype=np.float32)
        self.returns2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts1 = np.zeros((size_buffer1, self.n_envs), dtype=np.float32)
        self.episode_starts2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values1 = np.zeros((size_buffer1, self.n_envs), dtype=np.float32)
        self.values2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs1 = np.zeros((size_buffer1, self.n_envs), dtype=np.float32)
        self.log_probs2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages1 = np.zeros((size_buffer1, self.n_envs), dtype=np.float32)
        self.advantages2 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.generator_ready1 = False
        self.generator_ready2 = False

        super(HRolloutBuffer, self).reset()

    def add1(  # type: ignore[override]
        self,
        obs1: dict[str, np.ndarray],
        action1: np.ndarray,
        reward1: np.ndarray,
        episode_start1: np.ndarray,
        value1: th.Tensor,
        log_prob1: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        pos1 = int(self.pos/self.p_interval)
        if len(log_prob1.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob1 = log_prob1.reshape(-1, 1)

        for key in self.observations1.keys():
            obs_ = np.array(obs1[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space1.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape1[key])
            self.observations1[key][pos1] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action1 = action1.reshape((self.n_envs, self.action_dim1))

        self.actions1[pos1] = np.array(action1)
        self.rewards1[pos1] = np.array(reward1)
        self.episode_starts1[pos1] = np.array(episode_start1)
        self.values1[pos1] = value1.clone().cpu().numpy().flatten()
        self.log_probs1[pos1] = log_prob1.clone().cpu().numpy()
        
    def add2(  # type: ignore[override]
        self,
        obs2: dict[str, np.ndarray],
        action2: np.ndarray,
        reward2: np.ndarray,
        episode_start2: np.ndarray,
        value2: th.Tensor,
        log_prob2: th.Tensor,
    ) -> None:
        """
        :param obs2: Observation
        :param action2: Action
        :param reward2:
        :param episode_start2: Start of episode signal.
        :param value2: estimated value of the current state
            following the current policy.
        :param log_prob2: log probability of the action
            following the current policy.
        """
        if len(log_prob2.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob2 = log_prob2.reshape(-1, 1)

        for key in self.observations2.keys():
            obs_ = np.array(obs2[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space2.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape2[key])
            self.observations2[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action2 = action2.reshape((self.n_envs, self.action_dim2))

        self.actions2[self.pos] = np.array(action2)
        self.rewards2[self.pos] = np.array(reward2)
        self.episode_starts2[self.pos] = np.array(episode_start2)
        self.values2[self.pos] = value2.clone().cpu().numpy().flatten()
        self.log_probs2[self.pos] = log_prob2.clone().cpu().numpy()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        
    def get1(  # type: ignore[override]
        self,
        batch_size1: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        perm = int(self.buffer_size / self.p_interval)
        indices = np.random.permutation( perm * self.n_envs)
        # Prepare the data
        if not self.generator_ready1:
            for key, obs in self.observations1.items():
                self.observations1[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions1", "values1", "log_probs1", "advantages1", "returns1"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready1 = True
        # Return everything, don't create minibatches
        if batch_size1 is None:
            batch_size1 = self.buffer_size / self.n_inteval * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size // self.p_interval * self.n_envs:
            yield self._get_samples1(indices[start_idx : start_idx + batch_size1])
            start_idx += batch_size1

    def _get_samples1(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env1: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations1.items()},
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            actions=self.to_torch(self.actions1[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values1[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs1[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages1[batch_inds].flatten()),
            returns=self.to_torch(self.returns1[batch_inds].flatten()),
        )
    
    def get2(  # type: ignore[override]
        self,
        batch_size2: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready2:
            for key, obs in self.observations2.items():
                self.observations2[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions2", "values2", "log_probs2", "advantages2", "returns2"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready2 = True

        # Return everything, don't create minibatches
        if batch_size2 is None:
            batch_size2 = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples2(indices[start_idx : start_idx + batch_size2])
            start_idx += batch_size2

    def _get_samples2(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env2: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations2.items()},
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            actions=self.to_torch(self.actions2[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values2[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs2[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages2[batch_inds].flatten()),
            returns=self.to_torch(self.returns2[batch_inds].flatten()),
        )
