# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:15:46 2025

@author: thoma
"""

"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Callable, Any, ClassVar, Optional, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common import utils
from callbacks_modified import HBaseCallback, HCallbackList, HConvertCallback, HProgressBarCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from policies_modified import BasePolicy
#from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict
from stable_baselines3.common.utils import (
    FloatSchedule,
    check_for_correct_spaces,
    get_device,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env

MaybeCallback = Union[None, Callable, list["HBaseCallback"], "HBaseCallback"]
SelfHBaseAlgorithm = TypeVar("SelfHBaseAlgorithm", bound="HierarchicalBaseAlgorithm")


def maybe_make_env(env: Union[GymEnv, str], verbose: int) -> GymEnv:
    """If env is a string, make the environment; otherwise, return env.

    :param env: The environment to learn from.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating if environment is created
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        env_id = env
        if verbose >= 1:
            print(f"Creating environment from the given name '{env_id}'")
        # Set render_mode to `rgb_array` as default, so we can record video
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except TypeError:
            env = gym.make(env_id)
    return env


class HierarchicalBaseAlgorithm(ABC):
    """
    The base of RL algorithms

    :param policy1: The policy model to use in env 1 (MlpPolicy, CnnPolicy, ...)
    :param policy2: The policy model to use in env 2(MlpPolicy, CnnPolicy, ...)

    :param env1: The environment 1 to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param env2: The environment 2 to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param policy_kwargs1: Additional arguments to be passed to the policy on creation
    :param policy_kwargs2: Additional arguments to be passed to the policy on creation

    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    # Policy aliases (see _get_policy_from_name())
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {}
    policy1: BasePolicy
    policy2: BasePolicy

    observation_space1: spaces.Space
    observation_space2: spaces.Space

    action_space1: spaces.Space
    action_space2: spaces.Space

    n_envs: int
    lr_schedule: Schedule
    _logger: Logger

    def __init__(
        self,
        policy1: Union[str, type[BasePolicy]],
        policy2: Union[str, type[BasePolicy]],
        env1: Union[GymEnv, str, None],
        env2: Union[GymEnv, str, None],
        learning_rate: Union[float, Schedule],
        policy_kwargs1: Optional[dict[str, Any]] = None,
        policy_kwargs2: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ) -> None:
        if isinstance(policy1, str):
            self.policy_class1 = self._get_policy_from_name(policy1)
        else:
            self.policy_class1 = policy1
            
        if isinstance(policy2, str):
            self.policy_class2 = self._get_policy_from_name(policy2)
        else:
            self.policy_class2 = policy2

        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs1 = {} if policy_kwargs1 is None else policy_kwargs1
        self.policy_kwargs2 = {} if policy_kwargs2 is None else policy_kwargs2

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs1 = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._last_obs2 = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._last_episode_starts1 = None  # type: Optional[np.ndarray]
        self._last_episode_starts2 = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs1 = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._last_original_obs2 = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env1: Optional[VecEnv] = None
        self.env2: Optional[VecEnv] = None
        self._vec_normalize_env1: Optional[VecNormalize] = None
        self._vec_normalize_env2: Optional[VecNormalize] = None

        # Create and wrap the env if needed
        if env1 is not None:
            env1 = maybe_make_env(env1, self.verbose)
            env1 = self._wrap_env(env1, self.verbose, monitor_wrapper)

            self.observation_space1 = env1.observation_space
            self.action_space1 = env1.action_space
            self.n_envs = env1.num_envs
            self.env1 = env1

            # get VecNormalize object if needed
            self._vec_normalize_env1 = unwrap_vec_normalize(env1)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space1, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} for env1 as action spaces "
                    f"but {self.action_space1} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy1 in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space1, spaces.Dict):
                raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy1}")

            if self.use_sde and not isinstance(self.action_space1, spaces.Box):
                raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space1, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space1.low, self.action_space1.high]))
                ), "Continuous action space must have a finite lower and upper bound"
                
        # Create and wrap the env if needed
        if env2 is not None:
            env2 = maybe_make_env(env2, self.verbose)
            env2 = self._wrap_env(env2, self.verbose, monitor_wrapper)
            self.observation_space2 = env2.observation_space
            self.action_space2 = env2.action_space
            self.env2 = env2

            # get VecNormalize object if needed
            self._vec_normalize_env2 = unwrap_vec_normalize(env2)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space2, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} for env2 as action spaces "
                    f"but {self.action_space2} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy2 in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space2, spaces.Dict):
                raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy2}")

            if self.use_sde and not isinstance(self.action_space2, spaces.Box):
                raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space2, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space2.low, self.action_space2.high]))
                ), "Continuous action space must have a finite lower and upper bound"

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = FloatSchedule(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[list[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _excluded_save_params(self) -> list[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy1",
            "policy2",
            "device",
            "env1",
            "env2",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env1",
            "_vec_normalize_env2",
            "_logger",
            "_custom_logger",
        ]

    def _get_policy_from_name(self, policy_name: str) -> type[BasePolicy]:
        """
        Get a policy class from its name representation.

        The goal here is to standardize policy naming, e.g.
        all algorithms can call upon "MlpPolicy" or "CnnPolicy",
        and they receive respective policies that work for them.

        :param policy_name: Alias of the policy
        :return: A policy class (type)
        """

        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy1.optimizer" would point to ``optimizer`` object of ``self.policy1``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy1", "policy2"]

        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> HBaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = HCallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, HBaseCallback):
            callback = HConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = HCallbackList([callback, HProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, HBaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs1 is None:
            assert self.env1 is not None
            self._last_obs1 = self.env1.reset()  # type: ignore[assignment]
            self._last_episode_starts1 = np.ones((self.env1.num_envs,), dtype=bool)

            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env1 is not None:
                self._last_original_obs1 = self._vec_normalize_env1.get_original_obs()
                
        if reset_num_timesteps or self._last_obs2 is None:
            assert self.env2 is not None
            self._last_obs2 = self.env2.reset()  # type: ignore[assignment]
            self._last_episode_starts2 = np.ones((self.env2.num_envs,), dtype=bool)
            
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env2 is not None:
                self._last_original_obs2 = self._vec_normalize_env2.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: list[dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env1(self) -> Optional[VecEnv]:
        """
        Returns the current environment 1 (can be None if not defined).

        :return: The current environment 1
        """
        return self.env1
    
    def get_env2(self) -> Optional[VecEnv]:
        """
        Returns the current environment 2 (can be None if not defined).

        :return: The current environment 2
        """
        return self.env2

    def get_vec_normalize_env1(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env 1
        if it exists.

        :return: The ``VecNormalize`` env 1.
        """
        return self._vec_normalize_env1

    def get_vec_normalize_env2(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: The ``VecNormalize`` env.
        """
        return self._vec_normalize_env2

    def set_env1(self, env1: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment 1.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space1
        - action_space1

        :param env1: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env1 = self._wrap_env(env1, self.verbose)
        assert env1.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env1.num_envs} != {self.n_envs}), whereas `set_env1` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env1)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env1, self.observation_space1, self.action_space1)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env1 = unwrap_vec_normalize(env1)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs1 = None

        self.n_envs = env1.num_envs
        self.env1 = env1
        
    def set_env2(self, env2: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment 2.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space2
        - action_space2

        :param env2: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env2 = self._wrap_env(env2, self.verbose)
        assert env2.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env2.num_envs} != {self.n_envs}), whereas `set_env2` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env2)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env2, self.observation_space2, self.action_space2)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env2 = unwrap_vec_normalize(env2)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs2 = None

        self.n_envs = env2.num_envs
        self.env2 = env2

    @abstractmethod
    def learn(
        self: SelfHBaseAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHBaseAlgorithm:
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
            Note: it is a lower bound, see `issue #1150 <https://github.com/DLR-RM/stable-baselines3/issues/1150>`_
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: for on-policy algos (e.g., PPO, A2C, ...) this is the number of
            training iterations (i.e., log_interval * n_steps * n_envs timesteps) before logging;
            for off-policy algos (e.g., TD3, SAC, ...) this is the number of episodes before
            logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """

    def predict1(
        self,
        observation1: Union[np.ndarray, dict[str, np.ndarray]],
        state1: Optional[tuple[np.ndarray, ...]] = None,
        episode_start1: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation1: the input observation
        :param state1: The last hidden states (can be None, used in recurrent policies)
        :param episode_start1: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy1.predict(observation1, state1, episode_start1, deterministic)


    def predict2(
        self,
        observation2: Union[np.ndarray, dict[str, np.ndarray]],
        state2: Optional[tuple[np.ndarray, ...]] = None,
        episode_start2: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation2: the input observation
        :param state2: The last hidden states (can be None, used in recurrent policies)
        :param episode_start2: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy2.predict(observation2, state2, episode_start2, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space1.seed(seed)
        
        self.action_space2.seed(seed)

        # self.env1 is always a VecEnv
        if self.env1 is not None:
            self.env1.seed(seed)
            
        # self.env2 is always a VecEnv
        if self.env2 is not None:
            self.env2.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(  # noqa: C901
        cls: type[SelfHBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env1: Optional[GymEnv] = None,
        env2: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfHBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env1: the new environment 1 to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param env2: the new environment 2 to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs1" in data:
            if "device" in data["policy_kwargs1"]:
                del data["policy_kwargs1"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs1"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs1"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs1" in kwargs and kwargs["policy_kwargs1"] != data["policy_kwargs1"]:
            raise ValueError(
                f"The specified policy 1 kwargs do not equal the stored policy 1 kwargs."
                f"Stored kwargs: {data['policy_kwargs1']}, specified kwargs: {kwargs['policy_kwargs1']}"
            )
        
        if "policy_kwargs2" in data:
            if "device" in data["policy_kwargs2"]:
                del data["policy_kwargs2"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs2"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs2"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs2" in kwargs and kwargs["policy_kwargs2"] != data["policy_kwargs2"]:
            raise ValueError(
                f"The specified policy 2 kwargs do not equal the stored policy 2 kwargs."
                f"Stored kwargs: {data['policy_kwargs2']}, specified kwargs: {kwargs['policy_kwargs2']}"
            )
            

        if "observation_space1" not in data or "action_space1" not in data:
            raise KeyError("The observation_space1 and action_space1 were not given, can't verify new environments")
        
        if "observation_space2" not in data or "action_space2" not in data:
            raise KeyError("The observation_space2 and action_space2 were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space1", "action_space1"}:
            data[key] = _convert_space(data[key])

        if env1 is not None:
            # Wrap first if needed
            env1 = cls._wrap_env(env1, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env1, data["observation_space1"], data["action_space1"])
            # Discard `_last_obs1`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs1"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env1.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env1" in data:
                env1 = data["env1"]
                
        for key in {"observation_space2", "action_space2"}:
            data[key] = _convert_space(data[key])

        if env2 is not None:
            # Wrap first if needed
            env2 = cls._wrap_env(env2, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env2, data["observation_space2"], data["action_space2"])
            # Discard `_last_obs2`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs2"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env2.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env2" in data:
                env2 = data["env2"]

        model = cls(
            policy1=data["policy_class1"],
            policy2=data["policy_class2"],
            env1=env1,
            env2=env2,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params1 = params["policy1.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            saved_optim_params2 = params["policy2.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]

            n_params_saved1 = len(saved_optim_params1)
            n_params_saved2 = len(saved_optim_params2)

            n_params1 = len(model.policy1.optimizer.param_groups[0]["params"])
            n_params2 = len(model.policy2.optimizer.param_groups[0]["params"])

            if n_params_saved1 == 2 * n_params1:
                # Truncate to include only online network params
                params["policy1.optimizer"]["param_groups"][0]["params"] = saved_optim_params1[:n_params1]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            if n_params_saved2 == 2 * n_params2:
                # Truncate to include only online network params
                params["policy2.optimizer"]["param_groups"][0]["params"] = saved_optim_params2[:n_params2]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy1.reset_noise()  # type: ignore[operator]
            model.policy2.reset_noise()  # type: ignore[operator]

        return model

    def get_parameters(self) -> dict[str, dict]:
        """
        Return the parameters of the (two) agents. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()
        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    def dump_logs(self) -> None:
        """
        Write log data. (Implemented by OffPolicyAlgorithm and OnPolicyAlgorithm)
        """
        raise NotImplementedError()

    def _dump_logs(self, *args) -> None:
        warnings.warn("algo._dump_logs() is deprecated in favor of algo.dump_logs(). It will be removed in SB3 v2.7.0")
        self.dump_logs(*args)