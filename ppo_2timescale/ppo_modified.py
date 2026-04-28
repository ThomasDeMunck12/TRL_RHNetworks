# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:43:54 2025

@author: thoma
"""

import warnings
from typing import Callable, Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from buffers_modified import HRolloutBuffer
from on_policy_algorithm_modified import HierarchicalOnPolicyAlgorithm
from policies_modified import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
#from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from callbacks_modified import HBaseCallback

MaybeCallback = Union[None, Callable, list["HBaseCallback"], "HBaseCallback"]
SelfHPPO = TypeVar("SelfHPPO", bound="Hierarchical_PPO")

class Hierarchical_PPO(HierarchicalOnPolicyAlgorithm):
    """
    Hierarchical Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation extends the PPO implementation from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy1: The policy model to use (MlpPolicy, CnnPolicy, ...) in env1
    :param policy2: The policy model to use (MlpPolicy, CnnPolicy, ...) in env2

    :param env1: The environment to learn from (if registered in Gym, can be str) in env1
    :param env2: The environment to learn from (if registered in Gym, can be str) in env2

    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for environment 2 per update. Equivalently, n_steps/p_interval is the number of steps to 
    run for environment 1 per update.
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
        
    :param p_interval: The number of steps to run for env2 per step of env1   
    :param batch_size: Minibatch size in env2, the batch size in env1 is batch_size/timing
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs1: additional arguments to be passed to the policy1 on creation. See :ref:`ppo_policies`
    :param policy_kwargs2: additional arguments to be passed to the policy2 on creation. See :ref:`ppo_policies`

    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy1: Union[str, type[ActorCriticPolicy]],
        policy2: Union[str, type[ActorCriticPolicy]],
        env1: Union[GymEnv, str],
        env2: Union[GymEnv, str],
        p_interval : int,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 3600,
        batch_size: int = 600,
        n_epochs: int = 10,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[HRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs1: Optional[dict[str, Any]] = None,
        policy_kwargs2: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy1,
            policy2,
            env1,
            env2,
            learning_rate=learning_rate,
            n_steps=n_steps,
            p_interval=p_interval,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs1=policy_kwargs1,
            policy_kwargs2=policy_kwargs2,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env1 is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env1.num_envs * self.n_steps // p_interval
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // (batch_size/p_interval)
            if buffer_size % (batch_size // p_interval) > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size // p_interval},"
                    f" but because the `HRolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env1.num_envs})"
                )
                
        if self.env2 is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env2.num_envs * self.n_steps 
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % (batch_size) > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `HRolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
                
        self.batch_size1 = int(batch_size / p_interval)
        self.batch_size2 = batch_size

        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy1.set_training_mode(True)
        self.policy2.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy1.optimizer)
        self._update_learning_rate(self.policy2.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses1 = []
        entropy_losses2 = []

        pg_losses1, value_losses1 = [], []
        pg_losses2, value_losses2 = [], []

        clip_fractions1 = []
        clip_fractions2 = []

        continue_training = True
        
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs1 = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get1(self.batch_size1):
                actions1 = rollout_data.actions
                if isinstance(self.action_space1, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions1 = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy1.evaluate_actions(rollout_data.observations, actions1)
                values = values.flatten()
                #print(rollout_data.observations)
                #print(actions1)
                #print(values)
                # Normalize advantage
                advantages1 = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages1) > 1:
                    advantages1 = (advantages1 - advantages1.mean()) / (advantages1.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_a = advantages1 * ratio
                policy_loss_b = advantages1 * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss1 = -th.min(policy_loss_a, policy_loss_b).mean()

                # Logging
                pg_losses1.append(policy_loss1.item())
                clip_fraction1 = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions1.append(clip_fraction1)

                values_pred = values
                
                # Value loss using the TD(gae_lambda) target
                value_loss1 = F.mse_loss(rollout_data.returns, values_pred)
                value_losses1.append(value_loss1.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss1 = -th.mean(-log_prob)
                else:
                    entropy_loss1 = -th.mean(entropy)

                entropy_losses1.append(entropy_loss1.item())

                loss1 = policy_loss1 + self.ent_coef * entropy_loss1 + self.vf_coef * value_loss1

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs1.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy1.optimizer.zero_grad()
                loss1.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy1.parameters(), self.max_grad_norm)
                self.policy1.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
            
        for epoch in range(self.n_epochs):
            approx_kl_divs2 = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get2(self.batch_size2):
                actions2 = rollout_data.actions
                if isinstance(self.action_space2, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions2 = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy2.evaluate_actions(rollout_data.observations, actions2)
                values = values.flatten()
                #print(rollout_data.observations)
                #print(actions2)
                #print(values)
                # Normalize advantage
                advantages2 = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages2) > 1:
                    advantages2 = (advantages2 - advantages2.mean()) / (advantages2.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_a = advantages2 * ratio
                policy_loss_b = advantages2 * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss2 = -th.min(policy_loss_a, policy_loss_b).mean()

                # Logging
                pg_losses2.append(policy_loss2.item())
                clip_fraction2 = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions2.append(clip_fraction2)

                # Value loss using the TD(gae_lambda) target
                values_pred = values
                
                value_loss2 = F.mse_loss(rollout_data.returns, values_pred)
                value_losses2.append(value_loss2.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss2 = -th.mean(-log_prob)
                else:
                    entropy_loss2 = -th.mean(entropy)

                entropy_losses2.append(entropy_loss2.item())

                loss2 = policy_loss2 + self.ent_coef * entropy_loss2 + self.vf_coef * value_loss2

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs2.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy2.optimizer.zero_grad()
                loss2.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy2.parameters(), self.max_grad_norm)
                self.policy2.optimizer.step()

            if not continue_training:
                break
            
        explained_var1 = explained_variance(self.rollout_buffer.values1.flatten(), self.rollout_buffer.returns1.flatten())
        explained_var2 = explained_variance(self.rollout_buffer.values2.flatten(), self.rollout_buffer.returns2.flatten())

        # Logs
        self.logger.record("train/entropy_loss1", np.mean(entropy_losses1))
        self.logger.record("train/entropy_loss2", np.mean(entropy_losses2))

        self.logger.record("train/policy_gradient_loss1", np.mean(pg_losses1))
        self.logger.record("train/policy_gradient_loss2", np.mean(pg_losses2))

        self.logger.record("train/value_loss1", np.mean(value_losses1))
        self.logger.record("train/value_loss2", np.mean(value_losses2))

        self.logger.record("train/approx_kl1", np.mean(approx_kl_divs1))        
        self.logger.record("train/approx_kl2", np.mean(approx_kl_divs2))

        self.logger.record("train/clip_fraction1", np.mean(clip_fractions1))
        self.logger.record("train/clip_fraction2", np.mean(clip_fractions2))

        self.logger.record("train/loss1", loss1.item())
        self.logger.record("train/loss2", loss2.item())

        self.logger.record("train/explained_variance1", explained_var1)
        self.logger.record("train/explained_variance2", explained_var2)

        if hasattr(self.policy1, "log_std"):
            self.logger.record("train/std1", th.exp(self.policy1.log_std).mean().item())
            
        if hasattr(self.policy2, "log_std"):
            self.logger.record("train/std2", th.exp(self.policy2.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfHPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )