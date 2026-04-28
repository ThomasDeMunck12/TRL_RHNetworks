# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:43:28 2026

@author: thoma
"""

print('Start task number: ', parameter)

import numpy as np 

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor

from gymnasium import spaces

import torch as T
from torch.utils.data.dataset import Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def pretrain_vf(
    student,
    batch_size=64,
    epochs=10,
    learning_rate=1e-4,
    n_rollout_steps=2400,
    n_iterations = 1e3):

    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    loss_fn = nn.MSELoss()

    policy_student = student.policy.to(device)
    rollout_buffer_student = student.rollout_buffer
    env_student = student.env

    def collect_rollouts(env_student, rollout_buffer_student, policy_student, n_rollout_steps, device):

        policy_student.set_training_mode(True)

        n_steps = 0
        rollout_buffer_student.reset()
        student._last_obs = env_student.reset()
        student._last_episode_starts = np.ones((env_student.num_envs,), dtype=bool)

        while n_steps < n_rollout_steps:
            n_steps += 1

            with T.no_grad():
                obs_tensor = obs_as_tensor(student._last_obs, device)
                actions, values, log_probs = policy_student(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(env_student.action_space, spaces.Box):
                clipped_actions = np.clip(actions, env_student.action_space.low, env_student.action_space.high)

            new_obs, rewards, dones, infos = env_student.step(clipped_actions)

            rollout_buffer_student.add(
                student._last_obs,
                actions,
                rewards,
                student._last_episode_starts,
                values,
                log_probs)

            student._last_obs = new_obs
            student._last_episode_starts = dones

        with T.no_grad():
            values = policy_student.predict_values(obs_as_tensor(new_obs, device))  # type: ignore[arg-type]

        rollout_buffer_student.compute_returns_and_advantage(last_values=values, dones=dones)
        return True

    def train(policy_student, rollout_buffer_student, optimizer, epoch):
      policy_student.set_training_mode(True)
      losses = []

      for rollout_data in rollout_buffer_student.get(batch_size):
          _, values, _ = policy_student(rollout_data.observations)
          values = values.flatten()
          loss = loss_fn(values, rollout_data.returns)

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          if epoch == epochs:
            losses.append(loss.item())            

    #Define the optimizer

    critic_layers = [9,10,11,12,13,14,15,16,19,20]
    optimizer = optim.Adam([list(policy_student.parameters())[i] for i in critic_layers], lr=learning_rate)

    # Train the value function.
    n = 0
    while n < n_iterations:
      continue_training = collect_rollouts(env_student, rollout_buffer_student, policy_student, n_rollout_steps, device)
      n += 1
      for epoch in range(1, epochs + 1):
          train(policy_student, rollout_buffer_student, optimizer, epoch)

    # Implement back the trained policy
    pppo.policy = policy_student
   