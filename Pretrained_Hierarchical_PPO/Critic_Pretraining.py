# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:16:13 2025

@author: thoma
"""

from stable_baselines3.common.utils import obs_as_tensor
import torch as T
import torch.nn as nn
import numpy as np
from gymnasium import spaces
import torch.optim as optim

def pretrain_critic(
    hpppo,
    batch_size = 540,
    epochs=10,
    p_interval=2,
    learning_rate = 0.001,
    n_rollout_steps = 5400,
    n_iterations = 100):

    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    policy1_pt = hpppo.policy1.to(device)
    policy2_pt = hpppo.policy2.to(device)

    rollout_buffer_pt = hpppo.rollout_buffer
    env1_pt = hpppo.env1
    env2_pt = hpppo.env2
    batch_size1 = int(batch_size / p_interval)
    batch_size2 = batch_size
    mse_loss1 = nn.MSELoss()
    mse_loss2 = nn.MSELoss()

    def collect_rollouts(env1_pt, env2_pt, rollout_buffer_pt, policy1_pt, policy2_pt, n_rollout_steps, p_interval, device):

        policy1_pt.set_training_mode(False)
        policy2_pt.set_training_mode(False)

        n_step_total = 0
        rollout_buffer_pt.reset()

        hpppo._last_obs1 = env1_pt.reset()
        hpppo._last_obs2 = env2_pt.reset()

        hpppo._last_episode_starts1 = np.ones((env1_pt.num_envs,), dtype=bool)
        hpppo._last_episode_starts2 = np.ones((env2_pt.num_envs,), dtype=bool)

        while n_step_total < n_rollout_steps:
            if n_step_total % p_interval == 0:
                with T.no_grad():
                    obs_tensor = obs_as_tensor(hpppo._last_obs1, device)
                    actions, values, log_probs = policy1_pt(obs_tensor)
                    actions = actions.cpu().numpy()
                    clipped_actions = actions

                if isinstance(hpppo.action_space1, spaces.Box):
                    clipped_actions = np.clip(actions, hpppo.action_space1.low, hpppo.action_space1.high)
                else:
                    clipped_actions = actions

                new_obs, rewards, dones, infos = env1_pt.step(clipped_actions)

                for idx, done in enumerate(dones):
                    if (
                            done
                            and infos[idx].get("terminal_observation") is not None
                            and infos[idx].get("TimeLimit.truncated", False)
                            ):
                        terminal_obs = policy1_pt.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with T.no_grad():
                            terminal_value = policy1_pt.predict_values(terminal_obs)[0]
                        rewards[idx] += hpppo.gamma * terminal_value

                rollout_buffer_pt.add1(
                    hpppo._last_obs1,
                    actions,
                    rewards,
                    hpppo._last_episode_starts1,
                    values,
                    log_probs)
                #print("1: ", hpppo._last_episode_starts1)

                hpppo._last_obs1 = new_obs
                hpppo._last_episode_starts1 = dones
                hpppo._last_obs2, hpppo._last_episode_starts2 = hpppo.sync_env_state(env1_pt, env2_pt, new_obs, hpppo._last_obs1, hpppo._last_episode_starts1)
            with T.no_grad():
                obs_tensor = obs_as_tensor(hpppo._last_obs2, device)
                actions, values, log_probs = policy2_pt(obs_tensor)
                actions = actions.cpu().numpy()
                clipped_actions = actions

            if isinstance(hpppo.action_space2, spaces.Box):
                clipped_actions = np.clip(actions, hpppo.action_space2.low, hpppo.action_space2.high)

            new_obs, rewards, dones, infos = env2_pt.step(clipped_actions)


            n_step_total += 1

            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                        ):
                    terminal_obs = policy2_pt.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with T.no_grad():
                        terminal_value = policy2_pt.predict_values(terminal_obs)[0]
                    rewards[idx] += hpppo.gamma * terminal_value

            rollout_buffer_pt.add2(
                hpppo._last_obs2,
                actions,
                rewards,
                hpppo._last_episode_starts2,
                values,
                log_probs)
            #print("2: ", hpppo._last_episode_starts2)

            hpppo._last_obs2 = new_obs
            hpppo._last_episode_starts2 = dones

            if n_step_total % p_interval == 0:
                hpppo._last_obs1, hpppo._last_episode_starts1 = hpppo.sync_env_state(env2_pt, env1_pt, new_obs, hpppo._last_obs2, hpppo._last_episode_starts2)

        with T.no_grad():
            values2 = policy2_pt.predict_values(obs_as_tensor(new_obs, device))  # type: ignore[arg-type]
        rollout_buffer_pt.compute_returns_and_advantage(last_values=values2, dones=dones)
        return True

    def train(policy1_pt, policy2_pt, rollout_buffer_pt, optimizer1_pt, optimizer2_pt, epochs):
      policy1_pt.set_training_mode(True)
      policy2_pt.set_training_mode(True)

      value_losses1, value_losses2 = [], []
      for epoch in range(epochs):
          for rollout_data in rollout_buffer_pt.get1(batch_size1):
              actions1 = rollout_data.actions
              values, _, _ = policy1_pt.evaluate_actions(rollout_data.observations, actions1)
              #_, values, _ = policy1_pt(rollout_data.observations)
              values = values.flatten()

              #returns = rollout_data.returns.unsqueeze(1)    # shape: [360, 1]
              loss1 = mse_loss1(values, rollout_data.returns)
              optimizer1_pt.zero_grad()
              loss1.backward()
              optimizer1_pt.step()
              if epoch == epochs-1:
                value_losses1.append(loss1.item())
      print("L1: ", np.mean(value_losses1))

      for epoch in range(epochs):
          for rollout_data in rollout_buffer_pt.get2(batch_size2):
              actions2 = rollout_data.actions
              values, _, _ = policy2_pt.evaluate_actions(rollout_data.observations, actions2)
              values = values.flatten()
              #returns = rollout_data.returns.unsqueeze(1)

              loss2 = mse_loss2(values, rollout_data.returns)
              optimizer2_pt.zero_grad()
              loss2.backward()
              optimizer2_pt.step()
              if epoch == epochs-1:
                #print(rollout_data.returns)
                value_losses2.append(loss2.item())
      print("L2: ", np.mean(value_losses2))

    #Define the optimizer and select only critic parameters

    critic_layers1 = list(hpppo.policy1.mlp_extractor.value_net.parameters())
    critic_output1 = list(hpppo.policy1.value_net.parameters())

    all_critic_params1 = critic_layers1 + critic_output1
    optimizer1_pt = optim.Adam(all_critic_params1, lr=learning_rate)

    critic_layers2 = list(hpppo.policy2.mlp_extractor.value_net.parameters())
    critic_output2 = list(hpppo.policy2.value_net.parameters())

    all_critic_params2 = critic_layers2 + critic_output2
    optimizer2_pt = optim.Adam(all_critic_params2, lr=learning_rate)

    # Train the value function.
    n = 0
    while n < n_iterations:
      _ = collect_rollouts(env1_pt, env2_pt, rollout_buffer_pt, policy1_pt, policy2_pt, n_rollout_steps, p_interval, device)

      #print(rollout_buffer_pt.returns2)
      n += 1
      train(policy1_pt, policy2_pt, rollout_buffer_pt, optimizer1_pt, optimizer2_pt, epochs)

    # Implement back the trained policy
    hpppo.policy1 = policy1_pt
    hpppo.policy2 = policy2_pt