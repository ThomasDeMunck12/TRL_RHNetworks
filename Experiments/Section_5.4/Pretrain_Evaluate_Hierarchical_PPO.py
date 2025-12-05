# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:31:47 2025

@author: ThomasDM
"""

#import os 
import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "hierarchical_rl"))


from stable_baselines3.common.monitor import Monitor
from ppo_modified import Hierarchical_PPO
from callbacks_modified import HEvalCallback
from evaluation_modified import hevaluate_policy
from stable_baselines3.common.utils import obs_as_tensor

#from gymnasium import spaces

import torch as th
from torch.utils.data.dataset import Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import time
from env1_february import Env1_F
from env2_february import Env2_F

from env1_march import Env1_M
from env2_march import Env2_M

from gymnasium import spaces

#Base parameters

F = 600 #fleet size
F_max = 1500 #max fleet size
N = 20 #number of locations
H = 72 #number of decision times

beta0 = -2.5
beta1 = 3.0
beta2 = -1.5
temp = 1.0

C_q = 0.4
C_r = 0.2

n_adjustments = 1
p_interval = 2


C_q = 0.4
C_r = 0.2

# Load data

Lambda = np.load("../data/arrival_rates_20loc_february.npy")
Lambda_feb = np.mean(Lambda, axis=3)

expected_time = np.load("../data/average_trip_durations_20loc.npy")
tau = 1/expected_time

distance = np.load("../data/average_distances_20loc.npy")

mu = np.load("../data/entry_rates_20loc.npy")

#Initial state

periods_window = 12
trips_before_episode = np.round(Lambda_feb[:, :, 0:periods_window]*0.5)
Prob_Trip_Completed = 1 - np.exp(-tau)
kappa = np.zeros([N, N, periods_window])
for i in range(N):
    for j in range(N):
        for t in range(periods_window):
            kappa[i, j, t] = (1.0-Prob_Trip_Completed[i,j])**(periods_window-t)
y_init = trips_before_episode * kappa
y_init = np.sum(y_init, axis=2)
y_init = np.round(y_init)
y_init = y_init.astype(int)

if np.sum(y_init) > 3/4*F:
    ratio = 3*F/(4*np.sum(y_init))
    y_init = y_init * ratio
    y_init = np.round(y_init)
    y_init = y_init.astype(int)

trips_beginning_episode = Lambda_feb[:, :, 0:periods_window]
distribution_per_loc = np.sum(trips_beginning_episode, axis=(1,2))/np.sum(trips_beginning_episode)
x_init = F - np.sum(y_init)

x_init = x_init*distribution_per_loc
x_init = np.round(x_init)
for i in range(N):
    if x_init[i] < 0:
        x_init[i] = 0

x_init = x_init.astype(int)

p_init = np.zeros([N, N])

#February models 

env1_feb = Env1_F(F = F_max, N = N, H = H, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5
env2_feb = Env2_F(F = F_max, Lambda=Lambda, tau = tau, distance = distance, mu = mu, N = N, H = H, p_interval = p_interval,
           C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2 = beta2, temp = temp, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

#Hyperparameters

n_steps = 5400
target_kl = 1.5
gamma = 1.0

learning_rate = 0.00001
batch_size = 900
gae = 1.0
epochs = 10
neural_net = 128

policy_kwargs1 = dict(activation_fn=nn.ReLU,
                       net_arch=dict(pi=[neural_net, neural_net, neural_net, neural_net], vf=[neural_net, neural_net, neural_net, neural_net])) #network architecture

policy_kwargs2 = dict(activation_fn=nn.ReLU,
                       net_arch=dict(pi=[neural_net, neural_net, neural_net, neural_net], vf=[neural_net, neural_net, neural_net, neural_net])) #network architecture

hpppo = Hierarchical_PPO("MultiInputPolicy", "MultiInputPolicy", env1_feb, env2_feb,
           learning_rate = learning_rate, batch_size = batch_size,  gae_lambda = gae, n_epochs = epochs,
           n_steps = n_steps, p_interval = p_interval, target_kl = target_kl, gamma = gamma, policy_kwargs1 = policy_kwargs1, policy_kwargs2 = policy_kwargs2, verbose = 1) #pppo

n_timesteps = 5400000
eval_timesteps = 108000
eval_episodes = 50

eval_callback = HEvalCallback(env1_feb, env2_feb, eval_freq=eval_timesteps, p_interval=2,
                             log_path="./HPPO_evaluations/test/",
                             n_eval_episodes = eval_episodes, deterministic=True, verbose = 1.0)

#Actor pretraining
import torch as T

#Pretrain first actor network

def pretrain_actor1(
    hpppo,
    train_expert_dataset1,
    test_expert_dataset1,
    batch_size=360,
    epochs=100,
    scheduler_gamma=0.995,
    log_interval=54,
    learning_rate=1.0e-4):
    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")

    def train(policy1_pt, device, train_loader, optimizer):
        policy1_pt.train()

        for batch_idx, (obs_dict, target) in enumerate(train_loader):
            data = {
                "p": obs_dict["p"].to(device, dtype=T.float32),
                "xy": obs_dict["xy"].to(device, dtype=T.float32),
                "t": obs_dict["t"].to(device, dtype=T.float32),

                }
            target = target.to(device)
            optimizer.zero_grad()

            action, _, _ = policy1_pt(data)  #PPO policy outputs actions, values, log_prob
            action_prediction = action.double()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data["p"]),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(policy1_pt, device, test_loader):
        policy1_pt.eval()
        test_loss = 0
        with T.no_grad():
            for (obs_dict, target) in test_loader:
                data = {
                    "p": obs_dict["p"].to(device, dtype=T.float32),
                    "xy": obs_dict["xyt"].to(device, dtype=T.float32),
                    "t": obs_dict["t"].to(device, dtype=T.float32),
                    }
                target = target.to(device)

                action, _, _ = policy1_pt(data)
                action_prediction = action.double()

                test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.MSELoss()

    # Extract initial policy
    policy1_pt = hpppo.policy1.to(device)

    # Load dataset
    train_loader = T.utils.data.DataLoader(
        dataset=train_expert_dataset1, batch_size=batch_size, shuffle=True, **kwargs
    )

    #test_loader = T.utils.data.DataLoader(
    #    dataset=test_expert_dataset1,
    #    batch_size=batch_size,
    #    shuffle=True,
    #    **kwargs,
    #)

    # Define an optimizer and a learning rate schedule.
    #optimizer = optim.Adadelta(policy1_pt.parameters(), lr=learning_rate)
    optimizer = optim.Adam(policy1_pt.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Train and test the policy model.
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(policy1_pt, device, train_loader, optimizer)
        #test(policy1_pt, device, test_loader)

        scheduler.step()

    # Implement the trained policy network back into the RL student agent
    hpppo.policy1 = policy1_pt

#Pretrain second actor network.

def pretrain_actor2(
    hpppo,
    train_expert_dataset2,
    test_expert_dataset2,
    batch_size=720,
    epochs=100,
    scheduler_gamma=0.995,
    log_interval=54,

    learning_rate=1.0e-4):
    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")

    def train(policy2_pt, device, train_loader, optimizer):
        policy2_pt.train()

        for batch_idx, (obs_dict, target) in enumerate(train_loader):
            data = {
                "p": obs_dict["p"].to(device, dtype=T.float32),
                "xy": obs_dict["xy"].to(device, dtype=T.float32),
                "t": obs_dict["t"].to(device, dtype=T.float32),
                }

            target = target.to(device)

            optimizer.zero_grad()

            action, _, _ = policy2_pt(data)  #PPO policy outputs actions, values, log_prob
            action_prediction = action.double()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data["p"]),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(policy2_pt, device, test_loader):
        policy2_pt.eval()
        test_loss = 0
        with T.no_grad():
            for (obs_dict, target) in test_loader:
                data = {
                    "p": obs_dict["p"].to(device, dtype=T.float32),
                    "xy": obs_dict["xy"].to(device, dtype=T.float32),
                    "t": obs_dict["t"].to(device, dtype=T.float32),
                    }

                target = target.to(device)

                action, _, _ = policy2_pt(data)
                action_prediction = action.double()

                test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.MSELoss()

    # Extract initial policy
    policy2_pt = hpppo.policy2.to(device)

    # Load dataset
    train_loader = T.utils.data.DataLoader(
        dataset=train_expert_dataset2, batch_size=batch_size, shuffle=True, **kwargs
    )

    #test_loader = T.utils.data.DataLoader(
    #    dataset=test_expert_dataset2,
    #   batch_size=batch_size,
    #    shuffle=True,
    #    **kwargs,
    #)

    # Define an optimizer and a learning rate schedule.
    optimizer = optim.Adam(policy2_pt.parameters(), lr=learning_rate)
    #optimizer = optim.Adadelta(policy2_pt.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Train and test the policy model.
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(policy2_pt, device, train_loader, optimizer)
        #test(policy2_pt, device, test_loader)

        scheduler.step()

    # Implement the trained policy network back into the RL student agent
    hpppo.policy2 = policy2_pt

expert_obs1 = np.load("./expert_observations1.npy", allow_pickle=True)
expert_actions1 = np.load("./expert_actions1.npy", allow_pickle=True)

expert_obs2 = np.load("./expert_observations2.npy", allow_pickle=True)
expert_actions2 = np.load("./expert_actions2.npy", allow_pickle=True)

class ExpertDataSet(Dataset):
    def __init__(self, expert_obs, expert_actions):
        self.obs = expert_obs
        self.actions = expert_actions

    def __getitem__(self, index):
        obs, action = self.obs[index], self.actions[index]

        return obs, action

    def __len__(self):
        return len(self.obs)

expert_dataset1 = ExpertDataSet(expert_obs1, expert_actions1)
expert_dataset2 = ExpertDataSet(expert_obs2, expert_actions2)

train_size1 = int(1.0 * len(expert_dataset1))
test_size1 = len(expert_dataset1) - train_size1

train_size2 = int(1.0 * len(expert_dataset2))
test_size2 = len(expert_dataset2) - train_size2

train_expert_dataset1, test_expert_dataset1 = random_split(
    expert_dataset1, [train_size1, test_size1])

train_expert_dataset2, test_expert_dataset2 = random_split(
    expert_dataset2, [train_size2, test_size2])

Start_time = time.time()
pretrain_actor1(
    hpppo,
    train_expert_dataset1,
    test_expert_dataset1,
    batch_size=270,
    epochs = 80,
    scheduler_gamma = 0.995,
    learning_rate=0.0005
    #learning_rate=1.0, #if Adadelta
    )

pretrain_actor2(
    hpppo,
    train_expert_dataset2,
    test_expert_dataset2,
    batch_size=540,
    epochs = 80,
    scheduler_gamma = 0.995,
    learning_rate=0.0005
    #learning_rate=1.0, #if Adadelta
    )
print(f"Time of policy reproduction: {(time.time()-Start_time)/60} minutes")

#print(f"Time of policy reproduction: {(time.time()-Start_time)/60} minutes")
mean_reward, std_reward = hevaluate_policy(hpppo, env1_feb, env2_feb, p_interval=2, n_eval_episodes=100)
print(f"Mean reward = {mean_reward} +/- {1.96 * std_reward / np.sqrt(100)}")

#Pretrain critic 

def pretrain_critic(
    hpppo,
    batch_size = 720,
    epochs=10,
    p_interval=2,
    learning_rate = 0.02,
    n_rollout_steps = 3600,
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
    
pretrain_critic(
    hpppo,
    batch_size=540,
    epochs = 10,
    learning_rate = 0.001,
    n_rollout_steps = 5400,
    n_iterations = 100,
    p_interval=2
    )

hpppo.learn(total_timesteps = n_timesteps, callback = eval_callback)

from evaluation_modified import hevaluate_policy
#print(f"Time of policy reproduction: {(time.time()-Start_time)/60} minutes")
mean_reward, std_reward = hevaluate_policy(hpppo, env1_feb, env2_feb, p_interval=2, n_eval_episodes=300)
print(f"Mean reward = {mean_reward} +/- {1.96 * std_reward / np.sqrt(300)}")

#Evaluation

Lambda_march = np.load("arrival_rates_20loc_march.npy")
Lambda_1 = Lambda_march[:, :, :, 0]
Lambda_2 = Lambda_march[:, :, :, 1]
Lambda_3 = Lambda_march[:, :, :, 2]
Lambda_4 = Lambda_march[:, :, :, 3]
Lambda_5 = Lambda_march[:, :, :, 4]
Lambda_6 = Lambda_march[:, :, :, 5]
Lambda_7 = Lambda_march[:, :, :, 6]
Lambda_8 = Lambda_march[:, :, :, 7]
Lambda_9 = Lambda_march[:, :, :, 8]
Lambda_10 = Lambda_march[:, :, :, 9]
Lambda_11 = Lambda_march[:, :, :, 10]
Lambda_12 = Lambda_march[:, :, :, 11]
Lambda_13 = Lambda_march[:, :, :, 12]
Lambda_14 = Lambda_march[:, :, :, 13]
Lambda_15 = Lambda_march[:, :, :, 14]
Lambda_16 = Lambda_march[:, :, :, 15]
Lambda_17 = Lambda_march[:, :, :, 16]
Lambda_18 = Lambda_march[:, :, :, 17]
Lambda_19 = Lambda_march[:, :, :, 18]
Lambda_20 = Lambda_march[:, :, :, 19]

#Define March environments

env1_1 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_1 = Env2_M(F=F_max, Lambda=Lambda_1[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_2 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_2 = Env2_M(F=F_max, Lambda=Lambda_2[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_3 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_3 = Env2_M(F=F_max, Lambda=Lambda_3[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_4 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_4 = Env2_M(F=F_max, Lambda=Lambda_4[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_5 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_5 = Env2_M(F=F_max, Lambda=Lambda_5[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_6 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_6 = Env2_M(F=F_max, Lambda=Lambda_6[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_7 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_7 = Env2_M(F=F_max, Lambda=Lambda_7[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_8 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_8 = Env2_M(F=F_max, Lambda=Lambda_8[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_9 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_9 = Env2_M(F=F_max, Lambda=Lambda_9[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_10 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_10 = Env2_M(F=F_max, Lambda=Lambda_10[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_11 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_11 = Env2_M(F=F_max, Lambda=Lambda_11[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_12 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_12 = Env2_M(F=F_max, Lambda=Lambda_12[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_13 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_13 = Env2_M(F=F_max, Lambda=Lambda_13[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_14 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_14 = Env2_M(F=F_max, Lambda=Lambda_14[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_15 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_15 = Env2_M(F=F_max, Lambda=Lambda_15[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_16 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_16 = Env2_M(F=F_max, Lambda=Lambda_16[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_17 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_17 = Env2_M(F=F_max, Lambda=Lambda_17[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_18 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_18 = Env2_M(F=F_max, Lambda=Lambda_18[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_19 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_19 = Env2_M(F=F_max, Lambda=Lambda_19[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env1_20 = Env1_M(F=F_max, N=N, H=H, p_0 = p_init, x_0 = x_init, y_0 = y_init)
env2_20 = Env2_M(F=F_max, Lambda=Lambda_20[:, :, 0 : H], tau=tau, distance=distance, mu=mu, N=N, H=H, C_q = 0.4, C_r = 0.2, beta0 = -2.5, beta1 = 3.0, beta2 = -1.5, temp = 1.0, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

environments1 = [env1_1, env1_2, env1_3, env1_4, env1_5, env1_6, env1_7, env1_8, env1_9, env1_10,
             env1_11, env1_12, env1_13, env1_14, env1_15, env1_16, env1_17, env1_18, env1_19, env1_20]

environments2 = [env2_1, env2_2, env2_3, env2_4, env2_5, env2_6, env2_7, env2_8, env2_9, env2_10,
             env2_11, env2_12, env2_13, env2_14, env2_15, env2_16, env2_17, env2_18, env2_19, env2_20]

def sync_env_state_eval(dst_env, obs_dict):
    dst_env.SetState(obs_dict)

for i in range(20):
  k = float(i)
  env1_sim = environments1[i]
  env2_sim = environments2[i]

  seed = 10
  n_episodes = 50

  x_m = np.zeros([N, H, n_episodes]) #drivers available
  y_m = np.zeros([N, N, H, n_episodes]) #drivers in transit
  p_m = np.zeros([N, N, H, n_episodes]) #prices
  d_m = np.zeros([N, N, H, n_episodes]) #dispatched drivers
  c_m = np.zeros([N, N, H, n_episodes]) #customer requests
  lambda_m = np.zeros([N, N, H, n_episodes]) #arrival rates
  f_m = np.zeros([N, N, H, n_episodes]) #rejected customers
  w_m = np.zeros([N, N, H, n_episodes]) #self-relocating drivers
  r_m = np.zeros([N, N, H, n_episodes]) #repositioned drivers
  z_m = np.zeros([N, N, H, n_episodes]) #drivers completing service
  num_drivers = np.zeros([N, H, n_episodes]) #number of drivers available in each locat
  Revenue_v = np.zeros([H, n_episodes]) #revenues
  LostSales_v = np.zeros([H, n_episodes]) #lost sales costs
  Repositioning_v = np.zeros([H, n_episodes]) #repositioning costs
  Profit_v = np.zeros([H, n_episodes]) #profits

  scores = []

  for j in range(n_episodes):
    observation1 = env1_sim.reset(seed=seed+i)
    observation1 = observation1[0]

    observation2 = env2_sim.reset(seed=seed+i)
    observation2 = observation2[0]

    done = False
    idx = 0
    score = 0

    while not done:
      if (idx % p_interval) == 0:
        action1, _ = hpppo.predict1(observation1, deterministic=True)
        observation1_, _, _, _, _ = env1_sim.step(action1)
        sync_env_state_eval(env2_sim, observation1_)
        observation2 = observation1_

      x_m[:, idx, j] = env2_sim.x

      y_m[:, :, idx, j] = env2_sim.y
      p_m[:, :, idx, j] = env2_sim.p

      action2, _ = hpppo.predict2(observation2, deterministic=True)
      observation2_, reward, done, _, info = env2_sim.step(action2)

      r_m[:, :, idx, j] = info["Repositioned drivers"]
      w_m[:, :, idx, j] = info["Self-relocating drivers"]
      c_m[:, :, idx, j] = info["Realized requests"]
      lambda_m[:, :, idx, j] = info["Arrival rates"]
      d_m[:, :, idx, j] = info["Dispatched drivers"]
      z_m[:, :, idx, j] = info["Drivers completing service"]
      f_m[:, :, idx, j] = info["Requests unsatisfied"]

      Revenue_v[idx, j] = info["Revenue"]
      Repositioning_v[idx, j] = info["Repositioning costs"]
      LostSales_v[idx, j] = info["Lost sales costs"]
      Profit_v[idx, j] = info["Revenue"]-info["Lost sales costs"]-info["Repositioning costs"]

      score += reward
      observation2 = observation2_
      idx = idx + 1
      if (idx % p_interval) == 0:
            observation1 = observation2
            sync_env_state_eval(env1_sim, observation2)
    scores.append(score)
  print(np.mean(scores))
  np.save(f"./HPPPO/instance_{k+1}/scores.npy", scores)
  np.save(f"./HPPPO/instance_{k+1}/x_PPPO.npy", x_m)
  np.save(f"./HPPPO/instance_{k+1}/y_PPPO.npy", y_m)
  np.save(f"./HPPPO/instance_{k+1}/p_PPPO.npy", p_m)
  np.save(f"./HPPPO/instance_{k+1}/d_PPPO.npy", d_m)
  np.save(f"./HPPPO/instance_{k+1}/c_PPPO.npy", c_m)
  np.save(f"./HPPPO/instance_{k+1}/lambda_PPPO.npy", lambda_m)
  np.save(f"./HPPPO/instance_{k+1}/f_PPPO.npy", f_m)
  np.save(f"./HPPPO/instance_{k+1}/w_PPPO.npy", w_m)
  np.save(f"./HPPPO/instance_{k+1}/r_PPPO.npy", r_m)
  np.save(f"./HPPPO/instance_{k+1}/z_PPPO.npy", z_m)
  np.save(f"./HPPPO/instance_{k+1}/Revenue_PPPO.npy", Revenue_v)
  np.save(f"./HPPPO/instance_{k+1}/LostSales_PPPO.npy", LostSales_v)
  np.save(f"./HPPPO/instance_{k+1}/Repositioning_PPPO.npy", Repositioning_v)
  np.save(f"./HPPPO/instance_{k+1}/Profit_PPPO.npy", Profit_v)

