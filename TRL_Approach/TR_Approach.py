# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:25:46 2023

@author: ThomasDM
"""

import sys

#parameter = float(sys.argv[1]) #for Slurm cluster
parameter = 3.0 #default instance

print('Start task number: ', parameter)

import numpy as np 
import time 
from RideHailing_Env import RideHailingEnv

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

#Base parameters

F = 900 #fleet size
N = 8 #number of locations
H = 36 #number of decision times

traffic_const = 1.0 #constant adjusting transit times and arrival rates

beta0 = -0.15
beta1 = 8.0
beta2 = -1.5
temp = 1.0

C_q = 0.4
C_r = 0.2

#Change the parameters according to the instance 

if parameter <= 5.0: 
    param = parameter 
    traffic_const = (0.25+param*0.25) 
    
elif (parameter > 5.0) and (parameter <= 10.0):
    param = parameter - 5.0
    F = (0.25+param*0.25) * F 

elif (parameter > 10.0) and (parameter <= 15.0):
    param = parameter - 10.0
    temp = 0.25 * 2 ** (param-1.0)

elif (parameter > 15.0) and (parameter <= 20.0):
    param = parameter - 15.0
    
    if (param == 1):
        beta0 = -0.15
        beta1 = 8.0
        beta2 = -2.0
        
    elif (param == 2):
        beta0 = -0.15
        beta1 = 8.0
        beta2 = -1.0
        
    elif (param == 3):
        beta0 = -0.0
        beta1 = 8.0
        beta2 = -1.5
        
    elif (param == 4):
        beta0 = 0.3 #1.75
        beta1 = 0.0
        beta2 = -1.5
        
    elif (param == 5):
        beta0 = -5.1 #-7.5
        beta1 = 8.0
        beta2 = 0.0
        
elif (parameter > 20.0) and (parameter <= 25.0):
    param = parameter - 20.0
    if param <= 3:
        C_q = 0.2 * (param-1)
    elif param == 4:
        C_q = 0.8
    elif param == 5:
        C_q = 1.2
elif (parameter > 25.0) and (parameter <= 30.0):
    param = parameter - 25.0
    C_r = 0.1 * (param-1)
    
# Load data

Lambda = np.load("../Data/arrival_rates_8loc.npy")
Lambda_day = np.copy(Lambda)
Lambda = Lambda[:, :, 84:]
Lambda = Lambda * traffic_const

average_time = np.load("../Data/average_trip_durations_8loc.npy")
tau = 1/average_time
tau_init = np.copy(tau)
expected_time = 1/tau_init
expected_time = 1/tau_init

distance = np.load("../Data/average_distances_8loc.npy")

#Initial state

periods_window = 12
trips_before_episode = np.round(Lambda_day[:, :, 84:84+periods_window]*0.5)
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


trips_beginning_episode = Lambda_day[:, :, 84:84+periods_window]
distribution_per_loc = np.sum(trips_beginning_episode, axis=(1,2))/np.sum(trips_beginning_episode)
x_init = F - np.sum(y_init)

x_init = x_init*distribution_per_loc
x_init = np.round(x_init)
for i in range(N):
    if x_init[i] < 0:
        x_init[i] = 0
    

x_init = x_init.astype(int)

#Define environment

env = RideHailingEnv(F = F, Lambda = Lambda[:, :, 0:H], tau = tau, distance = distance,
                     N = N, H = H,
                     C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2=beta2, temp = temp,
                     x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env = Monitor(env)

#Define PPO 

#Parameters 
n_steps = 2400
target_kl = 1.5
gamma = 1.0

learning_rate = 5*10e-8
batch_size = 800 
gae = 1.00
epochs = 30 
ent_coef = 0.000
neural_net = 128 

policy_kwargs = dict(activation_fn=T.nn.Tanh,
                       net_arch=dict(pi=[neural_net, neural_net, neural_net, neural_net], vf=[neural_net, neural_net, neural_net, neural_net])) #network architecture

pppo = PPO("MlpPolicy", env,
           learning_rate = learning_rate, batch_size = batch_size,  gae_lambda = gae, n_epochs = epochs,
           n_steps = n_steps, target_kl = target_kl, ent_coef = ent_coef, gamma = gamma, policy_kwargs = policy_kwargs, verbose = 0) #pppo

#Define callbacks

eval_timesteps = 1e5
eval_episodes = 300

eval_callback = EvalCallback(env, eval_freq=eval_timesteps, 
                             log_path="./TRL_evaluations/instance_"+str(parameter)+"/", 
                             n_eval_episodes = eval_episodes, deterministic=True, verbose = 0.0)

###### Pretraining ############

## Actor Pretraining ##

expert_obs = np.load("../RH_Strategies/expert_datasets/expert_observations_"+str(parameter)+".npy")
expert_actions = np.load("../RH_Strategies/expert_datasets/expert_actions_"+str(parameter)+".npy")

class ExpertDataSet(Dataset):
    def __init__(self, expert_obs, expert_actions):
        self.obs = expert_obs
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.obs[index], self.actions[index])

    def __len__(self):
        return len(self.obs)

expert_dataset = ExpertDataSet(expert_obs, expert_actions)
train_size = int(1.0 * len(expert_dataset))
test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size])

def pretrain_agent(
    student,
    batch_size=128,
    epochs=100,
    scheduler_gamma=0.8,
    learning_rate=1.0):
    use_cuda = T.cuda.is_available()
    device = T.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    criterion = nn.MSELoss()

    # Extract initial policy
    policy_student = student.policy.to(device)

    def train(policy_student, device, train_loader, optimizer):
        policy_student.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            action, _, _ = policy_student(data)  #PPO policy outputs actions, values, log_prob
            action_prediction = action.double()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()

    train_loader = T.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    # Define an optimizer and a learning rate schedule.
    #optimizer = optim.Adadelta(policy_student.parameters(), lr=learning_rate)
    optimizer = optim.Adam(policy_student.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma, verbose=False)

    # Train and test the policy model.
    for epoch in range(1, epochs + 1):
        train(policy_student, device, train_loader, optimizer)
        scheduler.step()

    # Implement the trained policy network back into the RL student agent
    pppo.policy = policy_student
    
#Pretrain actor.

Start_time = time.time()
pretrain_agent(
    pppo,
    epochs = 100,
    scheduler_gamma = 0.995,
    learning_rate=10e-4,
    #learning_rate=1.0, #if Adadelta
    batch_size=400)

print('Task number: ', parameter, f"Time of policy reproduction: {(time.time()-Start_time)/60} minutes")
mean_reward, std_reward = evaluate_policy(pppo, env, n_eval_episodes=100)
print('Task number: ', parameter, f"Mean reward = {mean_reward} +/- {1.96 * std_reward / np.sqrt(200)}")

## Critic Pretraining ##

#Define policy evaluation function

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
    
#Pretrain critic.

Start_time = time.time()
pretrain_vf(
    pppo,
    batch_size = 400,
    epochs = 10,
    learning_rate = 0.002,
    n_rollout_steps = 2400,
    n_iterations = 150)

print('Task number: ', parameter, f"Time of policy evaluation: {(time.time()-Start_time)/60} minutes")
mean_reward, std_reward = evaluate_policy(pppo, env, n_eval_episodes=100)
print('Task number: ', parameter, f"Mean reward = {mean_reward} +/- {1.96 * std_reward / np.sqrt(100)}")

## Active Training ##

n_timesteps = 15e6

Start_time = time.time()
pppo.learn(total_timesteps = n_timesteps, callback = eval_callback)
print('Task number: ', parameter, f"Time of finetuning: {(time.time()-Start_time)/60} minutes")

mean_reward, std_reward = evaluate_policy(pppo, env, n_eval_episodes=300)
print('Task number: ', parameter, f"Mean reward = {mean_reward} +/- {1.96 * std_reward / np.sqrt(300)}")

pppo.save("./PPPO_parameters/PPPO_instance_"+str(parameter))