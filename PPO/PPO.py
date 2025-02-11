# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:04:16 2023

@author: ThomasDM
"""
from typing import Callable
import sys

#parameter = float(sys.argv[1]) # Slurm cluster
parameter = 3.0 # default instance
print('Start task number: ', parameter)

import numpy as np 
import time 
from RideHailing_Env import RideHailingEnv

from stable_baselines3 import PPO
import torch as T
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor
from gymnasium import spaces
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

from typing import Callable

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
        beta0 = -0.15
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

#Learning rate schedule

#Parameters 

n_steps = 2400
target_kl = 1.0
gamma = 1.0

learning_rate = 0.0003 # or 0.0001
batch_size = 150 
gae = 0.95
n_epochs = 10 
ent_coef = 0.000
vf_coef = 0.5
neural_net = 128 

policy_kwargs = dict(activation_fn=T.nn.Tanh,
                       net_arch=dict(pi=[neural_net, neural_net, neural_net, neural_net], vf=[neural_net, neural_net, neural_net, neural_net])) #network architecture

ppo = PPO("MlpPolicy", env, learning_rate = learning_rate, batch_size = batch_size, gae_lambda = gae, n_epochs = n_epochs,
          n_steps = n_steps, target_kl = target_kl, ent_coef = ent_coef, vf_coef = vf_coef, gamma = gamma, policy_kwargs = policy_kwargs, verbose = 0.0) #ppo

#Define callbacks

eval_timesteps = 50e4
eval_episodes = 300

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=75, verbose=0)

eval_callback = EvalCallback(env, eval_freq=eval_timesteps, callback_after_eval=stop_train_callback,
                             log_path="./PPO_evaluations/instance_"+str(parameter)+"/", 
                             n_eval_episodes = eval_episodes, deterministic=True, verbose = 0.0)

#Run PPO

n_timesteps = 50e6

Start_time = time.time()
ppo.learn(total_timesteps=n_timesteps, callback=eval_callback)
print('Task number: ', parameter, f", time for training: {(time.time() - Start_time)/60} minutes")

mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=300)
print('Task number: ', parameter, f", mean reward = {mean_reward} +/- {1.96 * std_reward / np.sqrt(300)}")

ppo.save("./PPO_parameters/PPO_instance_"+str(parameter))