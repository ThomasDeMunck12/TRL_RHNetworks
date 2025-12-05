# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:43:27 2025

@author: thomas
"""

import psutil
import sys
import os
#parameter = float(sys.argv[1]) #Cluster
part = 6.0 #to adapt for parallel computing

print('Start part number: ', part)

os.chdir("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\5_Large_Scale")
sys.path.append("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\hierarchical_rl")  # folder containing ppo_modified.py
sys.path.append("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\5_Large_Scale\\envs")


import numpy as np 
import time 

from env1_february import Env1_F
from env2_february import Env2_F
from env1_march import Env1_M
from env2_march import Env2_M

from Optimization_Problem import EVP

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

n_adjustments = 2
p_interval = 2
      
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

#Optimization parameters 

opt_horizon = 12 
opt_interval = 4

# Environments 

Start_time = time.time()

env2_feb = Env2_F(F = F_max, Lambda = Lambda, tau = tau, distance = distance, mu = mu, N = N, H = H, p_interval = p_interval,
           C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2 = beta2, temp = temp, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5


model = EVP(F = F, Lambda = Lambda_feb[:,:,0:H], tau = tau, distance = distance, mu = mu,
            N = Lambda_feb.shape[0], opt_horizon = opt_horizon, p_interval = p_interval, C_q = C_q, C_r = C_r)
  
#Run the policy and collect data
Start_time = time.time()

n_interactions = int(4800)
expert_obs_1 = []
expert_obs_2 = []
expert_actions_1 = []
expert_actions_2 = []

obs, _ = env2_feb.reset() 
idx = 0
score = 0 

for i in range(n_interactions):
  if idx % opt_interval == 0:
      exp_coef = np.zeros([N, N, opt_horizon])
      for n_adjustment in range(n_adjustments):
          if H - idx >= opt_horizon:
              model.DefineModel(Lambda = Lambda_feb[:, :, idx:idx+opt_horizon], x_start = env2_feb.x, y_start = env2_feb.y, p_start = env2_feb.p, t_start = idx, exp_coef = exp_coef, opt_horizon = opt_horizon)
              model.OptimizeModel()
              p_array, r_array, Repositioning_Perc, Pricing = model.ConvertToImplementableAction()
              Repositioning_Perc[Repositioning_Perc < -1] = -1.0
              Repositioning_Perc[Repositioning_Perc > 1] = 1.0
              Pricing[Pricing < -1] = -1.0
              Pricing[Pricing > 1] = 1.0

          elif H - idx < opt_horizon:
              opt_horizon_intermediate = H - idx
              model.DefineModel(Lambda = Lambda_feb[:, :, idx:idx + opt_horizon], x_start = env2_feb.x, y_start = env2_feb.y, p_start = env2_feb.p, t_start = idx, exp_coef = exp_coef, opt_horizon = opt_horizon_intermediate)
              model.OptimizeModel()
              p_array, r_array, Repositioning_Perc, Pricing = model.ConvertToImplementableAction()
              Pricing[Pricing < -1] = -1.0
              Pricing[Pricing > 1] = 1.0   
          masked = np.ma.masked_equal(p_array, 1)
          p_weighted = masked.mean(axis=1)
          p_normalized = p_weighted - p_weighted[:, None]
          utility = beta0 + beta1 * p_normalized + beta2 * expected_time[:,:, None] 

          mask = np.eye(20, dtype=bool)[:, :, None]  # shape (20, 20, 1)
          mask = np.broadcast_to(mask, utility.shape)   # (20, 20, 12)
          utility[mask] = 0
          exp_coef = np.exp(utility/temp)/np.sum(np.exp(utility/temp), axis=1, keepdims=True)
 
  if idx % p_interval == 0: 
      action_p = Pricing[idx % opt_interval, :]

      expert_obs_1.append(obs)
      expert_actions_1.append(action_p)
      #p_array_period = p_array[:, :, idx % opt_interval]
      
  action_r = Repositioning_Perc[idx % opt_interval, :]
  action_p1 = Pricing[idx % opt_interval, :]

  action_p1 = (action_p1 + 1)/2
  
  obs_dict = {"p": action_p1, "xy": obs["xy"], "t" : obs["t"]}
  #obs_dict = {"p": action_p1, "xyt": obs["xyt"]}
  env2_feb.SetState(obs_dict)
  expert_obs_2.append(obs_dict)
  expert_actions_2.append(action_r)
  
  obs, reward, done, truncation, info = env2_feb.step(action_r)
  score += reward
  idx += 1
  if done:
    obs, _ = env2_feb.reset()
    idx=0
    print(score)
    score = 0 

#Save_data

np.save("./expert_dataset_observations/expert_observations_1_" + str(part)+".npy", expert_obs_1) 
np.save("./expert_dataset_observations/expert_observations_2_" + str(part)+".npy", expert_obs_2) 

np.save("./expert_dataset_actions/expert_actions_1_" + str(part)+".npy", expert_actions_1)
np.save("./expert_dataset_actions/expert_actions_2_" + str(part)+".npy", expert_actions_2)

Time =  (time.time() - Start_time)    
print('opt horizon: ', opt_horizon, ', Total time ', Time)
