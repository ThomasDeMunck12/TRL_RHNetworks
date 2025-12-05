# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:43:27 2025

@author: thomas
"""

import sys
import os
parameter = 3.0 #personal laptop - default instance
#parameter = float(sys.argv[1]) #Cluster
part = 1.0 #to adapt for parallel computing
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print('Start task number: ', parameter, 'part. ', part)

import numpy as np 
import time 
#from env1 import Env1
from envs.env2 import Env2
from Optimization_Problem import EVP

#Base parameters

F = 900 #fleet size
N = 8 #number of locations
H = 36 #number of decision times

traffic_const = 1.0 #constant adjusting transit times and arrival rates
p_interval = 2

beta0 = -0.15
beta1 = 8.0
beta2 = -1.5
temp = 1.0

C_q = 0.4
C_r = 0.2

n_adjustments = 1

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
    if temp<1.0:
        n_adjustments = 1
elif (parameter > 15.0) and (parameter <= 20.0):
    param = parameter - 15.0
    
    if (param == 1):
        beta0 = 0.0
        beta1 = 3.0
        beta2 = -3.0
        
    elif (param == 2):
        beta0 = 0.0
        beta1 = 4.5
        beta2 = -2.0
        
    elif (param == 3):
        beta0 = 0.0
        beta1 = 3.0
        beta2 = -2.0
        
    elif (param == 4):
        beta0 = 1.71 #1.75
        beta1 = 0.0
        beta2 = -2.0
        
    elif (param == 5):
        beta0 = -6.5 #-7.5
        beta1 = 3.0
        beta2 = 0.0
elif (parameter > 20.0) and (parameter <= 25.0):
    param = parameter - 20.0
    if param <= 4:
        p_interval = int(param)
    else:
        p_interval = 6
        
elif (parameter > 25.0) and (parameter <= 30.0):
    param = parameter - 25.0
    if param <= 3:
        C_q = 0.2 * (param-1)
    elif param == 4:
        C_q = 0.8
    elif param == 5:
        C_q = 1.2
elif (parameter > 30.0) and (parameter <= 35.0):
    param = parameter - 30.0
    C_r = 0.1 * (param-1)
      
# Load data

Lambda = np.load("../data/arrival_rates_8loc.npy")
Lambda_day = np.copy(Lambda)
Lambda = Lambda[:, :, 84:]
Lambda = Lambda * traffic_const

average_time = np.load("../data/average_trip_durations_8loc.npy")
tau = 1/average_time
tau_init = np.copy(tau)
expected_time = 1/tau_init
expected_time = 1/tau_init

distance = np.load("../data/average_distances_8loc.npy")

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

p_init = np.zeros([N, N])

#Optimization parameters 

opt_horizon = 12 #Q^{RH}
opt_interval = 4 #T^{RH}

# Environments 

Lambda_traffic = 1.0*Lambda  
tau_traffic = 1.0*tau
Start_time = time.time()

env2 = Env2(F = F, Lambda=Lambda[:, :, 0 : H], tau = tau_traffic, distance = distance, N = N, H = H, p_interval = p_interval,
           C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2 = beta2, temp = temp, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

model = EVP(F = F, Lambda = Lambda[:, :, 0 : H], tau = tau_traffic, distance = distance, N = N, opt_horizon = opt_horizon, p_interval = p_interval, C_q = C_q, C_r = C_r)

#Run the policy and collect data
Start_time = time.time()

n_interactions = int(10800)
expert_obs_1 = []
expert_obs_2 = []
expert_actions_1 = []
expert_actions_2 = []

obs, _ = env2.reset() 
idx = 0

for i in range(n_interactions):
  if idx % opt_interval == 0:
      exp_coef = np.zeros([N, N, opt_horizon])
      for n_adjustment in range(n_adjustments):
          if H - idx >= opt_horizon:
              model.DefineModel(Lambda = Lambda[:, :, idx:idx+opt_horizon], x_start = env2.x, y_start = env2.y, p_start = env2.p, t_start = idx, exp_coef = exp_coef, opt_horizon = opt_horizon)
              model.OptimizeModel()
              p_array, r_array, Repositioning_Perc, Pricing = model.ConvertToImplementableAction()
              Repositioning_Perc[Repositioning_Perc < -1] = -1.0
              Repositioning_Perc[Repositioning_Perc > 1] = 1.0
              Pricing[Pricing < -1] = -1.0
              Pricing[Pricing > 1] = 1.0

          elif H - idx < opt_horizon:
              opt_horizon_intermediate = H - idx
              model.DefineModel(Lambda = Lambda[:, :, idx:idx + opt_horizon], x_start = env2.x, y_start = env2.y, p_start = env2.p, t_start = idx, exp_coef = exp_coef, opt_horizon = opt_horizon_intermediate)
              model.OptimizeModel()
              p_array, r_array, Repositioning_Perc, Pricing = model.ConvertToImplementableAction()
              Pricing[Pricing < -1] = -1.0
              Pricing[Pricing > 1] = 1.0   
          masked = np.ma.masked_equal(p_array, 1)
          p_weighted = masked.mean(axis=1)
          p_normalized = p_weighted - p_weighted[:, None]
          utility = beta0 + beta1 * p_normalized + beta2 * expected_time[:,:, None] 

          mask = np.eye(8, dtype=bool)[:, :, None]  # shape (8, 8, 1)
          mask = np.broadcast_to(mask, utility.shape)   # (8, 8, 12)
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
  env2.SetState(obs_dict)
  expert_obs_2.append(obs_dict)
  expert_actions_2.append(action_r)
  
  obs, reward, done, truncation, info = env2.step(action_r)
  idx += 1
  if done:
    obs, _ = env2.reset()
    idx=0

#Save_data

np.save("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_"+str(part)+".npy", expert_obs_1) 
np.save("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_"+str(part)+".npy", expert_obs_2) 

np.save("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_"+str(part)+".npy", expert_actions_1)
np.save("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_"+str(part)+".npy", expert_actions_2)

Time =  (time.time() - Start_time)    
print('Task number: ', parameter,', opt horizon: ', opt_horizon, ', Total time ', Time)
