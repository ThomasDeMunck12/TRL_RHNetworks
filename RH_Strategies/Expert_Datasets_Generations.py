# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:40:13 2022

@author: ThomasDM
"""
import psutil
import sys
#parameter = float(sys.argv[1]) #on Slurm cluster
parameter = 3.0 #personal laptop - default instance
part = 1.0 #to adapt for parallel computing

print('Start task number: ', parameter, 'part. ', part)

import numpy as np 
import time 
from RideHailing_Env import RideHailingEnv
from Optimization_Problem import EVP

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

periods_window = 12 # 1 hour before/after the beginning of the episode.
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

opt_horizon = 12 #Q^{RH}
opt_interval = 4 #T^{RH}
nepisodes = 1 #Number of episodes to evaluate

#Define environment, optimization model 

env = RideHailingEnv(F = F, Lambda = Lambda[:, :, 0:H], tau = tau, distance = distance,
                     N = N, H = H,
                     C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2=beta2, temp = temp,
                     x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

model = EVP(F=F, Lambda=Lambda[:,:, 0:H], tau = tau, distance = distance,
            N = N, opt_horizon = opt_horizon, C_q = C_q, C_r = C_r)
  
#Run the policy and collect data
Start_time = time.time()

n_interactions = int(1.0e4)

expert_obs = []
expert_actions = []

obs, _ = env.reset() 
idx = 0

for i in range(n_interactions):
  if idx % opt_interval == 0:
      if H-idx >= opt_horizon:
          model.DefineModel(Lambda = Lambda[:, :, idx:idx+opt_horizon], x_start = env.x, y_start = env.y, opt_horizon = opt_horizon)
          model.OptimizeModel()
          action_cp = model.ConvertToImplementableAction()
          action_cp[action_cp < -1] = -1.0
      elif H-idx < opt_horizon:
          opt_horizon_intermediate = H - idx
          model.DefineModel(Lambda = Lambda[:, :, idx:idx+opt_horizon_intermediate], x_start = env.x, y_start = env.y, opt_horizon = opt_horizon_intermediate)
          model.OptimizeModel()
          action_cp = model.ConvertToImplementableAction()
          action_cp[action_cp < -1] = -1.0
      
  action = action_cp[idx % opt_interval, :]
  print(psutil.virtual_memory()[0]/2.**30)  # physical memory usage

  expert_obs.append(obs)
  expert_actions.append(action)
  
  obs, reward, done, truncation, info = env.step(action)
  idx += 1
  if done:
    obs, _ = env.reset()
    idx=0
    
#Save_data

np.save("expert_dataset_observations/expert_observations_"+str(parameter)+"_"+str(part)+".npy", expert_obs) 
np.save("expert_dataset_actions/expert_actions_"+str(parameter)+"_"+str(part)+".npy", expert_actions)

Time =  (time.time() - Start_time)    
print('Task number: ', parameter,', opt horizon: ', opt_horizon, ', Total time ', Time)
