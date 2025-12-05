# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 01:12:34 2025

@author: thoma
"""

import sys
import os

#parameter = float(sys.argv[1]) #Cluster
parameter = 3.0 #Personal laptop - default instance
print('Start task number: ', parameter)
os.chdir("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\5_Large_Scale")
sys.path.append("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\hierarchical_rl")  # folder containing ppo_modified.py
sys.path.append("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\5_Large_Scale\\envs")

import numpy as np 
import time 
from env2_march import Env2_M
from Optimization_Problem import EVP
#import random 

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

param = int(parameter)

# Load data

Lambda = np.load("../data/arrival_rates_20loc_february.npy")
Lambda_feb = np.mean(Lambda, axis=3)

Lambda_march = np.load("../data/arrival_rates_20loc_march.npy")
Lambda_march = Lambda_march[:,:,:,param-1]

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

# Monitor the execution of the policy

opt_horizon = 12
opt_interval = 4
n_episodes = 1
opt_interval_last_period = H % opt_interval 

x_m = np.zeros([N, H, n_episodes]) #drivers available
y_m = np.zeros([N, N, H, n_episodes]) #drivers in transit
p_m = np.zeros([N, N, H, n_episodes]) #prices
d_m = np.zeros([N, N, H, n_episodes]) #dispatched drivers
c_m = np.zeros([N, N, H, n_episodes]) #customer requests
lambda_m = np.zeros([N, N, H, n_episodes]) #arrival rates
f_m = np.zeros([N, N, H, n_episodes]) #unsatisfied customers
w_m = np.zeros([N, N, H, n_episodes]) #self-relocating drivers
r_m = np.zeros([N, N, H, n_episodes]) #repositioned drivers
z_m = np.zeros([N, N, H, n_episodes]) #drivers completing service

Revenue_v = np.zeros([H, n_episodes]) #revenues
LostSales_v = np.zeros([H, n_episodes]) #lost sales costs
Repositioning_v = np.zeros([H, n_episodes]) #repositioning costs

Profit_v = np.zeros([H, n_episodes]) #profits

Start_time = time.time()
env2_march = Env2_M(F = F_max, Lambda=Lambda_march[:, :, 0 : H], tau = tau, distance = distance, mu = mu, N = N, H = H, p_interval = p_interval,
           C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2 = beta2, temp = temp, p_0 = p_init, x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

model = EVP(F = F, Lambda = Lambda_feb[:,:,0:H], tau = tau, distance = distance, mu = mu,
            N = Lambda_feb.shape[0], opt_horizon = opt_horizon, p_interval = p_interval, C_q = C_q, C_r = C_r)
  
#if opt_interval_last_period != 0: 
#    model_last = EVP(F = F, Lambda = Lambda_feb[:,:,0:H], tau = tau, distance = distance,
#                N = Lambda_feb.shape[0], opt_horizon = opt_interval_last_period, p_interval = p_interval, C_q = C_q, C_r = C_r)     
#    model_last.DefineModel()

Best_score = -np.inf
Score_history = []
Average_score = 0

for i in range(n_episodes):
    observation = env2_march.reset()
    done = False
    p_start = None
    score = 0
    idx = 0
    while not done:
        x_m[:, idx, i] = env2_march.x
        y_m[:, :, idx, i] = env2_march.y
        if idx % opt_interval == 0:
            exp_coef = np.zeros([N, N, opt_horizon])
            for n_adjustment in range(n_adjustments):
                if H-idx >= opt_horizon:
                    model.DefineModel(Lambda = Lambda_feb[:, :, idx:idx+opt_horizon], x_start = env2_march.x, y_start = env2_march.y, p_start = env2_march.p, t_start = idx, exp_coef = exp_coef, opt_horizon = opt_horizon)
                    model.OptimizeModel()
                    p_array, _, Repositioning_Perc, Pricing = model.ConvertToImplementableAction()
                    Repositioning_Perc[Repositioning_Perc < -1] = -1.0
                    Repositioning_Perc[Repositioning_Perc > 1] = 1.0
                    Pricing[Pricing < -1] = -1.0
                    Pricing[Pricing > 1] = 1.0

                elif H-idx < opt_horizon:
                    opt_horizon_intermediate = H - idx
                    model.DefineModel(Lambda = Lambda_feb[:, :, idx:idx+opt_horizon], x_start = env2_march.x, y_start = env2_march.y, p_start = env2_march.p, t_start = idx, exp_coef = exp_coef, opt_horizon = opt_horizon_intermediate)
                    model.OptimizeModel()
                    p_array, _, Repositioning_Perc, Pricing = model.ConvertToImplementableAction()
                    Pricing[Pricing < -1] = -1.0
                    Pricing[Pricing > 1] = 1.0
                masked = np.ma.masked_equal(p_array, 1)
                p_weighted = masked.mean(axis=1)
                p_normalized = p_weighted - p_weighted[:, None]
                utility = beta0 + beta1 * p_normalized + beta2 * expected_time[:,:, None] 

                mask = np.eye(N, dtype=bool)[:, :, None] 
                mask = np.broadcast_to(mask, utility.shape)   # (8, 8, 12)
                utility[mask] = 0
                exp_coef = np.exp(utility/temp)/np.sum(np.exp(utility/temp), axis=1, keepdims=True)

                    
        action_r = Repositioning_Perc[idx % opt_interval, :]
        #print(action_r)
        action_p1 = Pricing[idx % opt_interval, :]

        action_p1 = (action_p1+1)/2
        action_p2 = np.reshape(action_p1, [N, N])
        
        env2_march.p = action_p2
        p_m[:, :, idx, i] = env2_march.p
        observation_, reward, done, truncation, info = env2_march.step(action_r)
        d_m[:, :, idx, i] = info["Dispatched drivers"]
        c_m[:, :, idx, i] = info["Realized requests"]
        lambda_m[:, :, idx, i] = info["Arrival rates"]
        f_m[:, :, idx, i] = info["Requests unsatisfied"] 
        w_m[:, :, idx, i] = info["Self-relocating drivers"]
        r_m[:, :, idx, i] = info["Repositioned drivers"]
        z_m[:, :, idx, i] = info["Drivers completing service"]
        Revenue_v[idx, i] = info["Revenue"]
        LostSales_v[idx, i] = info["Lost sales costs"]
        Repositioning_v[idx, i] = info["Repositioning costs"]
        Profit_v[idx, i] = info["Revenue"]-info["Lost sales costs"]-info["Repositioning costs"]
        score += reward
        observation = observation_
        idx = idx + 1
    Score_history.append(score)

Avg_score = np.mean(Score_history)

Time =  (time.time() - Start_time)    
Average_time = Time/n_episodes

print('opt horizon: ', opt_horizon)
print('episode ', i, 'avg %.3f' % Avg_score, 'average time %13f' % Average_time)
np.save("./RH_1_12/score_history_"+str(parameter)+".npy", Score_history)


