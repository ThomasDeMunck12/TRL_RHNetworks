import sys

parameter = float(sys.argv[1]) #for Slurm cluster
#parameter = 3.0 #personal laptop

print('Start task number: ', parameter)

import numpy as np 
import time 
from env import Env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch as T

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


#Define environment

env = Env(F = F, Lambda = Lambda[:, :, 0:H], tau = tau, distance = distance,
                     N = N, H = H,
                     C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2=beta2, temp = temp,
                     x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5

env = Monitor(env)

#Define PPO 

n_steps = 5400
target_kl = 1.5
gamma = 1.0
learning_rate = 1e-4
batch_size = 540
gae = 0.95
epochs = 10
neural_net = 128 

policy_kwargs = dict(activation_fn=T.nn.Tanh,
                       net_arch=dict(pi=[neural_net, neural_net, neural_net, neural_net], vf=[neural_net, neural_net, neural_net, neural_net])) #network architecture

pppo = PPO("MlpPolicy", env,
           learning_rate = learning_rate, batch_size = batch_size,  gae_lambda = gae, n_epochs = epochs,
           n_steps = n_steps, target_kl = target_kl, gamma = gamma, policy_kwargs = policy_kwargs, verbose = 0) #pppo

#Define callbacks

n_timesteps = 10800000
eval_timesteps = 108000
eval_episodes = 300

eval_callback = EvalCallback(env, eval_freq=eval_timesteps, 
                             log_path="./PPO_evaluations_1e-4/instance_"+str(parameter)+"/", 
                             n_eval_episodes = eval_episodes, deterministic=True, verbose = 0.0)

Start_time = time.time()
pppo.learn(total_timesteps = n_timesteps, callback = eval_callback)
print('Task number: ', parameter, f"Time of finetuning: {(time.time()-Start_time)/60} minutes")

env_sim = Env(F = F, Lambda = Lambda[:, :, 0:H], tau = tau, distance = distance,
                     N = N, H = H,
                     C_q = C_q, C_r = C_r, beta0 = beta0, beta1 = beta1, beta2 = beta2, temp = temp,
                     x_0 = x_init, y_0 = y_init) #T->infty, prob->Unif, 4-8-12 loc:beta0 = 7.0, 6.00, -1.5
seed = 10

n_episodes = 300

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
Revenue_v = np.zeros([H, n_episodes]) #revenues
LostSales_v = np.zeros([H, n_episodes]) #lost sales costs
Repositioning_v = np.zeros([H, n_episodes]) #repositioning costs
Profit_v = np.zeros([H, n_episodes]) #profits

scores = []

for i in range(n_episodes):

    observation, _ = env_sim.reset(seed=seed+i)

    done = False
    idx = 0
    score = 0

    while not done:
        action, _ = pppo.predict(observation, deterministic=True)
        observation_, reward, done, _, info = env_sim.step(action)

        x_m[:, idx, i] = env_sim.x
        y_m[:, :, idx, i] = env_sim.y
        p_m[:, :, idx, i] = info["Prices"]
        r_m[:, :, idx, i] = info["Repositioned drivers"]
        w_m[:, :, idx, i] = info["Self-relocating drivers"]
        c_m[:, :, idx, i] = info["Realized requests"]
        lambda_m[:, :, idx, i] = info["Arrival rates"]
        d_m[:, :, idx, i] = info["Dispatched drivers"]
        z_m[:, :, idx, i] = info["Drivers completing service"]
        f_m[:, :, idx, i] = info["Requests unsatisfied"]

        Revenue_v[idx, i] = info["Revenue"]
        Repositioning_v[idx, i] = info["Repositioning costs"]
        LostSales_v[idx, i] = info["Lost sales costs"]
        Profit_v[idx, i] = info["Revenue"]-info["Lost sales costs"]-info["Repositioning costs"]

        score += reward
        observation = observation_
        idx = idx + 1
    scores.append(score)

#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/x_PPPO.npy", x_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/y_PPPO.npy", y_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/p_PPPO.npy", p_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/d_PPPO.npy", d_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/c_PPPO.npy", c_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/lambda_PPPO.npy", lambda_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/f_PPPO.npy", f_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/w_PPPO.npy", w_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/r_PPPO.npy", r_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/z_PPPO.npy", z_m)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/Revenue_PPPO.npy", Revenue_v)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/LostSales_PPPO.npy", LostSales_v)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/Repositioning_PPPO.npy", Repositioning_v)
#np.save("./PPO_simulations_1e-4/instance_"+str(parameter)+"/Profit_PPPO.npy", Profit_v)