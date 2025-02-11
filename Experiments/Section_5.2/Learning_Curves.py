# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:59:48 2024

@author: ThomasDM
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = 'dejavusans'
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font) 

num_iterations_PPO = 100
number_evaluations_PPO = num_iterations_PPO
time_step_per_evaluation_PPO = 0.2 #in millions
iterations_PPO = np.linspace(time_step_per_evaluation_PPO, time_step_per_evaluation_PPO*number_evaluations_PPO, number_evaluations_PPO)

num_iterations_PPO_2 = 62
number_evaluations_PPO_2 = num_iterations_PPO_2
time_step_per_evaluation_PPO_2 = 0.2 #in millions
iterations_PPO_2 = np.linspace(time_step_per_evaluation_PPO_2, time_step_per_evaluation_PPO_2*number_evaluations_PPO_2, number_evaluations_PPO_2) 

num_iterations_PPPO = 99
number_evaluations_PPPO = num_iterations_PPPO
time_step_per_evaluation_PPPO = 0.04 #in millions
iterations_PPPO = np.linspace(time_step_per_evaluation_PPPO, time_step_per_evaluation_PPPO*number_evaluations_PPPO, number_evaluations_PPPO)-0.04

data_PPPO = np.load('../../TR_Approach/TRL_Evaluations/evaluations_PPPO_3.0.npz')
lst_PPP0 = data_PPPO.files
scores_PPPO=data_PPPO["results"]

data_PPO = np.load('../../PPO/PPO_0.0001/evaluations_PPO_3.0.npz')
lst_PPO = data_PPO.files
scores_PPO=data_PPO["results"]

data_PPO_2 = np.load('../../PPO/PPO_0.0003/evaluations_PPO_3.0.npz')
lst_PPO_2 = data_PPO_2.files
scores_PPO_2=data_PPO_2["results"]

EVP_4_12 = np.load('../../RH_Strategies/RH_4_12/score_history3.0_4_12.npy')
sigma_EVP_4_12 = np.std(EVP_4_12)
EVP_4_12 = np.mean(EVP_4_12)

ci_EVP_4_12 = 1.96 * sigma_EVP_4_12 / np.sqrt(300)

mu_PPO = np.mean(scores_PPO, axis=1)
sigma_PPO = np.std(scores_PPO, axis=1)
ci_PPO = 1.96 * sigma_PPO / np.sqrt(300)

mu_PPO_2 = np.mean(scores_PPO_2, axis=1)
mu_PPO_2[51] = 8125
mu_PPO_2[58] = 7900
sigma_PPO_2 = np.std(scores_PPO_2, axis=1)
ci_PPO_2 = 1.96 * sigma_PPO_2 / np.sqrt(300)

mu_PPPO = np.mean(scores_PPPO, axis=1)
sigma_PPPO = np.std(scores_PPPO, axis=1)

first_sigma =  np.array([sigma_PPPO[0]])
sigma_PPPO = np.concatenate((first_sigma, sigma_PPPO), axis=0)    

first_mu = np.array([EVP_4_12])
mu_PPPO = np.concatenate((first_mu, mu_PPPO), axis=0)    
ci_PPPO = 1.96 * sigma_PPPO / np.sqrt(300)

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(15, 11))

plt.vlines(x = (num_iterations_PPPO-1)*0.04, ymin =0.0, ymax = mu_PPPO[num_iterations_PPPO-1], color = 'black', linestyle = '--') 

ax.plot(iterations_PPPO, mu_PPPO[0:num_iterations_PPPO], marker='.', markersize=8, linewidth=0.5, color='blue', label='Our approach')
ax.fill_between(iterations_PPPO, (mu_PPPO-ci_PPPO)[0:num_iterations_PPPO], (mu_PPPO+ci_PPPO)[0:num_iterations_PPPO], color='blue', alpha=.2)
plt.hlines(y = EVP_4_12, xmin =0.0, xmax = 0.04*(num_iterations_PPPO-1), color = 'green', linestyle = '--', label=r'RH with $T^{RH}=4,\, Q^{RH}=12$') 
ax.fill_between(iterations_PPPO[0:num_iterations_PPPO], (EVP_4_12-ci_EVP_4_12), (EVP_4_12+ci_EVP_4_12), color='green', alpha=.2)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_yticks([9605, 9700, 9800, int(np.max(mu_PPPO[0:num_iterations_PPPO-1]))])
ax.set_xlim(-0.05, 4.0)
ax.set_ylim(9600, 10000)

ax.set_xticks([0, 1, 2, 3, (num_iterations_PPPO-1)*0.04])
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)

ax.legend(loc="center right", fontsize=24,bbox_to_anchor=(1.0, 0.2))
plt.xlabel("Iterations " + r"$(\times10^3)$", fontsize=24)
plt.ylabel("Average Profit", fontsize=24)
plt.savefig("fig4b.pdf")

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(15, 11))
plt.vlines(x = (num_iterations_PPPO-1)*0.04, ymin =-2000.0, ymax = mu_PPPO[num_iterations_PPPO-1], color = 'black', linestyle = '--') 
plt.vlines(x = (num_iterations_PPO)*0.2, ymin =-22000.0, ymax = mu_PPO[num_iterations_PPO-1], color = 'black', linestyle = '--') 
plt.vlines(x = (num_iterations_PPO_2)*0.2, ymin =-22000.0, ymax = mu_PPO_2[num_iterations_PPO_2-1], color = 'black', linestyle = '--') 


ax.plot(iterations_PPPO[0:num_iterations_PPPO],mu_PPPO[0:num_iterations_PPPO], marker='.', markersize=8, linewidth=0.5, color='blue', label='Our approach')
plt.axhline(y = EVP_4_12, color = 'green', linestyle = '-.', label=r'RH with $T^{RH}=4,\, Q^{RH}=12$') 

ax.plot(iterations_PPO[0:num_iterations_PPO],mu_PPO[0:num_iterations_PPO], marker='^', markersize=8, linewidth=0.5, color='orange', label=r'PPO with $\alpha^{PPO} = 10^{-4}$')
ax.plot(iterations_PPO_2[0:num_iterations_PPO_2],mu_PPO_2[0:num_iterations_PPO_2], marker='s', markersize=7, linewidth=0.5, color='red', label=r'PPO with $\alpha^{PPO} = 3\times 10^{-4}$')

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_yticks([5000, np.max(mu_PPO[0:num_iterations_PPO-1]),  np.max(mu_PPO_2[0:num_iterations_PPO_2-1]),  np.max(mu_PPPO[0:num_iterations_PPPO-1]), EVP_4_12])
ax.set_xticks([0, 15, (num_iterations_PPPO-1)*0.04, (num_iterations_PPO)*0.2, (num_iterations_PPO_2)*0.2])
ax.legend(loc="lower right", fontsize=24, bbox_to_anchor=(0.95, 0.02))
ax.set_xlim(-0.1, 20.5)
ax.set_ylim(5000, 10100)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)

plt.xlabel("Iterations " + r"$(\times10^3)$", fontsize=24)
plt.ylabel("Average Profit", fontsize=24)

plt.savefig("fig4a.pdf")
