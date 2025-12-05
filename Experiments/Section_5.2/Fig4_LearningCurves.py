# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:59:48 2024

@author: ThomasDM
"""

import numpy as np
import matplotlib.pyplot as plt

data_PPPO = np.load('Results/P_HPPO/instance_3.0/evaluations.npz')
lst_PPP0 = data_PPPO.files
scores_PPPO=data_PPPO["results"]

data_PPO = np.load('Results/HPPO/lr1e_4/evaluations_3.0.npz')
lst_PPO = data_PPO.files
scores_PPO=data_PPO["results"]

data_PPO_2 = np.load('Results/HPPO/lr3e_4/evaluations_3.0.npz')
lst_PPO_2 = data_PPO_2.files
scores_PPO2=data_PPO_2["results"]

EVP_4_12 = np.load('Results/RH_Strategies/RH_4_12/score_history_3.0.npy')
EVP_36_36 = np.load('Results/RH_Strategies/RH_36_36/score_history_3.0.npy')

#Create timescales for PPO
number_evaluations_PPO = 99
time_steps_per_evaluation_PPO = 0.072 #in millions
iterations_PPO = np.arange(0.072, 0.072 + time_steps_per_evaluation_PPO*number_evaluations_PPO, time_steps_per_evaluation_PPO)

number_evaluations_PPO2 = 75
time_steps_per_evaluation_PPO2 = 0.072 #in millions
iterations_PPO2 = np.arange(0.144, 0.144 + time_steps_per_evaluation_PPO*number_evaluations_PPO2,time_steps_per_evaluation_PPO)

number_evaluations_PPPO = 54+1
time_steps_per_evaluation_PPPO = 0.072 #in millions
iterations_PPPO = np.arange(0, time_steps_per_evaluation_PPPO*number_evaluations_PPPO, time_steps_per_evaluation_PPPO)

#Compute mean and confidence interavals
sigma_EVP_4_12 = np.std(EVP_4_12)
EVP_4_12 = np.mean(EVP_4_12)
ci_EVP_4_12 = 1.96 * sigma_EVP_4_12 / np.sqrt(300)

sigma_EVP_36_36 = np.std(EVP_36_36)
EVP_36_36 = np.mean(EVP_36_36)
ci_EVP_36_36 = 1.96 * sigma_EVP_36_36 / np.sqrt(300)

mu_PPO = np.mean(scores_PPO, axis=1)*100
mu_PPO = mu_PPO[0 : number_evaluations_PPO]

sigma_PPO = np.std(scores_PPO, axis=1)*100
sigma_PPO = sigma_PPO[0 : number_evaluations_PPO]

ci_PPO = 1.96 * sigma_PPO / np.sqrt(100)

mu_PPO2 = np.mean(scores_PPO2, axis=1)*100
mu_PPO2 = mu_PPO2[0 : number_evaluations_PPO2]

sigma_PPO2 = np.std(scores_PPO2, axis=1)*100
sigma_PPO2 = sigma_PPO2[0 : number_evaluations_PPO2]

ci_PPO2 = 1.96 * sigma_PPO2 / np.sqrt(100)

mu_PPPO = np.mean(scores_PPPO, axis=1)*100
sigma_PPPO = np.std(scores_PPPO, axis=1)*100

mu_PPPO = np.insert(mu_PPPO, 0, EVP_4_12)
mu_PPPO = mu_PPPO[0 : number_evaluations_PPPO]

sigma_PPPO = np.insert(sigma_PPPO, 0, sigma_EVP_4_12) 
sigma_PPPO = sigma_PPPO[0 : number_evaluations_PPPO]

ci_PPPO = 1.96 * sigma_PPPO / np.sqrt(300)

#Create graph - Fig 4.b

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(12, 9))

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_yticks([9770, 9800, 9900, 10000, 10100, 10274])
ax.set_xticks([0, 1, 2, 3, np.round(iterations_PPPO[-1], decimals=2), 5])

ax.set_xlim(-0.05, 4.1)
ax.set_ylim(9750, 10300)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)

plt.xlabel("Decision Epochs " + r"$(\times10^6)$", fontsize=24)
plt.ylabel("Average Profit ($)", fontsize=24)

ax.plot(iterations_PPPO, mu_PPPO, marker='o', markersize=8, linewidth=0.5, color='blue', label='Our approach')
#ax.plot(iterations_PPO2, mu_PPO2, marker='o', markersize=8, linewidth=0.5, color='blue', label='Our approach')
ax.fill_between(iterations_PPPO, (mu_PPPO-ci_PPPO), (mu_PPPO+ci_PPPO), color='blue', alpha=.2)

plt.axhline(y = EVP_4_12, color = 'green', linestyle = '-.', label=r'RH, $T^{RH}=4,\, Q^{RH}=12$') 
ax.fill_between(iterations_PPO, (EVP_4_12-ci_EVP_4_12), (EVP_4_12+ci_EVP_4_12), color='green', alpha=.2)
plt.vlines(x = iterations_PPPO[-1], ymin =0.0, ymax = mu_PPPO[-1], color = 'black', linestyle = '--') 


ax.legend(loc='center right', fontsize=24)
plt.tight_layout()  # automatically adds margins
plt.savefig("fig4b.pdf")

#Create graph - Fig 4.a

fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(12, 9))

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.set_yticks([8000, 9770, int(np.max(mu_PPPO))])
ax.set_xticks(labels=["0","1","2","3","5", str(np.round(iterations_PPO2[-1], decimals=2)), "6",str(np.round(iterations_PPPO[-1], decimals=2)),"9", str(np.round(iterations_PPO[-1], decimals=2)), "12"],
              ticks=[0, 1, 2, 3, 5, iterations_PPO2[-1], 6, iterations_PPPO[-1], 9, iterations_PPO[-1], 12])

ax.set_xlim(-0.05, 7.2)
ax.set_ylim(7900, 10300)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)

plt.xlabel("Decision Epochs " + r"$(\times10^6)$", fontsize=24)
plt.ylabel("Average Profit ($)", fontsize=24)

ax.plot(iterations_PPPO, mu_PPPO, marker='o', markersize=8, linewidth=0.5, color='blue', label='Our approach')
ax.fill_between(iterations_PPPO, (mu_PPPO-ci_PPPO), (mu_PPPO+ci_PPPO), color='blue', alpha=.2)

ax.plot(iterations_PPO, mu_PPO, marker='^', markersize=8, linewidth=0.5, color='orange', label=r'PPO, $\alpha^{PPO} = 10^{-4}$')
ax.fill_between(iterations_PPO, (mu_PPO-ci_PPO), (mu_PPO+ci_PPO), color='orange', alpha=.2)

ax.plot(iterations_PPO2, mu_PPO2, marker='^', markersize=8, linewidth=0.5, color='red', label=r'PPO, $\alpha^{PPO} = 3\times10^{-4}$')
ax.fill_between(iterations_PPO2, (mu_PPO2-ci_PPO2), (mu_PPO2+ci_PPO2), color='red', alpha=.2)

plt.axhline(y = EVP_4_12, color = 'green', linestyle = '-.', label=r'RH, $T^{RH}=4,\, Q^{RH}=12$') 
ax.fill_between(iterations_PPO, (EVP_4_12-ci_EVP_4_12), (EVP_4_12+ci_EVP_4_12), color='green', alpha=.2)

plt.vlines(x = iterations_PPPO[-1], ymin =0.0, ymax = mu_PPPO[-1], color = 'black', linestyle = '--') 
plt.vlines(x = iterations_PPO[-1], ymin =0.0, ymax = mu_PPO[-1], color = 'black', linestyle = '--') 
plt.hlines(y = mu_PPO[-1], xmin = iterations_PPO[-1], xmax = 12, color = 'black', linestyle = '--') 

plt.vlines(x = iterations_PPO2[-1], ymin =0.0, ymax = mu_PPO2[-1], color = 'black', linestyle = '--') 

ax.legend(loc='lower right', fontsize=24)
plt.tight_layout()  # automatically adds margins
plt.savefig("fig4a.pdf")
