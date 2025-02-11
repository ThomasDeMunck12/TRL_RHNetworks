# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:17:48 2024

@author: ThomasDM
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data from simulation

w_PPPO_3 = np.load("PPPO_simulations/instance_3.0/w_PPPO.npy")
r_PPPO_3 = np.load("PPPO_simulations/instance_3.0/r_PPPO.npy") 

w_PPPO_3_av = np.mean(w_PPPO_3, axis=(3))

r_PPPO_3_av = np.mean(r_PPPO_3, axis=(3))

w_PPPO_3_sum = np.sum(w_PPPO_3_av, axis=(2))

r_PPPO_3_sum = np.sum(r_PPPO_3_av, axis=(2))


mask_r_3 = r_PPPO_3_sum < 0.51

mask_w_3 = w_PPPO_3_sum < 0.51


fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(12, 9))
sns.heatmap(w_PPPO_3_sum, cmap="Blues", ax=axes, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_w_3, vmin = 0, vmax = 50, xticklabels=[1,2,3,4,5,6,7,8], yticklabels=[1,2,3,4,5,6,7,8])
axes.set_ylabel('Origin', fontsize=20)
axes.set_xlabel('Destination', fontsize=20)
axes.tick_params(axis='both', which='major', labelsize=20)
axes.tick_params(axis='both', which='minor', labelsize=20)
plt.savefig("fig6a_heatmap_admission.pdf", bbox_inches='tight')

fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(12, 9))
sns.heatmap(r_PPPO_3_sum, cmap="Blues", ax=axes, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_r_3, vmin = 0, vmax = 100, xticklabels=[1,2,3,4,5,6,7,8], yticklabels=[1,2,3,4,5,6,7,8])
axes.set_ylabel('Origin', fontsize=20)
axes.set_xlabel('Destination', fontsize=20)
axes.tick_params(axis='both', which='major', labelsize=20)
axes.tick_params(axis='both', which='minor', labelsize=20)
plt.savefig("fig6b_heatmap_admission.pdf", bbox_inches='tight')
