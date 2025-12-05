# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 13:14:05 2025

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

#r = np.load("policy_metrics_default/r_PPPO.npy") 
r = np.load("HPPPO_simulations_MC/instance_3.0/r_PPPO.npy") 

r_av = np.mean(r, axis=(3))
r_av = np.sum(r_av, axis=(2))

w = np.load("HPPPO_simulations_MC/instance_3.0/w_PPPO.npy") 
#w = np.load("policy_metrics_default/w_PPPO.npy")

w_av = np.mean(w, axis=(3))
w_av = np.sum(w_av, axis=(2))

mask_r = r_av < 0.50

mask_w = w_av < 0.50


fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(12, 9))
sns.heatmap(w_av, cmap="Blues", ax=axes, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_w, vmin = 0, vmax = 70, xticklabels=[1,2,3,4,5,6,7,8], yticklabels=[1,2,3,4,5,6,7,8], annot_kws={"size": 24})
axes.set_ylabel('Origin', fontsize=24)
axes.set_xlabel('Destination', fontsize=24)
axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)

cbar = axes.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)      

plt.savefig("fig6a.pdf", bbox_inches='tight')

fig2, axes2 = plt.subplots(nrows = 1, ncols=1, figsize=(12, 9))
sns.heatmap(r_av, cmap="Blues", ax=axes2, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_r, vmin = 0, vmax = 70, xticklabels=[1,2,3,4,5,6,7,8], yticklabels=[1,2,3,4,5,6,7,8], annot_kws={"size": 24})
axes2.set_ylabel('Origin', fontsize=24)
axes2.set_xlabel('Destination', fontsize=24)
axes2.tick_params(axis='both', which='major', labelsize=24)
axes2.tick_params(axis='both', which='minor', labelsize=24)

cbar = axes2.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)

plt.savefig("fig6b.pdf", bbox_inches='tight')