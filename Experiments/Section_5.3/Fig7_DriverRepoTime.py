# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:27:45 2025

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

d = np.load("HPPPO_simulations_MC/instance_3.0/d_PPPO.npy") 
c = np.load("HPPPO_simulations_MC/instance_3.0/lambda_PPPO.npy") 
r = np.load("HPPPO_simulations_MC/instance_3.0/r_PPPO.npy") 
w = np.load("HPPPO_simulations_MC/instance_3.0/w_PPPO.npy") 
x = np.load("HPPPO_simulations_MC/instance_3.0/x_PPPO.npy") 

d_av = np.mean(d, axis=(3))
c_av = np.mean(c, axis=(3))
r_av = np.mean(r, axis=(3))
w_av = np.mean(w, axis=(3))
x_av = np.mean(x, axis=(2))

mask_r = r_av < 0.50
mask_w = r_av < 0.50

d_IN = np.sum(d_av, axis=0) 
d_OUT = np.sum(d_av, axis=1)
d_NF = d_IN - d_OUT

c_IN = np.sum(c_av, axis=0) 
c_OUT = np.sum(c_av, axis=1)
c_NF = c_IN - c_OUT

r_IN = np.sum(r_av, axis=0) 
r_OUT = np.sum(r_av, axis=1)
r_NF = r_IN - r_OUT

w_IN = np.sum(w_av, axis=0) 
w_OUT = np.sum(w_av, axis=1)
w_NF = w_IN - w_OUT

ncols = 1
nrows = 1

iterations = np.linspace(0, 35, 36)

fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(12,9))

axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)
axes.set_xlabel("Decision Epochs", fontsize=24)
axes.set_ylabel("Net Flows", fontsize=24)
#axes.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8], labels=[0.4, 0.5, 0.6, 0.7, 0.8])

#axes.plot(iterations, c_NF[5, :], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='Dispatched drivers')
axes.plot(iterations, d_NF[5, :], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='Dispatched drivers')
#axes.plot(iterations, x_av[5, :], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='Dispatched drivers')
axes.plot(iterations, r_NF[5, :], marker='o', color='blue', markersize=6, linewidth=1.0,label='Repositioned drivers')
axes.plot(iterations, w_NF[5, :], marker='s', linestyle = 'dashed', color='red', markersize=6, linewidth=1.0,label='Self-relocating drivers')

axes.set_ylim((-15 , 25))
axes.set_xlim((-0.5, 35.5))
axes.legend(loc='upper right', prop={'size': 24})      
plt.savefig("fig7a.pdf", bbox_inches='tight')

service_rate = np.sum(d_OUT, axis=1)/np.sum(c_OUT, axis=1)

fig2, axes2 = plt.subplots(nrows = 1, ncols=1, figsize=(3, 9))
sns.heatmap(r_av[:,5,2].reshape(-1, 1)+r_av[:,5,3].reshape(-1, 1), cmap="Blues", ax=axes2, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_r[:,5,2].reshape(-1, 1), vmin = 0, vmax = 41, yticklabels=[1,2,3,4,5,6,7,8], xticklabels=False, annot_kws={"size": 24})
axes2.set_ylabel('Origin', fontsize=24)
axes2.set_title('Epochs 3, 4', fontsize=24, fontweight='bold', pad=20)
axes2.tick_params(axis='both', which='major', labelsize=24)
axes2.tick_params(axis='both', which='minor', labelsize=24)

cbar = axes2.collections[0].colorbar
cbar.set_ticks([0, 10, 20, 30, 40])
cbar.ax.tick_params(labelsize=24)

plt.savefig("fig7b.pdf", bbox_inches='tight')

fig2, axes2 = plt.subplots(nrows = 1, ncols=1, figsize=(3, 9))
sns.heatmap(r_av[:,5,8].reshape(-1, 1)+r_av[:,5,9].reshape(-1, 1)+r_av[:,5,10].reshape(-1, 1), cmap="Blues", ax=axes2, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_r[:,5,8].reshape(-1, 1), vmin = 0, vmax = 41, xticklabels=False, yticklabels=[1,2,3,4,5,6,7,8], annot_kws={"size": 24})
axes2.set_ylabel('Origin', fontsize=24)
axes2.set_title('Epochs 9, 10', fontsize=24, fontweight='bold', pad=20)
axes2.tick_params(axis='both', which='major', labelsize=24)
axes2.tick_params(axis='both', which='minor', labelsize=24)

cbar = axes2.collections[0].colorbar
cbar.set_ticks([0, 10, 20, 30, 40])
cbar.ax.tick_params(labelsize=24)

plt.savefig("fig7b2.pdf", bbox_inches='tight')

mask = w_av[:,5,2]+w_av[:,5,3] < 0.5 
fig2, axes2 = plt.subplots(nrows = 1, ncols=1, figsize=(3, 9))
sns.heatmap(w_av[:,5,2].reshape(-1, 1)+w_av[:,5,3].reshape(-1, 1), cmap="Blues", ax=axes2, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask.reshape(-1, 1), vmin = 0, vmax = 20, xticklabels=False, yticklabels=[1,2,3,4,5,6,7,8], annot_kws={"size": 24})
axes2.set_ylabel('Origin', fontsize=24)
axes2.set_title('Epochs 3, 4', fontsize=24, fontweight='bold', pad=20)
axes2.tick_params(axis='both', which='major', labelsize=24)
axes2.tick_params(axis='both', which='minor', labelsize=24)

cbar = axes2.collections[0].colorbar
cbar.set_ticks([0, 5, 10, 15, 20])
cbar.ax.tick_params(labelsize=24)

plt.savefig("fig7c.pdf", bbox_inches='tight')

mask = w_av[:,5,8]+w_av[:,5,9]+w_av[:,5,10]<0.5
fig2, axes2 = plt.subplots(nrows = 1, ncols=1, figsize=(3, 9))
sns.heatmap(w_av[:,5,8].reshape(-1, 1)+w_av[:,5,9].reshape(-1, 1)+w_av[:,5,10].reshape(-1, 1), cmap="Blues", ax=axes2, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask.reshape(-1, 1), vmin = 0, vmax = 20, xticklabels=False, yticklabels=[1,2,3,4,5,6,7,8], annot_kws={"size": 24})
axes2.set_ylabel('Origin', fontsize=24)
axes2.set_title('Epochs 9, 10', fontsize=24, fontweight='bold', pad=20)
axes2.tick_params(axis='both', which='major', labelsize=24)
axes2.tick_params(axis='both', which='minor', labelsize=24)

cbar = axes2.collections[0].colorbar
cbar.set_ticks([0, 5, 10, 15, 20])
cbar.ax.tick_params(labelsize=24)

plt.savefig("fig7c2.pdf", bbox_inches='tight')
