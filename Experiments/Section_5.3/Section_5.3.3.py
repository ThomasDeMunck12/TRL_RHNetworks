# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:41:12 2024

@author: ThomasDM
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

q_PPPO_6 = np.load("PPPO_simulations/instance_6.0/q_PPPO.npy")
q_PPPO_5 = np.load("PPPO_simulations/instance_5.0/q_PPPO.npy")

q_PPPO_6_mean = np.mean(q_PPPO_6, axis=(2,3))
q_PPPO_5_mean = np.mean(q_PPPO_5, axis=(2,3))

mask_q_6 = q_PPPO_6_mean < 0.02
mask_q_5 = q_PPPO_5_mean < 0.02

fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(12, 9))

sns.heatmap(q_PPPO_6_mean*100, cmap="Blues", ax=axes, annot=True, linewidths=1, linecolor='black', fmt='.0f', mask = mask_q_6, vmin = 0.0, vmax = 20.0, xticklabels=[1,2,3,4,5,6,7,8], yticklabels=[1,2,3,4,5,6,7,8])
for t in axes.texts: t.set_text(t.get_text() + " %")
axes.set_ylabel('Origin', fontsize=20)
axes.set_xlabel('Destination', fontsize=20)
axes.tick_params(axis='both', which='major', labelsize=20)
axes.tick_params(axis='both', which='minor', labelsize=20)

plt.savefig("fig7a_heatmap_admission.pdf", bbox_inches='tight')

fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(12, 9))
sns.set(font_scale=1.5)
sns.heatmap(q_PPPO_5_mean*100, cmap="Blues", ax=axes, annot=True, fmt='.0f', linewidths=1, linecolor='black', mask = mask_q_5, vmin = 0.0, vmax = 20.0, xticklabels=[1,2,3,4,5,6,7,8], yticklabels=[1,2,3,4,5,6,7,8])
for t in axes.texts: t.set_text(t.get_text() + " %")
axes.set_ylabel('Origin', fontsize=20)
axes.set_xlabel('Destination', fontsize=20)
axes.tick_params(axis='both', which='major', labelsize=20)
axes.tick_params(axis='both', which='minor', labelsize=20)

plt.savefig("fig7b_heatmap_admission.pdf", bbox_inches='tight')
