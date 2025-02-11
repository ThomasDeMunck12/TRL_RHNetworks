# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:31:54 2025

@author: ThomasDM
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
score_1_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_1.0.npy")
score_2_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_2.0.npy")
score_3_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_3.0.npy")
score_4_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_4.0.npy")
score_5_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_5.0.npy")
score_6_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_6.0.npy")
score_7_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_7.0.npy")
score_8_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_8.0.npy")
score_9_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_9.0.npy")
score_10_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_10.0.npy")
score_11_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_11.0.npy")
score_12_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_12.0.npy")
score_13_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_13.0.npy")
score_14_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_14.0.npy")
score_15_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_15.0.npy")
score_16_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_16.0.npy")
score_17_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_17.0.npy")
score_18_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_18.0.npy")
score_19_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_19.0.npy")
score_20_evp_1 = np.load("simulations_EVP_1_12/score_history_evp_20.0.npy")

score_1_pppo = np.load("simulations_PPPO/score_history_pppo_0.0.npy")
score_2_pppo = np.load("simulations_PPPO/score_history_pppo_1.0.npy")
score_3_pppo = np.load("simulations_PPPO/score_history_pppo_2.0.npy")
score_4_pppo = np.load("simulations_PPPO/score_history_pppo_3.0.npy")
score_5_pppo = np.load("simulations_PPPO/score_history_pppo_4.0.npy")
score_6_pppo = np.load("simulations_PPPO/score_history_pppo_5.0.npy")
score_7_pppo = np.load("simulations_PPPO/score_history_pppo_6.0.npy")
score_8_pppo = np.load("simulations_PPPO/score_history_pppo_7.0.npy")
score_9_pppo = np.load("simulations_PPPO/score_history_pppo_8.0.npy")
score_10_pppo = np.load("simulations_PPPO/score_history_pppo_9.0.npy")
score_11_pppo = np.load("simulations_PPPO/score_history_pppo_10.0.npy")
score_12_pppo = np.load("simulations_PPPO/score_history_pppo_11.0.npy")
score_13_pppo = np.load("simulations_PPPO/score_history_pppo_12.0.npy")
score_14_pppo = np.load("simulations_PPPO/score_history_pppo_13.0.npy")
score_15_pppo = np.load("simulations_PPPO/score_history_pppo_14.0.npy")
score_16_pppo = np.load("simulations_PPPO/score_history_pppo_15.0.npy")
score_17_pppo = np.load("simulations_PPPO/score_history_pppo_16.0.npy")
score_18_pppo = np.load("simulations_PPPO/score_history_pppo_17.0.npy")
score_19_pppo = np.load("simulations_PPPO/score_history_pppo_18.0.npy")
score_20_pppo = np.load("simulations_PPPO/score_history_pppo_19.0.npy")

requests_1_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_1.0.npy")
requests_2_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_2.0.npy")
requests_3_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_3.0.npy")
requests_4_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_4.0.npy")
requests_5_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_5.0.npy")
requests_6_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_6.0.npy")
requests_7_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_7.0.npy")
requests_8_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_8.0.npy")
requests_9_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_9.0.npy")
requests_10_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_10.0.npy")
requests_11_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_11.0.npy")
requests_12_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_12.0.npy")
requests_13_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_13.0.npy")
requests_14_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_14.0.npy")
requests_15_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_15.0.npy")
requests_16_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_16.0.npy")
requests_17_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_17.0.npy")
requests_18_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_18.0.npy")
requests_19_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_19.0.npy")
requests_20_evp_1 = np.load("simulations_evp_1_12/realized_requests_history_evp_20.0.npy")

requests_1_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_1.0.npy")
requests_2_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_2.0.npy")
requests_3_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_3.0.npy")
requests_4_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_4.0.npy")
requests_5_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_5.0.npy")
requests_6_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_6.0.npy")
requests_7_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_7.0.npy")
requests_8_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_8.0.npy")
requests_9_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_9.0.npy")
requests_10_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_10.0.npy")
requests_11_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_11.0.npy")
requests_12_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_12.0.npy")
requests_13_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_13.0.npy")
requests_14_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_14.0.npy")
requests_15_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_15.0.npy")
requests_16_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_16.0.npy")
requests_17_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_17.0.npy")
requests_18_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_18.0.npy")
requests_19_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_19.0.npy")
requests_20_pppo = np.load("simulations_PPPO/realized_requests_history_pppo_20.0.npy")

dispatch_1_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_1.0.npy")
dispatch_2_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_2.0.npy")
dispatch_3_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_3.0.npy")
dispatch_4_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_4.0.npy")
dispatch_5_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_5.0.npy")
dispatch_6_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_6.0.npy")
dispatch_7_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_7.0.npy")
dispatch_8_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_8.0.npy")
dispatch_9_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_9.0.npy")
dispatch_10_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_10.0.npy")
dispatch_11_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_11.0.npy")
dispatch_12_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_12.0.npy")
dispatch_13_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_13.0.npy")
dispatch_14_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_14.0.npy")
dispatch_15_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_15.0.npy")
dispatch_16_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_16.0.npy")
dispatch_17_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_17.0.npy")
dispatch_18_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_18.0.npy")
dispatch_19_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_19.0.npy")
dispatch_20_evp_1 = np.load("simulations_evp_1_12/dispatched_drivers_history_evp_20.0.npy")

dispatch_1_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_1.0.npy")
dispatch_2_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_2.0.npy")
dispatch_3_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_3.0.npy")
dispatch_4_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_4.0.npy")
dispatch_5_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_5.0.npy")
dispatch_6_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_6.0.npy")
dispatch_7_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_7.0.npy")
dispatch_8_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_8.0.npy")
dispatch_9_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_9.0.npy")
dispatch_10_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_10.0.npy")
dispatch_11_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_11.0.npy")
dispatch_12_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_12.0.npy")
dispatch_13_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_13.0.npy")
dispatch_14_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_14.0.npy")
dispatch_15_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_15.0.npy")
dispatch_16_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_16.0.npy")
dispatch_17_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_17.0.npy")
dispatch_18_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_18.0.npy")
dispatch_19_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_19.0.npy")
dispatch_20_pppo = np.load("simulations_PPPO/dispatched_drivers_history_pppo_20.0.npy")

number_drivers_1_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_1.0.npy")
number_drivers_2_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_2.0.npy")
number_drivers_3_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_3.0.npy")
number_drivers_4_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_4.0.npy")
number_drivers_5_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_5.0.npy")
number_drivers_6_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_6.0.npy")
number_drivers_7_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_7.0.npy")
number_drivers_8_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_8.0.npy")
number_drivers_9_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_9.0.npy")
number_drivers_10_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_10.0.npy")
number_drivers_11_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_11.0.npy")
number_drivers_12_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_12.0.npy")
number_drivers_13_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_13.0.npy")
number_drivers_14_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_14.0.npy")
number_drivers_15_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_15.0.npy")
number_drivers_16_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_16.0.npy")
number_drivers_17_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_17.0.npy")
number_drivers_18_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_18.0.npy")
number_drivers_19_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_19.0.npy")
number_drivers_20_pppo = np.load("simulations_PPPO/number_drivers_history_pppo_20.0.npy")

import seaborn as sns
import pandas as pd
from matplotlib import rcParams

rcParams['figure.figsize'] = 20.0,8.0
rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = 'dejavusans'
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
#Profit
data = {
    'Day': ['1']*50 + ['2']*50 + ['3']*50 + ['4']*50 + ['5']*50 + ['6']*50 + ['7']*50 + ['8']*50 + ['9']*50 + ['10']*50 
    + ['11']*50 + ['12']*50 + ['13']*50 + ['14']*50 + ['15']*50 + ['16']*50 + ['17']*50 + ['18']*50 + ['19']*50 + ['20']*50
    + ['1']*50 + ['2']*50 + ['3']*50 + ['4']*50 + ['5']*50 + ['6']*50 + ['7']*50 + ['8']*50 + ['9']*50 + ['10']*50
    + ['11']*50 + ['12']*50 + ['13']*50 + ['14']*50 + ['15']*50 + ['16']*50 + ['17']*50 + ['18']*50 + ['19']*50 + ['20']*50,
    'Total Profit': np.concatenate([score_1_pppo, score_2_pppo, score_3_pppo, score_4_pppo, score_5_pppo, score_6_pppo, score_7_pppo, score_8_pppo, score_9_pppo, score_10_pppo,
    score_11_pppo, score_12_pppo, score_13_pppo, score_14_pppo, score_15_pppo, score_16_pppo, score_17_pppo, score_18_pppo, score_19_pppo, score_20_pppo,
        score_1_evp_1, score_2_evp_1, score_3_evp_1, score_4_evp_1, score_5_evp_1, score_6_evp_1, score_7_evp_1, score_8_evp_1, score_9_evp_1, score_10_evp_1,
                              score_11_evp_1, score_12_evp_1, score_13_evp_1, score_14_evp_1, score_15_evp_1, score_16_evp_1, score_17_evp_1, score_18_evp_1, score_19_evp_1, score_20_evp_1])
}

df = pd.DataFrame(data)

average_score_PPO = np.mean(np.concatenate([score_1_pppo, score_2_pppo, score_3_pppo, score_4_pppo, score_5_pppo, score_6_pppo, score_7_pppo, score_8_pppo, score_9_pppo, score_10_pppo,
score_11_pppo, score_12_pppo, score_13_pppo, score_14_pppo, score_15_pppo, score_16_pppo, score_17_pppo, score_18_pppo, score_19_pppo, score_20_pppo]))

average_score_EVP_1 = np.mean(np.concatenate([score_1_evp_1, score_2_evp_1, score_3_evp_1, score_4_evp_1, score_5_evp_1, score_6_evp_1, score_7_evp_1, score_8_evp_1, score_9_evp_1, score_10_evp_1,
                      score_11_evp_1, score_12_evp_1, score_13_evp_1, score_14_evp_1, score_15_evp_1, score_16_evp_1, score_17_evp_1, score_18_evp_1, score_19_evp_1, score_20_evp_1]))

difference_score = (average_score_PPO-average_score_EVP_1)/average_score_EVP_1
# Create additional grouping data
df['Method'] = np.concatenate([['Our approach']*50*20, [r'RH with $T^{RH}=1,\, Q^{RH}=12$']*50*20])

# Plots graph
fig9a, ax = plt.subplots(figsize=[18,8])
sns.boxplot(x='Day', y='Total Profit', data=df, hue='Method', palette=['blue','green'])
ax.set_ylabel("TP ($)",fontsize=20)
ax.set_xlabel("Days",fontsize=20)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_xticklabels(range(1, 21), size = 14)
#fig9a.legend_.set_title(None)

#fig9a.subplots_adjust(right=0.9)  # Expands right side of the figure

lgd = ax.legend(fontsize=18, title_fontsize=10, bbox_to_anchor=(1.3, 1.0))

# figure size in inches
fig9a.savefig("fig9.pdf", bbox_extra_artists=[lgd], bbox_inches='tight')

#Utilization rates

average_time = np.load("average_trip_durations_20loc.npy")
average_time = np.floor(average_time)
tau = 1/average_time

distance = np.load("average_distances_20loc.npy")

mu = np.load("entry_rates_20loc.npy")

number_drivers = np.mean(number_drivers_1_pppo, axis=0)

T = 72 

load_1_pppo = np.zeros([50, 72])
load_2_pppo = np.zeros([50, 72])
load_3_pppo = np.zeros([50, 72])
load_4_pppo = np.zeros([50, 72])
load_5_pppo = np.zeros([50, 72])
load_6_pppo = np.zeros([50, 72])
load_7_pppo = np.zeros([50, 72])
load_8_pppo = np.zeros([50, 72])
load_9_pppo = np.zeros([50, 72])
load_10_pppo = np.zeros([50, 72])
load_11_pppo = np.zeros([50, 72])
load_12_pppo = np.zeros([50, 72])
load_13_pppo = np.zeros([50, 72])
load_14_pppo = np.zeros([50, 72])
load_15_pppo = np.zeros([50, 72])
load_16_pppo = np.zeros([50, 72])
load_17_pppo = np.zeros([50, 72])
load_18_pppo = np.zeros([50, 72])
load_19_pppo = np.zeros([50, 72])
load_20_pppo = np.zeros([50, 72])

load_1_evp_1 = np.zeros([50, 72])
load_2_evp_1 = np.zeros([50, 72])
load_3_evp_1 = np.zeros([50, 72])
load_4_evp_1 = np.zeros([50, 72])
load_5_evp_1 = np.zeros([50, 72])
load_6_evp_1 = np.zeros([50, 72])
load_7_evp_1 = np.zeros([50, 72])
load_8_evp_1 = np.zeros([50, 72])
load_9_evp_1 = np.zeros([50, 72])
load_10_evp_1 = np.zeros([50, 72])
load_11_evp_1 = np.zeros([50, 72])
load_12_evp_1 = np.zeros([50, 72])
load_13_evp_1 = np.zeros([50, 72])
load_14_evp_1 = np.zeros([50, 72])
load_15_evp_1 = np.zeros([50, 72])
load_16_evp_1 = np.zeros([50, 72])
load_17_evp_1 = np.zeros([50, 72])
load_18_evp_1 = np.zeros([50, 72])
load_19_evp_1 = np.zeros([50, 72])
load_20_evp_1 = np.zeros([50, 72])

for i in range(50):
    for j in range(72): 
        load_1_evp_1[i, j] = np.sum(dispatch_1_evp_1[i,:, :, j] / tau)
        load_2_evp_1[i, j] = np.sum(dispatch_2_evp_1[i,:, :, j] / tau)
        load_3_evp_1[i, j] = np.sum(dispatch_3_evp_1[i,:, :, j] / tau)
        load_4_evp_1[i, j] = np.sum(dispatch_4_evp_1[i,:, :, j] / tau)
        load_5_evp_1[i, j] = np.sum(dispatch_5_evp_1[i,:, :, j] / tau)
        load_6_evp_1[i, j] = np.sum(dispatch_6_evp_1[i,:, :, j] / tau)
        load_7_evp_1[i, j] = np.sum(dispatch_7_evp_1[i,:, :, j] / tau)
        load_8_evp_1[i, j] = np.sum(dispatch_8_evp_1[i,:, :, j] / tau)
        load_9_evp_1[i, j] = np.sum(dispatch_9_evp_1[i,:, :, j] / tau)
        load_10_evp_1[i, j] = np.sum(dispatch_10_evp_1[i, :,:, j] / tau)
        load_11_evp_1[i, j] = np.sum(dispatch_11_evp_1[i, :,:, j] / tau)
        load_12_evp_1[i, j] = np.sum(dispatch_12_evp_1[i, :,:, j] / tau)
        load_13_evp_1[i, j] = np.sum(dispatch_13_evp_1[i, :,:, j] / tau)
        load_14_evp_1[i, j] = np.sum(dispatch_14_evp_1[i, :,:, j] / tau)
        load_15_evp_1[i, j] = np.sum(dispatch_15_evp_1[i, :,:, j] / tau)
        load_16_evp_1[i, j] = np.sum(dispatch_16_evp_1[i, :,:, j] / tau)
        load_17_evp_1[i, j] = np.sum(dispatch_17_evp_1[i, :,:, j] / tau)
        load_18_evp_1[i, j] = np.sum(dispatch_18_evp_1[i, :,:, j] / tau)
        load_19_evp_1[i, j] = np.sum(dispatch_19_evp_1[i, :,:, j] / tau)
        load_20_evp_1[i, j] = np.sum(dispatch_20_evp_1[i, :,:, j] / tau)
        
        load_1_pppo[i, j] = np.sum(dispatch_1_pppo[i, :, :, j] / tau)
        load_2_pppo[i, j] = np.sum(dispatch_2_pppo[i, :, :, j] / tau)
        load_3_pppo[i, j] = np.sum(dispatch_3_pppo[i, :, :, j] / tau)
        load_4_pppo[i, j] = np.sum(dispatch_4_pppo[i, :, :, j] / tau)
        load_5_pppo[i, j] = np.sum(dispatch_5_pppo[i, :, :, j] / tau)
        load_6_pppo[i, j] = np.sum(dispatch_6_pppo[i, :, :, j] / tau)
        load_7_pppo[i, j] = np.sum(dispatch_7_pppo[i, :, :, j] / tau)
        load_8_pppo[i, j] = np.sum(dispatch_8_pppo[i, :, :, j] / tau)
        load_9_pppo[i, j] = np.sum(dispatch_9_pppo[i, :, :, j] / tau)
        load_10_pppo[i, j] = np.sum(dispatch_10_pppo[i, :, :, j] / tau)
        load_11_pppo[i, j] = np.sum(dispatch_11_pppo[i, :, :, j] / tau)
        load_12_pppo[i, j] = np.sum(dispatch_12_pppo[i, :, :, j] / tau)
        load_13_pppo[i, j] = np.sum(dispatch_13_pppo[i, :, :, j] / tau)
        load_14_pppo[i, j] = np.sum(dispatch_14_pppo[i, :, :, j] / tau)
        load_15_pppo[i, j] = np.sum(dispatch_15_pppo[i, :, :, j] / tau)
        load_16_pppo[i, j] = np.sum(dispatch_16_pppo[i, :, :, j] / tau)
        load_17_pppo[i, j] = np.sum(dispatch_17_pppo[i, :, :, j] / tau)
        load_18_pppo[i, j] = np.sum(dispatch_18_pppo[i, :, :, j] / tau)
        load_19_pppo[i, j] = np.sum(dispatch_19_pppo[i, :, :, j] / tau)
        load_20_pppo[i, j] = np.sum(dispatch_20_pppo[i, :, :, j] / tau)

utilization_rates_1_evp_1 = load_1_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_2_evp_1 = load_2_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_3_evp_1 = load_3_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_4_evp_1 = load_4_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_5_evp_1 = load_5_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_6_evp_1 = load_6_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_7_evp_1 = load_7_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_8_evp_1 = load_8_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_9_evp_1 = load_9_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_10_evp_1 = load_10_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_11_evp_1 = load_11_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_12_evp_1 = load_12_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_13_evp_1 = load_13_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_14_evp_1 = load_14_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_15_evp_1 = load_15_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_16_evp_1 = load_16_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_17_evp_1 = load_17_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_18_evp_1 = load_18_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_19_evp_1 = load_19_evp_1 / np.mean(number_drivers_1_pppo)
utilization_rates_20_evp_1 = load_20_evp_1 / np.mean(number_drivers_1_pppo)

utilization_rates_1_pppo = load_1_pppo / number_drivers_1_pppo
utilization_rates_2_pppo = load_2_pppo / number_drivers_1_pppo
utilization_rates_3_pppo = load_3_pppo / number_drivers_1_pppo
utilization_rates_4_pppo = load_4_pppo / number_drivers_1_pppo
utilization_rates_5_pppo = load_5_pppo / number_drivers_1_pppo
utilization_rates_6_pppo = load_6_pppo / number_drivers_1_pppo
utilization_rates_7_pppo = load_7_pppo / number_drivers_1_pppo
utilization_rates_8_pppo = load_8_pppo / number_drivers_1_pppo
utilization_rates_9_pppo = load_9_pppo / number_drivers_1_pppo
utilization_rates_10_pppo = load_10_pppo / number_drivers_1_pppo
utilization_rates_11_pppo = load_11_pppo / number_drivers_1_pppo
utilization_rates_12_pppo = load_12_pppo / number_drivers_1_pppo
utilization_rates_13_pppo = load_13_pppo / number_drivers_1_pppo
utilization_rates_14_pppo = load_14_pppo / number_drivers_1_pppo
utilization_rates_15_pppo = load_15_pppo / number_drivers_1_pppo
utilization_rates_16_pppo = load_16_pppo / number_drivers_1_pppo
utilization_rates_17_pppo = load_17_pppo / number_drivers_1_pppo
utilization_rates_18_pppo = load_18_pppo / number_drivers_1_pppo
utilization_rates_19_pppo = load_19_pppo / number_drivers_1_pppo
utilization_rates_20_pppo = load_20_pppo / number_drivers_1_pppo

utilization_rate_1_pppo = np.mean(utilization_rates_1_pppo[:,12:60])
utilization_rate_2_pppo = np.mean(utilization_rates_2_pppo[:,12:60])
utilization_rate_3_pppo = np.mean(utilization_rates_3_pppo[:,12:60])
utilization_rate_4_pppo = np.mean(utilization_rates_4_pppo[:,12:60])
utilization_rate_5_pppo = np.mean(utilization_rates_5_pppo[:,12:60])
utilization_rate_6_pppo = np.mean(utilization_rates_6_pppo[:,12:60])
utilization_rate_7_pppo = np.mean(utilization_rates_7_pppo[:,12:60])
utilization_rate_8_pppo = np.mean(utilization_rates_8_pppo[:,12:60])
utilization_rate_9_pppo = np.mean(utilization_rates_9_pppo[:,12:60])
utilization_rate_10_pppo = np.mean(utilization_rates_10_pppo[:,12:60])
utilization_rate_11_pppo = np.mean(utilization_rates_11_pppo[:,12:60])
utilization_rate_12_pppo = np.mean(utilization_rates_12_pppo[:,12:60])
utilization_rate_13_pppo = np.mean(utilization_rates_13_pppo[:,12:60])
utilization_rate_14_pppo = np.mean(utilization_rates_14_pppo[:,12:60])
utilization_rate_15_pppo = np.mean(utilization_rates_15_pppo[:,12:60])
utilization_rate_16_pppo = np.mean(utilization_rates_16_pppo[:,12:60])
utilization_rate_17_pppo = np.mean(utilization_rates_17_pppo[:,12:60])
utilization_rate_18_pppo = np.mean(utilization_rates_18_pppo[:,12:60])
utilization_rate_19_pppo = np.mean(utilization_rates_19_pppo[:,12:60])
utilization_rate_20_pppo = np.mean(utilization_rates_20_pppo[:,12:60])

utilization_rate_1_evp_1 = np.mean(utilization_rates_1_evp_1[:,12:60])
utilization_rate_2_evp_1 = np.mean(utilization_rates_2_evp_1[:,12:60])
utilization_rate_3_evp_1 = np.mean(utilization_rates_3_evp_1[:,12:60])
utilization_rate_4_evp_1 = np.mean(utilization_rates_4_evp_1[:,12:60])
utilization_rate_5_evp_1 = np.mean(utilization_rates_5_evp_1[:,12:60])
utilization_rate_6_evp_1 = np.mean(utilization_rates_6_evp_1[:,12:60])
utilization_rate_7_evp_1 = np.mean(utilization_rates_7_evp_1[:,12:60])
utilization_rate_8_evp_1 = np.mean(utilization_rates_8_evp_1[:,12:60])
utilization_rate_9_evp_1 = np.mean(utilization_rates_9_evp_1[:,12:60])
utilization_rate_10_evp_1 = np.mean(utilization_rates_10_evp_1[:,12:60])
utilization_rate_11_evp_1 = np.mean(utilization_rates_11_evp_1[:,12:60])
utilization_rate_12_evp_1 = np.mean(utilization_rates_12_evp_1[:,12:60])
utilization_rate_13_evp_1 = np.mean(utilization_rates_13_evp_1[:,12:60])
utilization_rate_14_evp_1 = np.mean(utilization_rates_14_evp_1[:,12:60])
utilization_rate_15_evp_1 = np.mean(utilization_rates_15_evp_1[:,12:60])
utilization_rate_16_evp_1 = np.mean(utilization_rates_16_evp_1[:,12:60])
utilization_rate_17_evp_1 = np.mean(utilization_rates_17_evp_1[:,12:60])
utilization_rate_18_evp_1 = np.mean(utilization_rates_18_evp_1[:,12:60])
utilization_rate_19_evp_1 = np.mean(utilization_rates_19_evp_1[:,12:60])
utilization_rate_20_evp_1 = np.mean(utilization_rates_20_evp_1[:,12:60])

ur = np.array([[utilization_rate_1_pppo, utilization_rate_2_pppo, utilization_rate_3_pppo, utilization_rate_4_pppo, utilization_rate_5_pppo, utilization_rate_6_pppo, utilization_rate_7_pppo, utilization_rate_8_pppo, utilization_rate_9_pppo, utilization_rate_10_pppo, utilization_rate_11_pppo, utilization_rate_12_pppo, utilization_rate_13_pppo, utilization_rate_14_pppo, utilization_rate_15_pppo, utilization_rate_16_pppo, utilization_rate_17_pppo, utilization_rate_18_pppo, utilization_rate_19_pppo, utilization_rate_20_pppo],
    [utilization_rate_1_evp_1, utilization_rate_2_evp_1, utilization_rate_3_evp_1, utilization_rate_4_evp_1, utilization_rate_5_evp_1, utilization_rate_6_evp_1, utilization_rate_7_evp_1, utilization_rate_8_evp_1, utilization_rate_9_evp_1, utilization_rate_10_evp_1, utilization_rate_11_evp_1, utilization_rate_12_evp_1, utilization_rate_13_evp_1, utilization_rate_14_evp_1, utilization_rate_15_evp_1, utilization_rate_16_evp_1, utilization_rate_17_evp_1, utilization_rate_18_evp_1, utilization_rate_19_evp_1, utilization_rate_20_evp_1]])

#Fulfillment rate

fulfil_rate_1_evp_1 = np.mean(np.sum(dispatch_1_evp_1, axis = (1,2,3))/np.sum(requests_1_evp_1, axis = (1,2,3)))
fulfil_rate_2_evp_1 = np.mean(np.sum(dispatch_2_evp_1, axis = (1,2,3))/np.sum(requests_2_evp_1, axis = (1,2,3)))
fulfil_rate_3_evp_1 = np.mean(np.sum(dispatch_3_evp_1, axis = (1,2,3))/np.sum(requests_3_evp_1, axis = (1,2,3)))
fulfil_rate_4_evp_1 = np.mean(np.sum(dispatch_4_evp_1, axis = (1,2,3))/np.sum(requests_4_evp_1, axis = (1,2,3)))
fulfil_rate_5_evp_1 = np.mean(np.sum(dispatch_5_evp_1, axis = (1,2,3))/np.sum(requests_5_evp_1, axis = (1,2,3)))
fulfil_rate_6_evp_1 = np.mean(np.sum(dispatch_6_evp_1, axis = (1,2,3))/np.sum(requests_6_evp_1, axis = (1,2,3)))
fulfil_rate_7_evp_1 = np.mean(np.sum(dispatch_7_evp_1, axis = (1,2,3))/np.sum(requests_7_evp_1, axis = (1,2,3)))
fulfil_rate_8_evp_1 = np.mean(np.sum(dispatch_8_evp_1, axis = (1,2,3))/np.sum(requests_8_evp_1, axis = (1,2,3)))
fulfil_rate_9_evp_1 = np.mean(np.sum(dispatch_9_evp_1, axis = (1,2,3))/np.sum(requests_9_evp_1, axis = (1,2,3)))
fulfil_rate_10_evp_1 = np.mean(np.sum(dispatch_10_evp_1, axis = (1,2,3))/np.sum(requests_10_evp_1, axis = (1,2,3)))
fulfil_rate_11_evp_1 = np.mean(np.sum(dispatch_11_evp_1, axis = (1,2,3))/np.sum(requests_11_evp_1, axis = (1,2,3)))
fulfil_rate_12_evp_1 = np.mean(np.sum(dispatch_12_evp_1, axis = (1,2,3))/np.sum(requests_12_evp_1, axis = (1,2,3)))
fulfil_rate_13_evp_1 = np.mean(np.sum(dispatch_13_evp_1, axis = (1,2,3))/np.sum(requests_13_evp_1, axis = (1,2,3)))
fulfil_rate_14_evp_1 = np.mean(np.sum(dispatch_14_evp_1, axis = (1,2,3))/np.sum(requests_14_evp_1, axis = (1,2,3)))
fulfil_rate_15_evp_1 = np.mean(np.sum(dispatch_15_evp_1, axis = (1,2,3))/np.sum(requests_15_evp_1, axis = (1,2,3)))
fulfil_rate_16_evp_1 = np.mean(np.sum(dispatch_16_evp_1, axis = (1,2,3))/np.sum(requests_16_evp_1, axis = (1,2,3)))
fulfil_rate_17_evp_1 = np.mean(np.sum(dispatch_17_evp_1, axis = (1,2,3))/np.sum(requests_17_evp_1, axis = (1,2,3)))
fulfil_rate_18_evp_1 = np.mean(np.sum(dispatch_18_evp_1, axis = (1,2,3))/np.sum(requests_18_evp_1, axis = (1,2,3)))
fulfil_rate_19_evp_1 = np.mean(np.sum(dispatch_19_evp_1, axis = (1,2,3))/np.sum(requests_19_evp_1, axis = (1,2,3)))
fulfil_rate_20_evp_1 = np.mean(np.sum(dispatch_20_evp_1, axis = (1,2,3))/np.sum(requests_20_evp_1, axis = (1,2,3)))

fulfil_rate_1_pppo = np.mean(np.sum(dispatch_1_pppo, axis = (1,2,3))/np.sum(requests_1_pppo, axis = (1,2,3)))
fulfil_rate_2_pppo = np.mean(np.sum(dispatch_2_pppo, axis = (1,2,3))/np.sum(requests_2_pppo, axis = (1,2,3)))
fulfil_rate_3_pppo = np.mean(np.sum(dispatch_3_pppo, axis = (1,2,3))/np.sum(requests_3_pppo, axis = (1,2,3)))
fulfil_rate_4_pppo = np.mean(np.sum(dispatch_4_pppo, axis = (1,2,3))/np.sum(requests_4_pppo, axis = (1,2,3)))
fulfil_rate_5_pppo = np.mean(np.sum(dispatch_5_pppo, axis = (1,2,3))/np.sum(requests_5_pppo, axis = (1,2,3)))
fulfil_rate_6_pppo = np.mean(np.sum(dispatch_6_pppo, axis = (1,2,3))/np.sum(requests_6_pppo, axis = (1,2,3)))
fulfil_rate_7_pppo = np.mean(np.sum(dispatch_7_pppo, axis = (1,2,3))/np.sum(requests_7_pppo, axis = (1,2,3)))
fulfil_rate_8_pppo = np.mean(np.sum(dispatch_8_pppo, axis = (1,2,3))/np.sum(requests_8_pppo, axis = (1,2,3)))
fulfil_rate_9_pppo = np.mean(np.sum(dispatch_9_pppo, axis = (1,2,3))/np.sum(requests_9_pppo, axis = (1,2,3)))
fulfil_rate_10_pppo = np.mean(np.sum(dispatch_10_pppo, axis = (1,2,3))/np.sum(requests_10_pppo, axis = (1,2,3)))
fulfil_rate_11_pppo = np.mean(np.sum(dispatch_11_pppo, axis = (1,2,3))/np.sum(requests_11_pppo, axis = (1,2,3)))
fulfil_rate_12_pppo = np.mean(np.sum(dispatch_12_pppo, axis = (1,2,3))/np.sum(requests_12_pppo, axis = (1,2,3)))
fulfil_rate_13_pppo = np.mean(np.sum(dispatch_13_pppo, axis = (1,2,3))/np.sum(requests_13_pppo, axis = (1,2,3)))
fulfil_rate_14_pppo = np.mean(np.sum(dispatch_14_pppo, axis = (1,2,3))/np.sum(requests_14_pppo, axis = (1,2,3)))
fulfil_rate_15_pppo = np.mean(np.sum(dispatch_15_pppo, axis = (1,2,3))/np.sum(requests_15_pppo, axis = (1,2,3)))
fulfil_rate_16_pppo = np.mean(np.sum(dispatch_16_pppo, axis = (1,2,3))/np.sum(requests_16_pppo, axis = (1,2,3)))
fulfil_rate_17_pppo = np.mean(np.sum(dispatch_17_pppo, axis = (1,2,3))/np.sum(requests_17_pppo, axis = (1,2,3)))
fulfil_rate_18_pppo = np.mean(np.sum(dispatch_18_pppo, axis = (1,2,3))/np.sum(requests_18_pppo, axis = (1,2,3)))
fulfil_rate_19_pppo = np.mean(np.sum(dispatch_19_pppo, axis = (1,2,3))/np.sum(requests_19_pppo, axis = (1,2,3)))
fulfil_rate_20_pppo = np.mean(np.sum(dispatch_20_pppo, axis = (1,2,3))/np.sum(requests_20_pppo, axis = (1,2,3)))

fr = np.array([[fulfil_rate_1_pppo, fulfil_rate_2_pppo, fulfil_rate_3_pppo, fulfil_rate_4_pppo, fulfil_rate_5_pppo, fulfil_rate_6_pppo, fulfil_rate_7_pppo, fulfil_rate_8_pppo, fulfil_rate_9_pppo, fulfil_rate_10_pppo,
fulfil_rate_11_pppo, fulfil_rate_12_pppo, fulfil_rate_13_pppo, fulfil_rate_14_pppo, fulfil_rate_15_pppo, fulfil_rate_16_pppo, fulfil_rate_17_pppo, fulfil_rate_18_pppo, fulfil_rate_19_pppo, fulfil_rate_20_pppo],
    [fulfil_rate_1_evp_1, fulfil_rate_2_evp_1, fulfil_rate_3_evp_1, fulfil_rate_4_evp_1, fulfil_rate_5_evp_1, fulfil_rate_6_evp_1, fulfil_rate_7_evp_1, fulfil_rate_8_evp_1, fulfil_rate_9_evp_1, fulfil_rate_10_evp_1,
                          fulfil_rate_11_evp_1, fulfil_rate_12_evp_1, fulfil_rate_13_evp_1, fulfil_rate_14_evp_1, fulfil_rate_15_evp_1, fulfil_rate_16_evp_1, fulfil_rate_17_evp_1, fulfil_rate_18_evp_1, fulfil_rate_19_evp_1, fulfil_rate_20_evp_1]])
#Fulfillment rate


x = np.arange(1, 21)  # Days 1 to 20

# Create figure and axis
fig, ax = plt.subplots(figsize=(18, 8))  # Define figure size

# Plot both datasets with markers only (no lines)
ax.plot(x, fr[0], label='Our approach (FR)', marker='o',markersize=12,ls='', color='blue')  # Use 'ls' instead of 'linestyle'
ax.plot(x, fr[1], label=r'RH with $T^{RH}=1,\, Q^{RH}=12$ (FR)', marker='s',markersize=12,ls='', color='green')
ax.plot(x, ur[0], label='Our approach (UR)', marker='+',markersize=12,ls='', color='blue')  # Use 'ls' instead of 'linestyle'
ax.plot(x, ur[1], label=r'RH with $T^{RH}=1,\, Q^{RH}=12$ (UR)', markersize=12, marker='x',ls='', color='green')

# Set labels
ax.set_ylabel("UR(%)                                   FR (%)", fontsize=20)
ax.set_xlabel("Days", fontsize=20)

# Set y-ticks correctly
yticks = [0.5,0.6, 0.7, 0.8, 0.9, 1.0]
ax.set_yticks(yticks)  
ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=14)

# Set x-ticks correctly
ax.set_xticks(x)  
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

lgd = ax.legend(fontsize=18, title_fontsize=10, bbox_to_anchor=(1.0, 1.0))

# Show the plot
fig.savefig("fig10.pdf", bbox_extra_artists=[lgd], bbox_inches='tight')

average_fr_PPPO = np.mean([fulfil_rate_1_pppo, fulfil_rate_2_pppo, fulfil_rate_3_pppo, fulfil_rate_4_pppo, fulfil_rate_5_pppo, fulfil_rate_6_pppo, fulfil_rate_7_pppo, fulfil_rate_8_pppo, fulfil_rate_9_pppo, fulfil_rate_10_pppo,
fulfil_rate_11_pppo, fulfil_rate_12_pppo, fulfil_rate_13_pppo, fulfil_rate_14_pppo, fulfil_rate_15_pppo, fulfil_rate_16_pppo, fulfil_rate_17_pppo, fulfil_rate_18_pppo, fulfil_rate_19_pppo, fulfil_rate_20_pppo])

average_fr_EVP_1 = np.mean([fulfil_rate_1_evp_1, fulfil_rate_2_evp_1, fulfil_rate_3_evp_1, fulfil_rate_4_evp_1, fulfil_rate_5_evp_1, fulfil_rate_6_evp_1, fulfil_rate_7_evp_1, fulfil_rate_8_evp_1, fulfil_rate_9_evp_1, fulfil_rate_10_evp_1,
                      fulfil_rate_11_evp_1, fulfil_rate_12_evp_1, fulfil_rate_13_evp_1, fulfil_rate_14_evp_1, fulfil_rate_15_evp_1, fulfil_rate_16_evp_1, fulfil_rate_17_evp_1, fulfil_rate_18_evp_1, fulfil_rate_19_evp_1, fulfil_rate_20_evp_1])

difference_fr = (average_fr_PPPO-average_fr_EVP_1)/average_fr_EVP_1


