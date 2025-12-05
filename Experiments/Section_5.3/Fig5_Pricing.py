# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 23:51:19 2025

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt

Lambda_day = np.load("../../Data/arrival_rates_8loc.npy")
Lambda = Lambda_day[:, :, 84:120]
Lambda_weights = np.sum(Lambda, axis=2)
#Data from simulation

Lambda_weighted = Lambda_weights[:, :, None, None]              

p_PPPO_1 = np.load("HPPPO_simulations_MC/instance_1.0/p_PPPO.npy") 
p_PPPO_1_masked = np.ma.masked_equal(p_PPPO_1, 0.0)
weighted_sum_1 = np.sum(p_PPPO_1 * Lambda_weighted, axis=1)   
normalizer_1 = np.sum(Lambda_weighted, axis=1)       
p_PPPO_1_weighted = weighted_sum_1 / normalizer_1 

p_PPPO_3 = np.load("HPPPO_simulations_MC/instance_3.0/p_PPPO.npy") 
p_PPPO_3_masked = np.ma.masked_equal(p_PPPO_3, 0.0)
weighted_sum_3 = np.sum(p_PPPO_3 * Lambda_weighted, axis=1)   
normalizer_3 = np.sum(Lambda_weighted, axis=1)       
p_PPPO_3_weighted = weighted_sum_3 / normalizer_3 

p_PPPO_5 = np.load("HPPPO_simulations_MC/instance_5.0/p_PPPO.npy") 
p_PPPO_5_masked = np.ma.masked_equal(p_PPPO_5, 0.0)
weighted_sum_5 = np.sum(p_PPPO_5 * Lambda_weighted, axis=1)   
normalizer_5 = np.sum(Lambda_weighted, axis=1)       
p_PPPO_5_weighted = weighted_sum_5 / normalizer_5 

p_PPPO_1_av = np.mean(p_PPPO_1_weighted, axis=(2))
p_PPPO_3_av = np.mean(p_PPPO_3_weighted, axis=(2))
p_PPPO_5_av = np.mean(p_PPPO_5_weighted, axis=(2))

p_PPPO_1_sd = np.std(p_PPPO_1_weighted, axis=(2))
ci_p_1 = 1.96 * p_PPPO_1_sd 
p_PPPO_3_sd = np.std(p_PPPO_3_weighted, axis=(2))
ci_p_3 = 1.96 * p_PPPO_3_sd 
p_PPPO_5_sd = np.std(p_PPPO_5_weighted, axis=(2))
ci_p_5 = 1.96 * p_PPPO_5_sd 

p_PPPO_21 = np.load("HPPPO_simulations_MC/instance_1.0/p_PPPO.npy") 
p_PPPO_21_masked = np.ma.masked_equal(p_PPPO_21, 0.0)
weighted_sum_21 = np.sum(p_PPPO_21 * Lambda_weighted, axis=1)   
normalizer_21 = np.sum(Lambda_weighted, axis=1)       
p_PPPO_21_weighted = weighted_sum_21 / normalizer_21 

p_PPPO_22 = np.load("HPPPO_simulations_MC/instance_3.0/p_PPPO.npy") 
p_PPPO_22_masked = np.ma.masked_equal(p_PPPO_22, 0.0)
weighted_sum_22 = np.sum(p_PPPO_22 * Lambda_weighted, axis=1)   
normalizer_22 = np.sum(Lambda_weighted, axis=1)       
p_PPPO_22_weighted = weighted_sum_22 / normalizer_22 

p_PPPO_24 = np.load("HPPPO_simulations_MC/instance_5.0/p_PPPO.npy") 
p_PPPO_24_masked = np.ma.masked_equal(p_PPPO_24, 0.0)
weighted_sum_24 = np.sum(p_PPPO_24 * Lambda_weighted, axis=1)   
normalizer_24 = np.sum(Lambda_weighted, axis=1)       
p_PPPO_24_weighted = weighted_sum_24 / normalizer_24

p_PPPO_21_av = np.mean(p_PPPO_21_weighted, axis=(2))
p_PPPO_22_av = np.mean(p_PPPO_22_weighted, axis=(2))
p_PPPO_24_av = np.mean(p_PPPO_24_weighted, axis=(2))

p_PPPO_21_sd = np.std(p_PPPO_21_weighted, axis=(2))
ci_p_21 = 1.96 * p_PPPO_21_sd 
p_PPPO_22_sd = np.std(p_PPPO_22_weighted, axis=(2))
ci_p_22 = 1.96 * p_PPPO_22_sd 
p_PPPO_24_sd = np.std(p_PPPO_24_weighted, axis=(2))
ci_p_24 = 1.96 * p_PPPO_24_sd 

c_PPPO_1 = np.load("HPPPO_simulations_MC/instance_1.0/lambda_PPPO.npy") 
c_PPPO_3 = np.load("HPPPO_simulations_MC/instance_3.0/lambda_PPPO.npy") 
c_PPPO_5 = np.load("HPPPO_simulations_MC/instance_5.0/lambda_PPPO.npy") 

c_PPPO_1_av = np.mean(c_PPPO_1, axis=(3))
c_PPPO_3_av = np.mean(c_PPPO_3, axis=(3))
c_PPPO_5_av = np.mean(c_PPPO_5, axis=(3))

c_PPPO_1_sum = np.sum(c_PPPO_1_av, axis=1)
c_PPPO_3_sum= np.sum(c_PPPO_3_av, axis=1)
c_PPPO_5_sum = np.sum(c_PPPO_5_av, axis=1)

c_PPPO_1_fin = np.sum(c_PPPO_1, axis=1)
c_PPPO_3_fin = np.sum(c_PPPO_3, axis=1)
c_PPPO_5_fin = np.sum(c_PPPO_5, axis=1)

c_PPPO_1_sd = np.std(c_PPPO_1_fin, axis=2)
ci_c_1 = 1.96 * c_PPPO_1_sd 
c_PPPO_3_sd = np.std(c_PPPO_3_fin, axis=2)
ci_c_3 = 1.96 * c_PPPO_3_sd 
c_PPPO_5_sd = np.std(c_PPPO_5_fin, axis=2)
ci_c_5 = 1.96 * c_PPPO_5_sd 

#Fig 5.a
ncols = 1
nrows = 1

iterations = np.linspace(0, 35, 36)

fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(12,9))

axes.plot(iterations, p_PPPO_1_av[3,:], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=0.5$')
axes.plot(iterations, p_PPPO_3_av[3,:], marker='o', color='blue', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.0$')
axes.plot(iterations, p_PPPO_5_av[3,:], marker='s', linestyle = 'dashed', color='red', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.5$')
#axes.fill_between(iterations, (p_PPPO_1_av[3,:]-ci_p_1[3,:]), (p_PPPO_1_av[3,:]+ci_p_1[3,:]), color='green', alpha=.3)
#axes.fill_between(iterations, (p_PPPO_3_av[3,:]-ci_p_3[3,:]), (p_PPPO_3_av[3,:]+ci_p_3[3,:]), color='blue', alpha=.3)
#axes.fill_between(iterations, (p_PPPO_5_av[3,:]-ci_p_5[3,:]), (p_PPPO_5_av[3,:]+ci_p_5[3,:]), color='red', alpha=.3)

axes.errorbar(iterations, p_PPPO_1_av[3,:], yerr=ci_p_1[3,:], fmt='none', ecolor='green', capsize=5, elinewidth=1.5)  # Error bars for 0.5
axes.errorbar(iterations, p_PPPO_5_av[3,:], yerr=ci_p_5[3,:], fmt='none', ecolor='red', capsize=5, elinewidth=1.5)  # Error bars for 1.5
# Uncomment the next line if you want the error bars for the 1.0 case
axes.errorbar(iterations, p_PPPO_22_av[3,:], yerr=ci_p_22[3,:], fmt='none', ecolor='blue', capsize=5, elinewidth=1.5)

axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)
axes.set_xlabel("Decision Epochs", fontsize=24)
axes.set_ylabel("Prices", fontsize=24)
axes.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8], labels=[0.4, 0.5, 0.6, 0.7, 0.8])

axes.set_ylim((0.4,0.8))
axes.set_xlim((-0.5, 35.5))
axes.legend(loc='upper left', prop={'size': 24})      
plt.savefig("fig5a.pdf", bbox_inches='tight')

#Fig 5.b

fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(12,9))

axes.plot(iterations, np.sum(Lambda, axis=1)[3,:]*0.5, marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=0.5$')
axes.plot(iterations, np.sum(Lambda, axis=1)[3,:]*1.0, marker='o', color='blue', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.0$')
axes.plot(iterations, np.sum(Lambda, axis=1)[3,:]*1.5, marker='s', linestyle = 'dashed', color='red', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.5$')
axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)
axes.set_xlabel("Decision Epochs", fontsize=24)
axes.set_ylabel("Potential Customers", fontsize=24)
axes.set_ylim((0,250))

axes.set_xlim((-0.5, 35.5))
axes.legend(loc='upper left', prop={'size': 24})

plt.savefig("fig5b.pdf", bbox_inches='tight')

#Fig 5.c
fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(12,9))

axes.plot(iterations, c_PPPO_1_sum[3,:], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=0.5$')
axes.plot(iterations, c_PPPO_3_sum[3,:], marker='o', color='blue', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.0$')
axes.plot(iterations, c_PPPO_5_sum[3,:], marker='s', linestyle = 'dashed', color='red', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.5$')
#axes.fill_between(iterations, (lambda_PPPO_1_sum[3,:]-ci_lambda_1[3,:]), (lambda_PPPO_1_sum[3,:]+ci_lambda_1[3,:]), color='green', alpha=.3)
#axes.fill_between(iterations, (lambda_PPPO_3_sum[3,:]-ci_lambda_3[3,:]), (lambda_PPPO_3_sum[3,:]+ci_lambda_3[3,:]), color='blue', alpha=.3)
#axes.fill_between(iterations, (lambda_PPPO_5_sum[3,:]-ci_lambda_5[3,:]), (lambda_PPPO_5_sum[3,:]+ci_lambda_5[3,:]), color='red', alpha=.3)

axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)
axes.set_xlabel("Decision Epochs", fontsize=24)
axes.set_ylabel("Customer Requests", fontsize=24)
axes.set_ylim((0,250))

axes.set_xlim((-0.5, 35.5))
axes.legend(loc='upper left', prop={'size': 24})
plt.savefig("fig5c.pdf", bbox_inches='tight')


