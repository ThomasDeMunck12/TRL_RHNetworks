# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:43:09 2024

@author: ThomasDM
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string


Lambda_day = np.load("../../Data/arrival_rates_8loc.npy")
Lambda = Lambda_day[:, :, 84:120]

#Data from simulation

p_PPPO_1 = np.load("PPPO_simulations/instance_1.0/p_PPPO.npy") 
p_PPPO_2 = np.load("PPPO_simulations/instance_2.0/p_PPPO.npy") 
p_PPPO_3 = np.load("PPPO_simulations/instance_3.0/p_PPPO.npy") 
p_PPPO_4 = np.load("PPPO_simulations/instance_4.0/p_PPPO.npy") 
p_PPPO_5 = np.load("PPPO_simulations/instance_5.0/p_PPPO.npy") 


p_PPPO_1_av = np.mean(p_PPPO_1, axis=(2))
p_PPPO_2_av = np.mean(p_PPPO_2, axis=(2))
p_PPPO_3_av = np.mean(p_PPPO_3, axis=(2))
p_PPPO_4_av = np.mean(p_PPPO_4, axis=(2))
p_PPPO_5_av = np.mean(p_PPPO_5, axis=(2))

p_PPPO_1_sd = np.std(p_PPPO_1, axis=2)
ci_p_1 = 1.96 * p_PPPO_1_sd 
p_PPPO_2_sd = np.std(p_PPPO_2, axis=2)
ci_p_2 = 1.96 * p_PPPO_2_sd 
p_PPPO_3_sd = np.std(p_PPPO_3, axis=2)
ci_p_3 = 1.96 * p_PPPO_3_sd 
p_PPPO_4_sd = np.std(p_PPPO_4, axis=2)
ci_p_4 = 1.96 * p_PPPO_4_sd 
p_PPPO_5_sd = np.std(p_PPPO_5, axis=2)
ci_p_5 = 1.96 * p_PPPO_5_sd 

cv_p_1 = np.std(p_PPPO_1[:,1:36,:], axis=2)/np.mean(p_PPPO_1[:,1:36,:], axis=2)
cv_p_1 = np.mean(cv_p_1, axis=1)

cv_p_5 = np.std(p_PPPO_5[:,1:36,:], axis=2)/np.mean(p_PPPO_5[:,1:36,:], axis=2)
cv_p_5 = np.mean(cv_p_5, axis=1)

x_PPPO_1 = np.load("PPPO_simulations/instance_1.0/x_PPPO.npy") 
x_PPPO_2 = np.load("PPPO_simulations/instance_2.0/x_PPPO.npy") 
x_PPPO_3 = np.load("PPPO_simulations/instance_3.0/x_PPPO.npy") 
x_PPPO_4 = np.load("PPPO_simulations/instance_4.0/x_PPPO.npy") 
x_PPPO_5 = np.load("PPPO_simulations/instance_5.0/x_PPPO.npy") 


x_PPPO_1_av = np.mean(x_PPPO_1, axis=(2))
x_PPPO_2_av = np.mean(x_PPPO_2, axis=(2))
x_PPPO_3_av = np.mean(x_PPPO_3, axis=(2))
x_PPPO_4_av = np.mean(x_PPPO_4, axis=(2))
x_PPPO_5_av = np.mean(x_PPPO_5, axis=(2))


x_PPPO_1_sd = np.std(x_PPPO_1, axis=2)
ci_x_1 = 1.96 * x_PPPO_1_sd 
x_PPPO_2_sd = np.std(x_PPPO_2, axis=2)
ci_x_2 = 1.96 * x_PPPO_2_sd 
x_PPPO_3_sd = np.std(x_PPPO_3, axis=2)
ci_x_3 = 1.96 * x_PPPO_3_sd 
x_PPPO_4_sd = np.std(x_PPPO_4, axis=2)
ci_x_4 = 1.96 * x_PPPO_4_sd 
x_PPPO_5_sd = np.std(x_PPPO_5, axis=2)
ci_x_5 = 1.96 * x_PPPO_5_sd 

c_PPPO_1 = np.load("PPPO_simulations/instance_1.0/c_PPPO.npy") 
c_PPPO_2 = np.load("PPPO_simulations/instance_2.0/c_PPPO.npy") 
c_PPPO_3 = np.load("PPPO_simulations/instance_3.0/c_PPPO.npy") 
c_PPPO_4 = np.load("PPPO_simulations/instance_4.0/c_PPPO.npy") 
c_PPPO_5 = np.load("PPPO_simulations/instance_5.0/c_PPPO.npy") 

c_PPPO_1_av = np.mean(c_PPPO_1, axis=(3))
c_PPPO_2_av = np.mean(c_PPPO_2, axis=(3))
c_PPPO_3_av = np.mean(c_PPPO_3, axis=(3))
c_PPPO_4_av = np.mean(c_PPPO_4, axis=(3))
c_PPPO_5_av = np.mean(c_PPPO_5, axis=(3))

c_PPPO_1_sum = np.sum(c_PPPO_1_av, axis=1)
c_PPPO_2_sum = np.sum(c_PPPO_2_av, axis=1)
c_PPPO_3_sum= np.sum(c_PPPO_3_av, axis=1)
c_PPPO_4_sum = np.sum(c_PPPO_4_av, axis=1)
c_PPPO_5_sum = np.sum(c_PPPO_5_av, axis=1)

c_PPPO_1_fin = np.sum(c_PPPO_1, axis=1)
c_PPPO_2_fin = np.sum(c_PPPO_2, axis=1)
c_PPPO_3_fin = np.sum(c_PPPO_3, axis=1)
c_PPPO_4_fin = np.sum(c_PPPO_4, axis=1)
c_PPPO_5_fin = np.sum(c_PPPO_5, axis=1)

c_PPPO_1_sd = np.std(c_PPPO_1_fin, axis=2)
ci_c_1 = 1.96 * c_PPPO_1_sd 
c_PPPO_2_sd = np.std(c_PPPO_2_fin, axis=2)
ci_c_2 = 1.96 * c_PPPO_2_sd 
c_PPPO_3_sd = np.std(c_PPPO_3_fin, axis=2)
ci_c_3 = 1.96 * c_PPPO_3_sd 
c_PPPO_4_sd = np.std(c_PPPO_4_fin, axis=2)
ci_c_4 = 1.96 * c_PPPO_4_sd 
c_PPPO_5_sd = np.std(c_PPPO_5_fin, axis=2)
ci_c_5 = 1.96 * c_PPPO_5_sd 

cv_c_1 = np.std(c_PPPO_1_fin, axis=2)/c_PPPO_1_sum
cv_c_2 = np.std(c_PPPO_2_fin, axis=2)/c_PPPO_2_sum
cv_c_3 = np.std(c_PPPO_3_fin, axis=2)/c_PPPO_3_sum
cv_c_4 = np.std(c_PPPO_4_fin, axis=2)/c_PPPO_4_sum
cv_c_5 = np.std(c_PPPO_5_fin, axis=2)/c_PPPO_5_sum

lambda_PPPO_1 = np.load("PPPO_simulations/instance_1.0/lambda_PPPO.npy") 
lambda_PPPO_2 = np.load("PPPO_simulations/instance_2.0/lambda_PPPO.npy") 
lambda_PPPO_3 = np.load("PPPO_simulations/instance_3.0/lambda_PPPO.npy") 
lambda_PPPO_4 = np.load("PPPO_simulations/instance_4.0/lambda_PPPO.npy") 
lambda_PPPO_5 = np.load("PPPO_simulations/instance_5.0/lambda_PPPO.npy") 

lambda_PPPO_1_av = np.mean(lambda_PPPO_1, axis=(3))
lambda_PPPO_2_av = np.mean(lambda_PPPO_2, axis=(3))
lambda_PPPO_3_av = np.mean(lambda_PPPO_3, axis=(3))
lambda_PPPO_4_av = np.mean(lambda_PPPO_4, axis=(3))
lambda_PPPO_5_av = np.mean(lambda_PPPO_5, axis=(3))

lambda_PPPO_1_sum = np.sum(lambda_PPPO_1_av, axis=1)
lambda_PPPO_2_sum = np.sum(lambda_PPPO_2_av, axis=1)
lambda_PPPO_3_sum= np.sum(lambda_PPPO_3_av, axis=1)
lambda_PPPO_4_sum = np.sum(lambda_PPPO_4_av, axis=1)
lambda_PPPO_5_sum = np.sum(lambda_PPPO_5_av, axis=1)

lambda_PPPO_1_fin = np.sum(lambda_PPPO_1, axis=1)
lambda_PPPO_2_fin = np.sum(lambda_PPPO_2, axis=1)
lambda_PPPO_3_fin = np.sum(lambda_PPPO_3, axis=1)
lambda_PPPO_4_fin = np.sum(lambda_PPPO_4, axis=1)
lambda_PPPO_5_fin = np.sum(lambda_PPPO_5, axis=1)

lambda_PPPO_1_sd = np.std(lambda_PPPO_1_fin, axis=2)
ci_lambda_1 = 2.576 * lambda_PPPO_1_sd 
lambda_PPPO_2_sd = np.std(lambda_PPPO_2_fin, axis=2)
ci_lambda_2 = 2.576 * lambda_PPPO_2_sd 
lambda_PPPO_3_sd = np.std(lambda_PPPO_3_fin, axis=2)
ci_lambda_3 = 2.576 * lambda_PPPO_3_sd 
lambda_PPPO_4_sd = np.std(lambda_PPPO_4_fin, axis=2)
ci_lambda_4 = 2.576 * lambda_PPPO_4_sd 
lambda_PPPO_5_sd = np.std(lambda_PPPO_5_fin, axis=2)
ci_lambda_5 = 2.576 * lambda_PPPO_5_sd 

cv_lambda_1 = np.std(lambda_PPPO_1_fin, axis=2)/lambda_PPPO_1_sum
cv_lambda_2 = np.std(lambda_PPPO_2_fin, axis=2)/lambda_PPPO_2_sum
cv_lambda_3 = np.std(lambda_PPPO_3_fin, axis=2)/lambda_PPPO_3_sum
cv_lambda_4 = np.std(lambda_PPPO_4_fin, axis=2)/lambda_PPPO_4_sum
cv_lambda_5 = np.std(lambda_PPPO_5_fin, axis=2)/lambda_PPPO_5_sum

y_PPPO_1 = np.load("PPPO_simulations/instance_1.0/y_PPPO.npy") 
y_PPPO_2 = np.load("PPPO_simulations/instance_2.0/y_PPPO.npy") 
y_PPPO_3 = np.load("PPPO_simulations/instance_3.0/y_PPPO.npy") 
y_PPPO_4 = np.load("PPPO_simulations/instance_4.0/y_PPPO.npy") 
y_PPPO_5 = np.load("PPPO_simulations/instance_5.0/y_PPPO.npy") 

y_PPPO_1_av = np.mean(y_PPPO_1, axis=(3))
y_PPPO_2_av = np.mean(y_PPPO_2, axis=(3))
y_PPPO_3_av = np.mean(y_PPPO_3, axis=(3))
y_PPPO_4_av = np.mean(y_PPPO_4, axis=(3))
y_PPPO_5_av = np.mean(y_PPPO_5, axis=(3))

y_PPPO_1_sum = np.sum(y_PPPO_1_av, axis=0)
y_PPPO_2_sum = np.sum(y_PPPO_2_av, axis=0)
y_PPPO_3_sum= np.sum(y_PPPO_3_av, axis=0)
y_PPPO_4_sum = np.sum(y_PPPO_4_av, axis=0)
y_PPPO_5_sum = np.sum(y_PPPO_5_av, axis=0)

y_PPPO_1_fin = np.sum(y_PPPO_1, axis=0)
y_PPPO_2_fin = np.sum(y_PPPO_2, axis=0)
y_PPPO_3_fin = np.sum(y_PPPO_3, axis=0)
y_PPPO_4_fin = np.sum(y_PPPO_4, axis=0)
y_PPPO_5_fin = np.sum(y_PPPO_5, axis=0)

y_PPPO_1_sd = np.std(y_PPPO_1_fin, axis=2)
ci_y_1 = 1.96 * y_PPPO_1_sd 
y_PPPO_2_sd = np.std(y_PPPO_2_fin, axis=2)
ci_y_2 = 1.96 * y_PPPO_2_sd 
y_PPPO_3_sd = np.std(y_PPPO_3_fin, axis=2)
ci_y_3 = 1.96 * y_PPPO_3_sd 
y_PPPO_4_sd = np.std(y_PPPO_4_fin, axis=2)
ci_y_4 = 1.96 * y_PPPO_4_sd 
y_PPPO_5_sd = np.std(y_PPPO_5_fin, axis=2)
ci_y_5 = 1.96 * y_PPPO_5_sd 

cv_y_1 = np.std(y_PPPO_1_fin, axis=2)/y_PPPO_1_sum
cv_y_2 = np.std(y_PPPO_2_fin, axis=2)/y_PPPO_2_sum
cv_y_3 = np.std(y_PPPO_3_fin, axis=2)/y_PPPO_3_sum
cv_y_4 = np.std(y_PPPO_4_fin, axis=2)/y_PPPO_4_sum
cv_y_5 = np.std(y_PPPO_5_fin, axis=2)/y_PPPO_5_sum

#Timeline Net Flows

#Supply 

flow_Lambda_1 = (np.sum(Lambda, axis=0)*0.5 - np.sum(Lambda, axis=1)*0.5)*0.5
flow_Lambda_2 = (np.sum(Lambda, axis=0)*0.75 - np.sum(Lambda, axis=1)*0.75)*0.5
flow_Lambda_3 = (np.sum(Lambda, axis=0)*1.0 - np.sum(Lambda, axis=1)*1.0)*0.5
flow_Lambda_4 = (np.sum(Lambda, axis=0)*1.25 - np.sum(Lambda, axis=1)*1.25)*0.5
flow_Lambda_5 = (np.sum(Lambda, axis=0)*1.5 - np.sum(Lambda, axis=1)*1.5)*0.5

flow_c_1 = np.sum(c_PPPO_1, axis=(0)) - np.sum(c_PPPO_1, axis=(1)) 
flow_c_2 = np.sum(c_PPPO_2, axis=(0)) - np.sum(c_PPPO_2, axis=(1)) 
flow_c_3 = np.sum(c_PPPO_3, axis=(0)) - np.sum(c_PPPO_3, axis=(1)) 
flow_c_4 = np.sum(c_PPPO_4, axis=(0)) - np.sum(c_PPPO_4, axis=(1)) 
flow_c_5 = np.sum(c_PPPO_5, axis=(0)) - np.sum(c_PPPO_5, axis=(1)) 

flow_c_1_av = np.mean(flow_c_1, axis=(2))
flow_c_2_av = np.mean(flow_c_2, axis=(2))
flow_c_3_av = np.mean(flow_c_3, axis=(2))
flow_c_4_av = np.mean(flow_c_4, axis=(2))
flow_c_5_av = np.mean(flow_c_5, axis=(2))

flow_c_1_sd = np.std(flow_c_1, axis=2)
ci_c_1 = 1.96 * flow_c_1_sd 
flow_c_2_sd = np.std(flow_c_2, axis=2)
ci_c_2 = 1.96 * flow_c_2_sd 
flow_c_3_sd = np.std(flow_c_3, axis=2)
ci_c_3 = 1.96 * flow_c_3_sd 
flow_c_4_sd = np.std(flow_c_4, axis=2)
ci_c_4 = 1.96 * flow_c_4_sd 
flow_c_5_sd = np.std(flow_c_5, axis=2)
ci_c_5 = 1.96 * flow_c_5_sd 

#Demand 
ncols = 1
nrows = 1

iterations = np.linspace(0, 35, 36)

fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(14,9))

axes.plot(iterations, p_PPPO_1_av[3,:], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=0.5$')
axes.plot(iterations, p_PPPO_3_av[3,:], marker='o', color='blue', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.0$')
axes.plot(iterations, p_PPPO_5_av[3,:], marker='s', linestyle = 'dashed', color='red', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.5$')
#axes.fill_between(iterations, (p_PPPO_1_av[3,:]-ci_p_1[3,:]), (p_PPPO_1_av[3,:]+ci_p_1[3,:]), color='green', alpha=.3)
#axes.fill_between(iterations, (p_PPPO_3_av[3,:]-ci_p_3[3,:]), (p_PPPO_3_av[3,:]+ci_p_3[3,:]), color='blue', alpha=.3)
#axes.fill_between(iterations, (p_PPPO_5_av[3,:]-ci_p_5[3,:]), (p_PPPO_5_av[3,:]+ci_p_5[3,:]), color='red', alpha=.3)

axes.errorbar(iterations, p_PPPO_1_av[3,:], yerr=ci_p_1[3,:], fmt='none', ecolor='green', capsize=5, elinewidth=1.5)  # Error bars for 0.5
axes.errorbar(iterations, p_PPPO_5_av[3,:], yerr=ci_p_5[3,:], fmt='none', ecolor='red', capsize=5, elinewidth=1.5)  # Error bars for 1.5
# Uncomment the next line if you want the error bars for the 1.0 case
axes.errorbar(iterations, p_PPPO_3_av[3,:], yerr=ci_p_3[3,:], fmt='none', ecolor='blue', capsize=5, elinewidth=1.5)

axes.tick_params(axis='both', which='major', labelsize=24)
axes.tick_params(axis='both', which='minor', labelsize=24)
axes.set_xlabel("Decision Epochs", fontsize=24)
axes.set_ylabel("Prices", fontsize=24)
axes.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8], labels=[0.4, 0.5, 0.6, 0.7, 0.8])

axes.set_ylim((0.42,0.75))
axes.set_xlim((-0.5, 35.5))
axes.legend(loc='upper left', prop={'size': 24})      
plt.savefig("fig5a.pdf", bbox_inches='tight')

fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(14,9))

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

fig, axes = plt.subplots(ncols = ncols, nrows = nrows, figsize=(14,9))

axes.plot(iterations, lambda_PPPO_1_sum[3,:], marker='^', linestyle = 'dashdot', color='green', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=0.5$')
axes.plot(iterations, lambda_PPPO_3_sum[3,:], marker='o', color='blue', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.0$')
axes.plot(iterations, lambda_PPPO_5_sum[3,:], marker='s', linestyle = 'dashed', color='red', markersize=6, linewidth=1.0,label='$\overline{\Lambda}=1.5$')
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

p_PPPO_1 = np.load("PPPO_simulations/instance_1.0/p_PPPO.npy") 
p_PPPO_2 = np.load("PPPO_simulations/instance_2.0/p_PPPO.npy") 
p_PPPO_3 = np.load("PPPO_simulations/instance_3.0/p_PPPO.npy") 
p_PPPO_4 = np.load("PPPO_simulations/instance_4.0/p_PPPO.npy") 
p_PPPO_5 = np.load("PPPO_simulations/instance_5.0/p_PPPO.npy") 

p_PPPO_1_av = np.mean(p_PPPO_1, axis=(2))
p_PPPO_2_av = np.mean(p_PPPO_2, axis=(2))
p_PPPO_3_av = np.mean(p_PPPO_3, axis=(2))
p_PPPO_4_av = np.mean(p_PPPO_4, axis=(2))
p_PPPO_5_av = np.mean(p_PPPO_5, axis=(2))

p_PPPO_1_sd = np.std(p_PPPO_1, axis=(2))
p_PPPO_2_sd = np.std(p_PPPO_2, axis=(2))
p_PPPO_3_sd = np.std(p_PPPO_3, axis=(2))
p_PPPO_4_sd = np.std(p_PPPO_4, axis=(2))
p_PPPO_5_sd = np.std(p_PPPO_5, axis=(2))

p_PPPO_1_cv = p_PPPO_1_sd / p_PPPO_1_av
p_PPPO_2_cv = p_PPPO_2_sd / p_PPPO_2_av
p_PPPO_3_cv = p_PPPO_3_sd / p_PPPO_3_av
p_PPPO_4_cv = p_PPPO_4_sd / p_PPPO_4_av
p_PPPO_5_cv = p_PPPO_5_sd / p_PPPO_5_av


p_PPPO_1_av = np.mean(p_PPPO_1, axis=(1,2))
p_PPPO_1_sd = np.std(np.mean(p_PPPO_1, axis=(2)),axis=1)

