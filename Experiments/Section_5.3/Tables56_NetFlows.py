# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 11:44:40 2025

@author: thoma
"""

import numpy as np

#OD Pricing 

#Inflows and outflows

Lambda_day = np.load("../../Data/arrival_rates_8loc.npy")
Lambda = Lambda_day[:, :, 84:120]
Lambda_weights = np.sum(Lambda, axis=2)
#Data from simulation

Lambda_weighted = Lambda_weights[:, :, None, None]              

p = np.load("HPPPO_simulations_MC/instance_3.0/p_PPPO.npy") 
weighted_sum = np.sum(p * Lambda_weighted, axis=1)   
normalizer = np.sum(Lambda_weighted, axis=1)       
p_weighted = weighted_sum / normalizer 
p_av = np.mean(p_weighted, axis=(1, 2))



p_single = np.sum(p_av * Lambda_weights.sum(axis=0))/np.sum(Lambda)

#p_av=np.mean(p, axis=3)
#weighted_sum = np.sum(p_av * Lambda, axis=(1, 2))   # sum over dest and time
#normalizer   = np.sum(Lambda, axis=(1, 2))            # sum of weights per origin
#weighted_avg = weighted_sum / normalizer  



c_single_pricing = Lambda * (1-p_single)

c = np.load("HPPPO_simulations_MC/instance_3.0/lambda_PPPO.npy") 
c_av = np.mean(c, axis=(3))
c_av = np.sum(c_av, axis=(2))

d = np.load("HPPPO_simulations_MC/instance_3.0/d_PPPO.npy")
d_av = np.mean(d, axis=(3))
d_av = np.sum(d_av, axis=(2))

r = np.load("HPPPO_simulations_MC/instance_3.0/r_PPPO.npy") 
r_av = np.mean(r, axis=(3))
r_av = np.sum(r_av, axis=(2))

w = np.load("HPPPO_simulations_MC/instance_3.0/w_PPPO.npy")
w_av = np.mean(w, axis=(3))
w_av = np.sum(w_av, axis=(2))

#Net flows

c_single_pricing_NF = np.sum(c_single_pricing, axis=(0,2)) - np.sum(c_single_pricing, axis=(1,2))
c_NF = np.sum(c_av, axis=0) - np.sum(c_av, axis=1)
d_NF = np.sum(d_av, axis=0) - np.sum(d_av, axis=1)
r_NF = np.sum(r_av, axis=0) - np.sum(r_av, axis=1)
w_NF = np.sum(w_av, axis=0) - np.sum(w_av, axis=1)


