# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:32:57 2024

@author: ThomasDM
"""
import numpy as np
import pandas as pd
import os 


instances = [0,1,2,3,4,5,6,8,9,10,11,13,14,15,16,18,19,22,23,25,26,28,29,30,31,33,34]
num_instances = len(instances)
num_iterations = np.zeros((num_instances,3))
profits = np.zeros((num_instances,6))

a=0
for i in instances: 
    l = float(i+1)

    HPPO = np.load('Results/HPPO/lr1e_4/evaluations_' + str(l) +'.npz')
    HPPO = HPPO['results']
    HPPO = np.mean(HPPO, axis=1)*100

    HPPO2 = np.load('Results/HPPO/lr3e_4/evaluations_' + str(l) +'.npz')
    HPPO2 = HPPO2['results']
    HPPO2 = np.mean(HPPO2, axis=1)*100
    
    P_HPPO = np.load('./Results/P_HPPO/instance_'+str(l)+'/evaluations.npz')
    P_HPPO = P_HPPO['results']
    P_HPPO = np.mean(P_HPPO, axis=1)*100

    
    k=0
    max_p = P_HPPO[0]

    for j in range(1, len(P_HPPO)):
        HPPO_eval = P_HPPO[j]
        if j <= 20:    
            if HPPO_eval >= max_p*1.005:
                max_p = HPPO_eval 
                k = 0
        
        if j > 20:
            if HPPO_eval >= max_p*1.005:
                max_p = HPPO_eval 
                k = 0
            else:
                k += 1
                if k >= 15:
                    #if max_p > 0:
                    num_iterations[a, 2] = j + 1
                    #print(f"instance: {i}, value: {max_p}")
                    break 
            if j == len(P_HPPO)-1:
                num_iterations[a, 2] = j + 1

    profits[a, 5] = max_p     
    
    max_p = HPPO[0]
    k=0
    
    for j in range(1, len(HPPO)):
        HPPO_eval = HPPO[j]
        if j <= 20:    
            if HPPO_eval >= max_p*1.005:
                max_p = HPPO_eval 
                k = 0
        
        if j > 20:
            if HPPO_eval >= max_p*1.005:
                max_p = HPPO_eval
                k = 0
            else:
                k += 1
                if k >= 15:
                    #if max_p > 0:
                    num_iterations[a, 1] = j + 1
                    #print(f"instance: {i}, value: {max_p}")
                    break 
            if j == len(P_HPPO)-1:
                num_iterations[a, 1] = j + 1
                
    profits[a, 4] = max_p  
    
    max_p = HPPO2[0]
    k=0
    
    for j in range(1, len(HPPO2)):
        HPPO_eval = HPPO2[j]
        if j <= 20:    
            if HPPO_eval >= max_p*1.005:
                max_p = HPPO_eval 
                k = 0
        
        if j > 20:
            if HPPO_eval >= max_p*1.005:
                max_p = HPPO_eval 
                k = 0
            else:
                k += 1
                if k >= 15:
                    #if max_p > 0:
                    num_iterations[a, 0] = j + 1
                    #print(f"instance: {i}, value: {max_p}")
                    break 
            if j == len(HPPO2)-1:
                num_iterations[a, 0] = j + 1
    
    profits[a, 3] = max_p  
    a += 1

                
ITERATIONS = pd.DataFrame(num_iterations) 

profit_diff = np.zeros((num_instances, 5))

j=0

for k in instances: 

    i = float(k+1)
    if i!=26:
        RH_1_12 = np.load('./Results/RH_Strategies/RH_1_12/score_history_'+str(i)+'.npy')
        RH_1_12 = np.mean(RH_1_12)
    
    RH_4_12 = np.load('./Results/RH_Strategies/RH_4_12/score_history_'+str(i)+'.npy')
    RH_4_12 = np.mean(RH_4_12)
    
    RH_36_36 = np.load('./Results/RH_Strategies/RH_36_36/score_history_'+str(i)+'.npy')
    RH_36_36 = np.mean(RH_36_36)
    
    P_HPPO = np.load('Results/P_HPPO/instance_'+str(i)+'/evaluations.npz')
    P_HPPO = P_HPPO['results']
    P_HPPO = np.mean(P_HPPO, axis=1)
    P_HPPO = np.max(P_HPPO)*100
    
    HPPO = np.load('./Results/HPPO/lr1e_4/evaluations_'+str(i)+'.npz')
    HPPO = HPPO['results']
    HPPO = np.mean(HPPO, axis=1)
    HPPO = HPPO[-1]*100
    
    HPPO2 = np.load('./Results/HPPO/lr3e_4/evaluations_'+str(i)+'.npz')
    HPPO2 = HPPO2['results']
    HPPO2 = np.mean(HPPO2, axis=1)
    HPPO2 = HPPO2[-1]*100
    
    profits[j, 0] = RH_36_36
    profits[j, 1] = RH_4_12
    profits[j, 2] = RH_1_12
    profits[j, 5] = P_HPPO
    j+=1

DF_1 = pd.DataFrame(profits) 

profit_diff[:, 0] = (profits[:, 5] - profits[:, 0])/profits[:, 0]
profit_diff[:, 1] = (profits[:, 5] - profits[:, 1])/profits[:, 1]
profit_diff[:, 2] = (profits[:, 5] - profits[:, 2])/profits[:, 2]
profit_diff[:, 3] = (profits[:, 5] - profits[:, 3])/profits[:, 3]
profit_diff[:, 4] = (profits[:, 5] - profits[:, 4])/profits[:, 4]

DF_2 = pd.DataFrame(profit_diff) 

            
   
        
        
        

