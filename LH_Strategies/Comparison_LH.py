# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:32:57 2024

@author: ThomasDM
"""
import numpy as np
import pandas as pd
import os 

os.chdir("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\4_Experiments\\Section_5.2")

num_instances = 35
profits = np.zeros((num_instances,4))
#range_=[0,1,2,3,4,5,6,8,9,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29] #avoid counting default instance several times

for j in range(num_instances): 

    i = float(j+1)
    
    RH_1_12 = np.load('./Results/RH_1_12/score_history_'+str(i)+'.npy')
    RH_1_12 = np.mean(RH_1_12)
    
    RH_4_12 = np.load('./Results/RH_4_12/score_history_'+str(i)+'.npy')
    RH_4_12 = np.mean(RH_4_12)
    
    RH_36_36 = np.load('./Results/RH_36_36/score_history_'+str(i)+'.npy')
    RH_36_36 = np.mean(RH_36_36)
    
    P_HPPO = np.load('./Results/P_HPPO/evaluations_'+str(i)+'.npz')
    P_HPPO = P_HPPO['results']
    P_HPPO = np.mean(P_HPPO, axis=1)
    P_HPPO = np.max(P_HPPO)
    
    profits[j, 0] = RH_36_36
    profits[j, 1] = RH_4_12
    profits[j, 2] = RH_1_12
    profits[j, 3] = P_HPPO
DF = pd.DataFrame(profits) 
print(DF)

        
        
        

