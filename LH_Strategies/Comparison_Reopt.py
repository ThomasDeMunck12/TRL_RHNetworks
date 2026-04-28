# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 11:03:47 2025

@author: thomas
"""

import numpy as np
import pandas as pd
import os 

os.chdir("C:\\Users\\thoma\\OneDrive\\Desktop\\Revision1\\2_RH_Strategies")
  
# RH 4 12

RH_4_12_13_1 = np.load('./RH_4_12_additional/score_history_13.0_1.npy')
RH_4_12_13_1 = np.mean(RH_4_12_13_1)

RH_4_12_13_2 = np.load('./RH_4_12_additional/score_history_13.0_2.npy')
RH_4_12_13_2 = np.mean(RH_4_12_13_2)

RH_4_12_13_3 = np.load('./RH_4_12_additional/score_history_13.0_3.npy')
RH_4_12_13_3 = np.mean(RH_4_12_13_3)

RH_4_12_13_4 = np.load('./RH_4_12_additional/score_history_13.0_4.npy')
RH_4_12_13_4 = np.mean(RH_4_12_13_4)

RH_4_12_14_1 = np.load('./RH_4_12_additional/score_history_14.0_1.npy')
RH_4_12_14_1 = np.mean(RH_4_12_14_1)

RH_4_12_14_2 = np.load('./RH_4_12_additional/score_history_14.0_2.npy')
RH_4_12_14_2 = np.mean(RH_4_12_14_2)

RH_4_12_14_3 = np.load('./RH_4_12_additional/score_history_14.0_3.npy')
RH_4_12_14_3 = np.mean(RH_4_12_14_3)

RH_4_12_14_4 = np.load('./RH_4_12_additional/score_history_14.0_4.npy')
RH_4_12_14_4 = np.mean(RH_4_12_14_4)

RH_4_12_15_1 = np.load('./RH_4_12_additional/score_history_15.0_1.npy')
RH_4_12_15_1 = np.mean(RH_4_12_15_1)

RH_4_12_15_2 = np.load('./RH_4_12_additional/score_history_15.0_2.npy')
RH_4_12_15_2 = np.mean(RH_4_12_15_2)

RH_4_12_15_3 = np.load('./RH_4_12_additional/score_history_15.0_3.npy')
RH_4_12_15_3 = np.mean(RH_4_12_15_3)

RH_4_12_15_4 = np.load('./RH_4_12_additional/score_history_15.0_4.npy')
RH_4_12_15_4 = np.mean(RH_4_12_15_4)
    
# RH 1 12

RH_1_12_13_1 = np.load('./RH_1_12_additional/score_history_13.0_1.npy')
RH_1_12_13_1 = np.mean(RH_1_12_13_1)

RH_1_12_13_2 = np.load('./RH_1_12_additional/score_history_13.0_2.npy')
RH_1_12_13_2 = np.mean(RH_1_12_13_2)

RH_1_12_13_3 = np.load('./RH_1_12_additional/score_history_13.0_3.npy')
RH_1_12_13_3 = np.mean(RH_1_12_13_3)

RH_1_12_13_4 = np.load('./RH_1_12_additional/score_history_13.0_4.npy')
RH_1_12_13_4 = np.mean(RH_1_12_13_4)

RH_1_12_14_1 = np.load('./RH_1_12_additional/score_history_14.0_1.npy')
RH_1_12_14_1 = np.mean(RH_1_12_14_1)

RH_1_12_14_2 = np.load('./RH_1_12_additional/score_history_14.0_2.npy')
RH_1_12_14_2 = np.mean(RH_1_12_14_2)

RH_1_12_14_3 = np.load('./RH_1_12_additional/score_history_14.0_3.npy')
RH_1_12_14_3 = np.mean(RH_1_12_14_3)

RH_1_12_14_4 = np.load('./RH_1_12_additional/score_history_14.0_4.npy')
RH_1_12_14_4 = np.mean(RH_1_12_14_4)

RH_1_12_15_1 = np.load('./RH_1_12_additional/score_history_15.0_1.npy')
RH_1_12_15_1 = np.mean(RH_1_12_15_1)

RH_1_12_15_2 = np.load('./RH_1_12_additional/score_history_15.0_2.npy')
RH_1_12_15_2 = np.mean(RH_1_12_15_2)

RH_1_12_15_3 = np.load('./RH_1_12_additional/score_history_15.0_3.npy')
RH_1_12_15_3 = np.mean(RH_1_12_15_3)

RH_1_12_15_4 = np.load('./RH_1_12_additional/score_history_15.0_4.npy')
RH_1_12_15_4 = np.mean(RH_1_12_15_4)
    
# RH 36 36

RH_36_36_13_1 = np.load('./RH_36_36_additional/score_history_13.0_1.npy')
RH_36_36_13_1 = np.mean(RH_36_36_13_1)

RH_36_36_13_2 = np.load('./RH_36_36_additional/score_history_13.0_2.npy')
RH_36_36_13_2 = np.mean(RH_36_36_13_2)

RH_36_36_13_3 = np.load('./RH_36_36_additional/score_history_13.0_3.npy')
RH_36_36_13_3 = np.mean(RH_36_36_13_3)

RH_36_36_13_4 = np.load('./RH_36_36_additional/score_history_13.0_4.npy')
RH_36_36_13_4 = np.mean(RH_36_36_13_4)

RH_36_36_14_1 = np.load('./RH_36_36_additional/score_history_14.0_1.npy')
RH_36_36_14_1 = np.mean(RH_36_36_14_1)

RH_36_36_14_2 = np.load('./RH_36_36_additional/score_history_14.0_2.npy')
RH_36_36_14_2 = np.mean(RH_36_36_14_2)

RH_36_36_14_3 = np.load('./RH_36_36_additional/score_history_14.0_3.npy')
RH_36_36_14_3 = np.mean(RH_36_36_14_3)

RH_36_36_14_4 = np.load('./RH_36_36_additional/score_history_14.0_4.npy')
RH_36_36_14_4 = np.mean(RH_36_36_14_4)

RH_36_36_15_1 = np.load('./RH_36_36_additional/score_history_15.0_1.npy')
RH_36_36_15_1 = np.mean(RH_36_36_15_1)

RH_36_36_15_2 = np.load('./RH_36_36_additional/score_history_15.0_2.npy')
RH_36_36_15_2 = np.mean(RH_36_36_15_2)

RH_36_36_15_3 = np.load('./RH_36_36_additional/score_history_15.0_3.npy')
RH_36_36_15_3 = np.mean(RH_36_36_15_3)

RH_36_36_15_4 = np.load('./RH_36_36_additional/score_history_15.0_4.npy')
RH_36_36_15_4 = np.mean(RH_36_36_15_4)

