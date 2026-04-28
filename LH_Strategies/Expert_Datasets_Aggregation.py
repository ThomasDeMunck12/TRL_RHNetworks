# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:43:37 2025

@author: thoma
"""
import sys 
import numpy as np 

parameter = float(sys.argv[1])
#parameter = 3.0
print('Start task number: ', parameter)

#Change the parameters according to the instance 

part1_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_1.0.npy", allow_pickle=True) 
part2_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_2.0.npy", allow_pickle=True) 
part3_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_3.0.npy", allow_pickle=True) 
part4_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_4.0.npy", allow_pickle=True) 
part5_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_5.0.npy", allow_pickle=True) 
part6_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_6.0.npy", allow_pickle=True) 
part7_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_7.0.npy", allow_pickle=True) 
part8_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_8.0.npy", allow_pickle=True) 
part9_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_9.0.npy", allow_pickle=True) 
part10_obs1 = np.load("./expert_dataset_observations/expert_observations_1_"+str(parameter)+"_10.0.npy", allow_pickle=True) 

part1_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_1.0.npy", allow_pickle=True) 
part2_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_2.0.npy", allow_pickle=True) 
part3_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_3.0.npy", allow_pickle=True) 
part4_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_4.0.npy", allow_pickle=True) 
part5_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_5.0.npy", allow_pickle=True) 
part6_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_6.0.npy", allow_pickle=True) 
part7_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_7.0.npy", allow_pickle=True) 
part8_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_8.0.npy", allow_pickle=True) 
part9_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_9.0.npy", allow_pickle=True) 
part10_obs2 = np.load("./expert_dataset_observations/expert_observations_2_"+str(parameter)+"_10.0.npy", allow_pickle=True) 

part1_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_1.0.npy", allow_pickle=True) 
part2_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_2.0.npy", allow_pickle=True) 
part3_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_3.0.npy", allow_pickle=True) 
part4_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_4.0.npy", allow_pickle=True) 
part5_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_5.0.npy", allow_pickle=True) 
part6_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_6.0.npy", allow_pickle=True) 
part7_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_7.0.npy", allow_pickle=True) 
part8_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_8.0.npy", allow_pickle=True) 
part9_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_9.0.npy", allow_pickle=True) 
part10_act1 = np.load("./expert_dataset_actions/expert_actions_1_"+str(parameter)+"_10.0.npy", allow_pickle=True) 

part1_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_1.0.npy", allow_pickle=True) 
part2_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_2.0.npy", allow_pickle=True) 
part3_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_3.0.npy", allow_pickle=True) 
part4_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_4.0.npy", allow_pickle=True) 
part5_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_5.0.npy", allow_pickle=True) 
part6_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_6.0.npy", allow_pickle=True) 
part7_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_7.0.npy", allow_pickle=True) 
part8_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_8.0.npy", allow_pickle=True) 
part9_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_9.0.npy", allow_pickle=True) 
part10_act2 = np.load("./expert_dataset_actions/expert_actions_2_"+str(parameter)+"_10.0.npy", allow_pickle=True) 

expert_observations1 = np.concatenate((part1_obs1, part2_obs1, part3_obs1, part4_obs1, part5_obs1, part6_obs1, part7_obs1, part8_obs1, part9_obs1, part10_obs1), axis=0)
expert_actions1 = np.concatenate((part1_act1, part2_act1, part3_act1, part4_act1, part5_act1, part6_act1, part7_act1, part8_act1, part9_act1, part10_act1), axis=0)

np.save("expert_datasets1/expert_observations1_"+str(parameter)+".npy", expert_observations1)
np.save("expert_datasets1/expert_actions1_"+str(parameter)+".npy", expert_actions1)

expert_observations2 = np.concatenate((part1_obs2, part2_obs2, part3_obs2, part4_obs2, part5_obs2, part6_obs2, part7_obs2, part8_obs2, part9_obs2, part10_obs2), axis=0)
expert_actions2 = np.concatenate((part1_act2, part2_act2, part3_act2, part4_act2, part5_act2, part6_act2, part7_act2, part8_act2, part9_act2, part10_act2), axis=0)

np.save("expert_datasets2/expert_observations2_"+str(parameter)+".npy", expert_observations2)
np.save("expert_datasets2/expert_actions2_"+str(parameter)+".npy", expert_actions2)
