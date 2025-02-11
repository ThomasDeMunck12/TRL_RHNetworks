# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:45:45 2023

@author: ThomasDM
"""


import pandas as pd
import numpy as np
from community import community_louvain
import networkx as nx
from matplotlib import pyplot as plt
import os 

#Part 1. Data Reduction

fhvhv_tripdata_2019_02 = pd.read_parquet('fhvhv_tripdata_2019-02.parquet', engine='pyarrow')
fhvhv_tripdata_2019_02 = fhvhv_tripdata_2019_02.dropna(subset=['PULocationID'])
fhvhv_tripdata_2019_02 = fhvhv_tripdata_2019_02.dropna(subset=['DOLocationID'])

fhvhv_tripdata_2019_03 = pd.read_parquet('fhvhv_tripdata_2019-03.parquet', engine='pyarrow')
fhvhv_tripdata_2019_03 = fhvhv_tripdata_2019_03.dropna(subset=['PULocationID'])
fhvhv_tripdata_2019_03 = fhvhv_tripdata_2019_03.dropna(subset=['DOLocationID'])

taxi_zone_lookupcsv = pd.read_csv("taxi_zone_lookup.csv")
taxi_zone_lookupcsv = taxi_zone_lookupcsv[taxi_zone_lookupcsv["Borough"]=="Manhattan"]

#Remove if outside Manhattan island
compatible_id = taxi_zone_lookupcsv["LocationID"].tolist()
compatible_id.remove(202)
compatible_id.remove(194)
compatible_id.remove(103)
compatible_id.remove(104)
compatible_id.remove(105)

df = fhvhv_tripdata_2019_02[fhvhv_tripdata_2019_02["PULocationID"].isin(compatible_id)]
df = df[df["DOLocationID"].isin(compatible_id)]

df_march = fhvhv_tripdata_2019_03[fhvhv_tripdata_2019_03["PULocationID"].isin(compatible_id)]
df_march = df_march[df_march["DOLocationID"].isin(compatible_id)]

#Replace old inscriptions by binary values
df = df.replace(['Y', 'N'], [1, 0])
df = df.replace(['Y', 'N'], [1, 0])
df = df[df['shared_match_flag'] == 0] #No shared rides
df = df[df['shared_request_flag'] == 0] #No shared requests
df = df[df['hvfhs_license_num'] == 'HV0003'] #Select only the trips that occurred with Uber
df['passenger fare'] = df['base_passenger_fare'] + df['tips'] 
df = df[['request_datetime', 'on_scene_datetime', 'pickup_datetime', 'dropoff_datetime',
    'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time', 'driver_pay', 'passenger fare']]
df = df[df['trip_time'] < 7200] #No unrealistic outliers
df = df[df['passenger fare'] > 0] 
df = df[df['trip_time'] > 0]
df = df[df['driver_pay'] > 0]

PU_id = df['PULocationID'].unique()
DO_id = df['DOLocationID'].unique()

#March 
#Replace old inscriptions by binary values
df_march = df_march.replace(['Y', 'N'], [1, 0])
df_march = df_march.replace(['Y', 'N'], [1, 0])
df_march = df_march[df_march['shared_match_flag'] == 0] #No shared rides
df_march = df_march[df_march['shared_request_flag'] == 0] #No shared requests
df_march = df_march[df_march['hvfhs_license_num'] == 'HV0003'] #Select only the trips that occured with Uber
df_march['passenger fare'] = df_march['base_passenger_fare'] + df_march['tips'] 
df_march = df_march[['request_datetime', 'on_scene_datetime', 'pickup_datetime', 'dropoff_datetime',
    'PULocationID', 'DOLocationID', 'trip_miles', 'trip_time', 'driver_pay', 'passenger fare']]
df_march = df_march[df_march['trip_time'] < 7200] #No unrealistic outliers
df_march = df_march[df_march['passenger fare'] > 0] 
df_march = df_march[df_march['trip_time'] > 0]
df_march = df_march[df_march['driver_pay'] > 0]

PU_id_march = df_march['PULocationID'].unique()
DO_id_march = df_march['DOLocationID'].unique()


#Part 2. Clustering 

#20 locs 

area_0_20loc = [12,13,261,231]
area_1_20loc = [125,158,249] 
area_2_20loc = [211,144,114,113,79]
area_3_20loc = [88,87,209,45]
area_4_20loc = [148,232,4]
area_5_20loc = [246,68]
area_6_20loc = [90,186,100] 
area_7_20loc = [164,234] 
area_8_20loc = [107,224]
area_9_20loc = [170,137]
area_10_20loc = [233,229,162]
area_11_20loc = [161,230,163]
area_12_20loc = [48,50]
area_13_20loc = [143,142,239,43]
area_14_20loc = [237,141,140]
area_15_20loc = [24,151,238]
area_16_20loc = [75,263,262,236] 
area_17_20loc = [166,152,116]
area_18_20loc = [41,42,74]
area_19_20loc = [244,120,243,127,128,153]

#8 locs

area_0_8loc = [13, 261, 12, 231, 125, 211, 144, 158, 249, 114, 113]
area_1_8loc = [88, 87, 209, 45,148, 232, 79, 4]
area_2_8loc = [246, 50, 68, 48, 90, 186, 100, 230]
area_3_8loc = [234, 164,  161, 163, 107, 224, 137, 170, 233, 162, 229]
area_4_8loc = [143, 142, 239, 238, 151, 24, 43]
area_5_8loc = [237, 236, 141, 140, 263, 262, 75]
area_6_8loc = [166, 152, 116, 41, 42, 74]
area_7_8loc = [244, 120, 243, 127, 128, 153] 
              
# Part 3. Dataset creation

#Month

df["month"] = df.pickup_datetime.dt.month
df_march["month"] = df_march.pickup_datetime.dt.month

#Day
 
df["day"] = df.pickup_datetime.dt.day
df_march["day"] = df_march.pickup_datetime.dt.day

#Hour

df["hour"] = df.pickup_datetime.dt.hour
df_march["hour"] = df_march.pickup_datetime.dt.hour

#Minutes

df["minute"] = df.pickup_datetime.dt.minute
df_march["minute"] = df_march.pickup_datetime.dt.minute

#5 minutes
def f(x):    
    a = x[13]%5
    b = x[13]-a
    return x[12]*12 + b/5    

df["five minutes"] = df.apply(f, axis=1) 
df_march["five minutes"] = df_march.apply(f, axis=1) 

#A. Medium instances

df_8loc=df.copy()

df_8loc.loc[df['PULocationID'].isin(area_0_8loc), 'PULocationID'] = 0
df_8loc.loc[df['PULocationID'].isin(area_1_8loc), 'PULocationID'] = 1
df_8loc.loc[df['PULocationID'].isin(area_2_8loc), 'PULocationID'] = 2
df_8loc.loc[df['PULocationID'].isin(area_3_8loc), 'PULocationID'] = 3
df_8loc.loc[df['PULocationID'].isin(area_4_8loc), 'PULocationID'] = 4
df_8loc.loc[df['PULocationID'].isin(area_5_8loc), 'PULocationID'] = 5
df_8loc.loc[df['PULocationID'].isin(area_6_8loc), 'PULocationID'] = 6
df_8loc.loc[df['PULocationID'].isin(area_7_8loc), 'PULocationID'] = 7


df_8loc.loc[df['DOLocationID'].isin(area_0_8loc), 'DOLocationID'] = 0
df_8loc.loc[df['DOLocationID'].isin(area_1_8loc), 'DOLocationID'] = 1
df_8loc.loc[df['DOLocationID'].isin(area_2_8loc), 'DOLocationID'] = 2
df_8loc.loc[df['DOLocationID'].isin(area_3_8loc), 'DOLocationID'] = 3
df_8loc.loc[df['DOLocationID'].isin(area_4_8loc), 'DOLocationID'] = 4
df_8loc.loc[df['DOLocationID'].isin(area_5_8loc), 'DOLocationID'] = 5
df_8loc.loc[df['DOLocationID'].isin(area_6_8loc), 'DOLocationID'] = 6
df_8loc.loc[df['DOLocationID'].isin(area_7_8loc), 'DOLocationID'] = 7

#B. Large instances 

df_20loc=df.copy()

df_20loc.loc[df['PULocationID'].isin(area_0_20loc), 'PULocationID'] = 0
df_20loc.loc[df['PULocationID'].isin(area_1_20loc), 'PULocationID'] = 1
df_20loc.loc[df['PULocationID'].isin(area_2_20loc), 'PULocationID'] = 2
df_20loc.loc[df['PULocationID'].isin(area_3_20loc), 'PULocationID'] = 3
df_20loc.loc[df['PULocationID'].isin(area_4_20loc), 'PULocationID'] = 4
df_20loc.loc[df['PULocationID'].isin(area_5_20loc), 'PULocationID'] = 5
df_20loc.loc[df['PULocationID'].isin(area_6_20loc), 'PULocationID'] = 6
df_20loc.loc[df['PULocationID'].isin(area_7_20loc), 'PULocationID'] = 7
df_20loc.loc[df['PULocationID'].isin(area_8_20loc), 'PULocationID'] = 8
df_20loc.loc[df['PULocationID'].isin(area_9_20loc), 'PULocationID'] = 9
df_20loc.loc[df['PULocationID'].isin(area_10_20loc), 'PULocationID'] = 10
df_20loc.loc[df['PULocationID'].isin(area_11_20loc), 'PULocationID'] = 11
df_20loc.loc[df['PULocationID'].isin(area_12_20loc), 'PULocationID'] = 12
df_20loc.loc[df['PULocationID'].isin(area_13_20loc), 'PULocationID'] = 13
df_20loc.loc[df['PULocationID'].isin(area_14_20loc), 'PULocationID'] = 14
df_20loc.loc[df['PULocationID'].isin(area_15_20loc), 'PULocationID'] = 15
df_20loc.loc[df['PULocationID'].isin(area_16_20loc), 'PULocationID'] = 16
df_20loc.loc[df['PULocationID'].isin(area_17_20loc), 'PULocationID'] = 17
df_20loc.loc[df['PULocationID'].isin(area_18_20loc), 'PULocationID'] = 18
df_20loc.loc[df['PULocationID'].isin(area_19_20loc), 'PULocationID'] = 19


df_20loc.loc[df['DOLocationID'].isin(area_0_20loc), 'DOLocationID'] = 0
df_20loc.loc[df['DOLocationID'].isin(area_1_20loc), 'DOLocationID'] = 1
df_20loc.loc[df['DOLocationID'].isin(area_2_20loc), 'DOLocationID'] = 2
df_20loc.loc[df['DOLocationID'].isin(area_3_20loc), 'DOLocationID'] = 3
df_20loc.loc[df['DOLocationID'].isin(area_4_20loc), 'DOLocationID'] = 4
df_20loc.loc[df['DOLocationID'].isin(area_5_20loc), 'DOLocationID'] = 5
df_20loc.loc[df['DOLocationID'].isin(area_6_20loc), 'DOLocationID'] = 6
df_20loc.loc[df['DOLocationID'].isin(area_7_20loc), 'DOLocationID'] = 7
df_20loc.loc[df['DOLocationID'].isin(area_8_20loc), 'DOLocationID'] = 8
df_20loc.loc[df['DOLocationID'].isin(area_9_20loc), 'DOLocationID'] = 9
df_20loc.loc[df['DOLocationID'].isin(area_10_20loc), 'DOLocationID'] = 10
df_20loc.loc[df['DOLocationID'].isin(area_11_20loc), 'DOLocationID'] = 11
df_20loc.loc[df['DOLocationID'].isin(area_12_20loc), 'DOLocationID'] = 12
df_20loc.loc[df['DOLocationID'].isin(area_13_20loc), 'DOLocationID'] = 13
df_20loc.loc[df['DOLocationID'].isin(area_14_20loc), 'DOLocationID'] = 14
df_20loc.loc[df['DOLocationID'].isin(area_15_20loc), 'DOLocationID'] = 15
df_20loc.loc[df['DOLocationID'].isin(area_16_20loc), 'DOLocationID'] = 16
df_20loc.loc[df['DOLocationID'].isin(area_17_20loc), 'DOLocationID'] = 17
df_20loc.loc[df['DOLocationID'].isin(area_18_20loc), 'DOLocationID'] = 18
df_20loc.loc[df['DOLocationID'].isin(area_19_20loc), 'DOLocationID'] = 19

#March 

df_20loc_march=df_march.copy()

df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_0_20loc), 'PULocationID'] = 0
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_1_20loc), 'PULocationID'] = 1
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_2_20loc), 'PULocationID'] = 2
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_3_20loc), 'PULocationID'] = 3
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_4_20loc), 'PULocationID'] = 4
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_5_20loc), 'PULocationID'] = 5
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_6_20loc), 'PULocationID'] = 6
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_7_20loc), 'PULocationID'] = 7
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_8_20loc), 'PULocationID'] = 8
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_9_20loc), 'PULocationID'] = 9
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_10_20loc), 'PULocationID'] = 10
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_11_20loc), 'PULocationID'] = 11
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_12_20loc), 'PULocationID'] = 12
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_13_20loc), 'PULocationID'] = 13
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_14_20loc), 'PULocationID'] = 14
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_15_20loc), 'PULocationID'] = 15
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_16_20loc), 'PULocationID'] = 16
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_17_20loc), 'PULocationID'] = 17
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_18_20loc), 'PULocationID'] = 18
df_20loc_march.loc[df_20loc_march['PULocationID'].isin(area_19_20loc), 'PULocationID'] = 19


df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_0_20loc), 'DOLocationID'] = 0
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_1_20loc), 'DOLocationID'] = 1
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_2_20loc), 'DOLocationID'] = 2
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_3_20loc), 'DOLocationID'] = 3
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_4_20loc), 'DOLocationID'] = 4
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_5_20loc), 'DOLocationID'] = 5
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_6_20loc), 'DOLocationID'] = 6
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_7_20loc), 'DOLocationID'] = 7
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_8_20loc), 'DOLocationID'] = 8
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_9_20loc), 'DOLocationID'] = 9
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_10_20loc), 'DOLocationID'] = 10
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_11_20loc), 'DOLocationID'] = 11
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_12_20loc), 'DOLocationID'] = 12
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_13_20loc), 'DOLocationID'] = 13
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_14_20loc), 'DOLocationID'] = 14
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_15_20loc), 'DOLocationID'] = 15
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_16_20loc), 'DOLocationID'] = 16
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_17_20loc), 'DOLocationID'] = 17
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_18_20loc), 'DOLocationID'] = 18
df_20loc_march.loc[df_20loc_march['DOLocationID'].isin(area_19_20loc), 'DOLocationID'] = 19

#Part 4. Compute the arrival rates, average transit times, average distances

#A. Medium instances - 8 locs

N = 8
T = 288

#Arrival rates

arrival_rates = np.zeros([N, N, T])

for i in range(N):
    for j in range(N):
        for k in range(T):
            x = df_8loc[(df_8loc['PULocationID'] == i) & (df_8loc['DOLocationID']==j) & (df_8loc['five minutes']==k) & (df_8loc['day']==6)] 
            arrival_rates[i,j,k] = len(x)

np.save("arrival_rates_8loc", arrival_rates)

#Average trip durations

average_trip_durations = np.zeros([N, N])

for i in range(N):
    for j in range(N):
        x = df_8loc[((df_8loc['PULocationID'] == i) & (df_8loc['DOLocationID']==j))] 
        y = x["trip_time"].mean()
        average_trip_durations[i,j] = y/300

np.save("average_trip_durations_8loc", average_trip_durations)

#Average distances

average_distances = np.zeros([N, N])

for i in range(N):
    for j in range(N):
        x = df_8loc[((df_8loc['PULocationID'] == i) & (df_8loc['DOLocationID']==j))] 
        y = x["trip_miles"].mean()
        average_distances[i,j] = y

np.save("average_distances_8loc", average_distances)

#b.  large instances - to be continued on Julia

df_20loc.to_csv('df_20.csv')
df_20loc_march.to_csv('df_20_march.csv')

