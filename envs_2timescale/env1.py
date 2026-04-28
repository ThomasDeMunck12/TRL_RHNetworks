# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:16:24 2025

@author: thoma
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Env1(gym.Env):

    def __init__(self, F = 600, N = 8, H = 36,
                 p_0 = np.zeros([8,8], dtype=int), x_0 = np.array([192, 221, 73, 172, 144, 150, 146, 102], dtype=int), y_0 = np.zeros([8,8], dtype=np.float32),
                 ):
        super(Env1, self).__init__()
                
        self.N = N #Locations
        self.H = H #Decision epochs
        self.F = F
        #self.expected_time_normalized = (self.expected_time_normalized-np.mean(self.expected_time_normalized, axis=1, keepdims=True))
        #self.expected_time_normalized = (self.expected_time_normalized-np.mean(self.expected_time_normalized, axis=1, keepdims=True))/np.std(self.expected_time_normalized, axis=1,keepdims=True)
        self.p_0 = p_0
        self.x_0 = x_0 #initial state
        self.y_0 = y_0

        self.action_space = spaces.Box(low = -1, high = 1, shape=(N**2,), dtype = np.float32)         
        self.observation_space = spaces.Dict(  
            {
                "p": spaces.Box(low = 0, high = 1, shape=(N**2,), dtype = np.float32),
                "xy": spaces.Box(low=0, high= 1, shape=(N+N**2,), dtype = np.float32),
                "t": spaces.Box(low=0, high= 1, dtype=np.float32)
            })
        
        self.Max_p = self.action_space.high[0]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.copy(self.x_0)
        self.y = np.copy(self.y_0)
        self.t = 0
        self.p = np.copy(self.p_0)

        x_v = self.x / self.F
        y_v = np.reshape(self.y, -1) /self.F
        xy_v = np.concatenate([x_v, y_v], dtype=np.float32)
        p_v = np.copy(self.p)
        p_v = np.reshape(p_v, -1)
        t_v = self.t / self.H
        t_v = np.array([t_v], dtype=np.float32)

        self.state = {"p": p_v, "xy": xy_v, "t": t_v}
        return self.state, {}
    
    def step(self, action):
        #print("############################# BEGIN ENV 1 ################################")
        #print("b4: ", self.state)
        info = {}

        # Actions
        p = action[0 : self.N ** 2]
        p = np.reshape(p, [self.N, self.N])        
        p = self.Action_Encoder_Pricing(p)
        info["Prices"] = p
        reward = 0
        
        # Termination ? 
        self.p = p 
        done = self.IsTerminal()
        self.Get_State(p)

        truncation = False
        #print("after: ", self.state)
        #print("############################# END ENV 1 ################################")

        return self.state, reward, done, truncation, info
    
    def render(self, mode='human'):
        pass
            
    def Get_State(self, p):
        p_v = p.astype(np.float32)
        p_v = np.reshape(p, -1)
        x_v = self.x / self.F
        y_v = np.reshape(self.y, -1) / self.F
        t_v = self.t / self.H
        t_v = np.array([t_v], dtype=np.float32)
        xy_v = np.concatenate([x_v, y_v], dtype=np.float32)
        
        self.state = {"p": p_v, "xy": xy_v, "t": t_v}
        
    def Action_Encoder_Pricing(self, p):
        return (p+1)/(2*self.Max_p)
    
    def IsTerminal(self):
        #self.t += 1
        return False
    
    def SetState(self, obs_dict):
        self.state = {k: np.array(v, copy=True) for k, v in obs_dict.items()}

        p1 = self.state["p"] 
        p2= np.reshape(p1, [self.N, self.N])        
        self.p = p2
        xy = self.state["xy"]
        self.x = xy[0:self.N] * self.F
        self.x = np.round(self.x)
        self.x = self.x.astype(int)
        y1 = xy[self.N: self.N+self.N**2] *self.F
        y1 = np.round(y1)
        self.y = self.y.astype(int)
        self.y = np.reshape(y1, [self.N, self.N])      
        self.t = self.state["t"][0] * self.H
        self.t = self.t.astype(int)
        
