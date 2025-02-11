# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:43:17 2023

@author: ThomasDM
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RideHailingEnv(gym.Env):

    def __init__(self, Lambda, tau, distance,
                 F = 1200, N = 8, H = 36, C_q = 0.4, C_r = 0.2, C_ab = 0.0, beta0 = -0.15, beta1 = 8.0, beta2 = -1.5, temp = 1.0,
                 x_0 = np.array([192, 221, 73, 172, 144, 150, 146, 102], dtype=int), y_0 = np.zeros([8,8], dtype=int),
                 ):
        super(RideHailingEnv, self).__init__()
        
        self.Lambda = Lambda #Potential arrival rates
        self.tau = tau #Average transit rates
        self.distance = distance 
        self.Prob_Trip_Completed = 1 - np.exp(-tau) # Transit times are exponentially distributed 
                
        self.F = F #Fleet size
        self.N = N #Locations
        self.H = H #Decision times
        self.C_q = C_q #Lost sale cost
        self.C_r = C_r #Repositioning cost rate 
        self.C_ab = C_ab #Abandonment cost 
        self.beta0 = beta0 #MNL bias
        self.beta1 = beta1 #MNL coefficient for price
        self.beta2 = beta2 #MNL coefficient for traveling time
        self.temp = temp
        
        self.expected_time = np.copy(1/tau)
        np.fill_diagonal(self.expected_time, 0) 

        #self.expected_time_normalized = (self.expected_time_normalized-np.mean(self.expected_time_normalized, axis=1, keepdims=True))
        #self.expected_time_normalized = (self.expected_time_normalized-np.mean(self.expected_time_normalized, axis=1, keepdims=True))/np.std(self.expected_time_normalized, axis=1,keepdims=True)
        
        self.x_0 = x_0 #initial state
        self.y_0 = y_0

        self.action_space = spaces.Box(low = -1, high = 1, shape=(N**2 + N,), dtype = np.float32)         
        self.observation_space = spaces.MultiDiscrete(np.array([F+1] * N**2 + [F+1] * N + [H+1]), dtype = int)
        
        self.Max_a = self.action_space.high[0]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.copy(self.x_0)
        self.y = np.copy(self.y_0)
        self.t = 0
        self.Get_State()
        return self.state, {}
    
    def step(self, action):
        
        info = {}
        
        # Actions
        a = action[0 : self.N ** 2]
        a = np.reshape(a, [self.N, self.N])        
        q, rho = self.Action_Encoder_Repositioning_Rejection(a)
        info["Perc rejection customers"] = q
        info["Perc reposition drivers"] = rho
        p = action[self.N ** 2 : self.N ** 2 + self.N]
        p = self.Action_Encoder_Pricing(p)
        info["Prices"] = p
        # Deterministic events 
        
        stay_1, r = self.RepositionDrivers(rho)
        info["Repositioned drivers"] = r   
        # Random events
        
        stay_2, w = self.SelfRelocateDrivers(stay_1, p)
        e, c, lambda_ = self.GenerateArrivals(p, q)
        stay_3, d = self.DispatchDrivers(stay_2, e)
        z = self.CompleteServiceDrivers(r, w, d)

        info["Self-relocating drivers"] = w        
        info["Realized requests"] = c
        info["Arrival rates"] = lambda_        
        info["Admitted customers"] = e
        info["Dispatched drivers"] = d
        info["Drivers completing service"] = z

        # State transition 
                
        self.DriversAvailable(stay_3, z)        
        self.DriversTraveling(r, w, d, z)
        
        info["Requests satisfied"] = np.minimum(c, d)
        info["Requests unsatisfied"] = np.maximum(c - d, 0)
        info["Requests abandoning"] = np.maximum(e - d, 0)

        # Reward function 
        
        Revenue = self.ComputeRevenue(d, p)
        Repositioning = self.ComputeRepositioningCosts(r)
        LostSales = self.ComputeLostSaleCosts(c, d)
        Abandonment = self.ComputeAbandonmentCosts(e, d) 
        
        reward = Revenue - Repositioning - LostSales - Abandonment

        info["Revenue"] = Revenue
        info["Repositioning costs"] = Repositioning
        info["Lost sales costs"] = LostSales
        
        # Termination ? 
        
        done = self.IsTerminal()
        self.Get_State()

        truncation = False
        return self.state, reward, done, truncation, info
    
    def render(self, mode='human'):
        pass
    
    def Get_State(self):
        y_v = np.reshape(self.y, -1)
        x_v = self.x
        t_v = np.array([self.t], dtype=int)
        self.state = np.concatenate([y_v, x_v, t_v], dtype=int)
    
    def Action_Encoder_Pricing(self, a):
        return (a+1)/(2*self.Max_a)
        
    def Action_Encoder_Repositioning_Rejection(self, a):
        q = -np.minimum(a, 0) 
        np.fill_diagonal(q, 0) # never rejects customers requesting service in the same location if drivers are available.

        rho = np.maximum(a, 0)
        for i in range(self.N): #divider greater than 0 
            if rho.sum(axis=1, keepdims=True)[i]==0.0:
                b = rho[i]
                b[b==0] = 1e-6
        rho = rho/rho.sum(axis=1, keepdims=True)
        return q, rho
    
    def RepositionDrivers(self, rho):
        r = rho * self.x[:, None]
        Rest = r - np.floor(r)
        r = np.floor(r)
        Rest_sum = Rest.sum(axis=1, keepdims=False)
        for i in range(self.N):
                r[i, i] += Rest_sum[i]
                r[i, i] = round(r[i, i])
        r = r.astype(int)
        stay = np.copy(np.diagonal(r))
        np.fill_diagonal(r, 0) #Drivers repositioning are only those leaving the location.
        return stay, r

    def SelfRelocateDrivers(self, stay_init, p):
        self.p_normalized = p  - p[:, None]
        Utility = self.beta0 + self.beta1 * self.p_normalized + self.beta2 * self.expected_time 
        np.fill_diagonal(Utility, 0) # Utility for staying in the location is normalized to 1.
        #print("period: ", self.t, " Utility: ", Utility)
        Exp_Coefficient = np.exp(Utility/self.temp)/np.sum(np.exp(Utility/self.temp), axis=1, keepdims=True)
        w = np.zeros([self.N, self.N])
        for i in range(self.N): 
            w[i] = np.random.multinomial(stay_init[i], Exp_Coefficient[i])
        w = w.astype(int)
        stay_final = np.zeros(self.N, dtype=int)
        stay_final = np.copy(np.diagonal(w))
        np.fill_diagonal(w, 0) #Drivers self-relocating are only those leaving the location.
        return stay_final, w
    
    def GenerateArrivals(self, p, q):
        prob_pay = 1 - p #probability that a customer is willing to pay the price
        prob_acc = 1 - q #probability that a customer is admitted
        lambda_ = self.Lambda[:,:,self.t] * prob_pay[:, None] 
        c = np.random.poisson(lambda_)
        c = c.astype(int)
        e = np.random.binomial(c, prob_acc)
        return e, c, lambda_
    
    def DispatchDrivers(self, stay_init, e):
        d = np.zeros([self.N, self.N])
        stay_final = np.zeros(self.N, dtype=int)
        for i in range(self.N): 
            if np.sum(e[i]) <= stay_init[i]:
                d[i] = e[i]
                stay_final[i] = stay_init[i] - np.sum(e[i])
            else:
                gen = np.random.Generator(np.random.PCG64(4861946401452))     
                d[i] = gen.multivariate_hypergeometric(e[i], stay_init[i])
                stay_final[i] = 0
        d = d.astype(int)
        return stay_final, d
    
    def CompleteServiceDrivers(self, r, w, d):
        y = self.y + r + w + d
        z = np.random.binomial(y, self.Prob_Trip_Completed)
        return z
    
    def DriversAvailable(self, stay, z):
        self.x = stay + z.sum(axis=0)
        
    def DriversTraveling(self, r, w, d, z):
        self.y += r + w + d - z

    def ComputeRevenue(self, d, p):
        Adjusted_Price = p[:, None]
        Adjusted_Price = Adjusted_Price * self.distance
        Revenue = Adjusted_Price * d 
        Revenue = np.sum(Revenue)
        return Revenue 
    
    def ComputeRepositioningCosts(self, r):
        Repositioning = self.C_r * self.distance
        Repositioning = Repositioning * r
        Repositioning = np.sum(Repositioning)
        return Repositioning
    
    def ComputeLostSaleCosts(self, c, d):
        LostSales = c - d
        LostSales = LostSales * self.C_q
        LostSales = np.sum(LostSales)
        return LostSales
    
    def ComputeAbandonmentCosts(self, e, d):
        Abandonment = e - d
        Abandonment = Abandonment * self.C_ab
        Abandonment = np.sum(Abandonment)
        return Abandonment
    
    def IsTerminal(self):
        self.t += 1
        if self.t == self.H:
            return True
        else:
            return False
        