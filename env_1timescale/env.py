# -*- coding: utf-8 -*-
"""
Created on Thu April 15 15:16:32 2026

@author: thoma
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Env(gym.Env):

    def __init__(self, Lambda, tau, distance,
                 F = 900, N = 8, H = 36, p_interval = 3, C_q = 0.4, C_r = 0.2, beta0 = -0.15, beta1 = 8.0, beta2 = -1.5, temp = 1.0, x_0 = np.array([192, 221, 73, 172, 144, 150, 146, 102], dtype=int), y_0 = np.zeros([8,8], dtype=int),
                 ):
        super(Env, self).__init__()
        
        self.Lambda = Lambda #Potential arrival rates
        self.tau = tau #Average transit rates
        self.distance = distance 
        self.Prob_Trip_Completed = 1 - np.exp(-tau) # Transit times are exponentially distributed 
                
        self.F = F #Fleet size
        self.N = N #Locations
        self.H = H #Decision times
        self.p_interval = p_interval
        self.C_q = C_q #Lost sale cost
        self.C_r = C_r #Repositioning cost rate 
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

        self.action_space = spaces.Box(low = -1, high = 1, shape=(N**2 + N**2,), dtype = np.float32)         
        self.observation_space = spaces.Box(low = 0, high = 1, shape=(N+N**2 + 1,), dtype = np.float32)
                
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.copy(self.x_0)
        self.y = np.copy(self.y_0)
        self.t = 0
        self.Get_State()
        return self.state, {}
    
    def step(self, action):
        
        info = {}
        
        #Update prices 
        #print("p 2: ", self.state["p"])
        #print("st 2: ", self.state["xyt"])
        #print("############################# BEGIN ENV 2 ################################")
        #print("b4: ", self.state)

        # Actions
        a = action[0 : self.N ** 2]
        a = np.reshape(a, [self.N, self.N])        
        rho = self.Action_Encoder_Repositioning(a)
        info["Perc reposition drivers"] = rho
        p = action[self.N ** 2 : 2 * self.N ** 2]
        p = np.reshape(p, [self.N, self.N])
        p = self.Action_Encoder_Pricing(p)
        info["Prices"] = p

        stay_1, r = self.RepositionDrivers(rho)
        info["Repositioned drivers"] = r   
        

        # Random events
        c, lambda_ = self.GenerateArrivals(p)
        stay_2, w = self.SelfRelocateDrivers(stay_1, p)
        stay_3, d = self.DispatchDrivers(stay_2, c)
        z = self.CompleteServiceDrivers(r, w, d)

        info["Self-relocating drivers"] = w        
        info["Realized requests"] = c
        info["Arrival rates"] = lambda_       
        info["Dispatched drivers"] = d
        info["Drivers completing service"] = z

        # State transition 
        
        self.DriversAvailable(stay_3, z)        
        self.DriversTraveling(r, w, d, z)
        
        info["Requests satisfied"] = np.minimum(c, d)
        info["Requests unsatisfied"] = np.maximum(c - d, 0)

        # Reward function 
        
        Revenue = self.ComputeRevenue(d, p)
        Repositioning = self.ComputeRepositioningCosts(r)
        LostSales = self.ComputeLostSaleCosts(c, d)
        
        reward = Revenue - Repositioning - LostSales 
        info["Revenue"] = Revenue
        info["Repositioning costs"] = Repositioning
        info["Lost sales costs"] = LostSales
        
        # Termination ? 
        
        done = self.IsTerminal()
        self.Get_State()
        
        truncation = False
        reward_bis = reward / 100
        #print("after: ", self.state)
        #print("############################# END ENV 2 ################################")

        return self.state, reward_bis, done, truncation, info
    
    def render(self, mode='human'):
        pass
    
    #def Get_State1(self):
    #    x_v = self.x
    #    y_v = np.reshape(self.y, -1)
    #    t_v = np.array([self.t], dtype=int)
    #    self.state = np.concatenate([x_v, y_v, t_v], dtype=int)

    def Get_State(self):
        
        y_v = np.reshape(self.y, -1)
        y_v = np.reshape(self.y, -1)/self.F
        x_v = self.x
        x_v = self.x/self.F
        t_v = self.t / self.H
        t_v = np.array([t_v], dtype=np.float32)
        self.state = np.concatenate([y_v, x_v, t_v], dtype=np.float32)
        
    def Action_Encoder_Repositioning(self, a):
        rho = np.maximum(a, 0)
        for i in range(self.N): #divider greater than 0 
            if rho.sum(axis=1, keepdims=True)[i]==0.0:
                b = rho[i]
                b[b==0] = 1e-6
        rho2 = rho/rho.sum(axis=1, keepdims=True)
        return rho2
    
    def Action_Encoder_Pricing(self, a):
        return (a+1)/2
    
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
        masked = np.ma.masked_equal(p, 1)
        self.p_weighted = masked.mean(axis=1)
        self.p_normalized = self.p_weighted - self.p_weighted[:, None]

        Utility = self.beta0 + self.beta1 * self.p_normalized + self.beta2 * self.expected_time 
        np.fill_diagonal(Utility, 0) # Utility for staying in the location is normalized to 1.
        #print("period: ", self.t, " Utility: ", Utility)
        Exp_Coefficient = np.exp(Utility/self.temp)/np.sum(np.exp(Utility/self.temp), axis=1, keepdims=True)
        Exp_Coefficient = Exp_Coefficient.filled(0.0).astype(np.float64)
        w = np.zeros([self.N, self.N])
        
        for i in range(self.N): 
            w[i] = np.random.multinomial(stay_init[i], Exp_Coefficient[i])
        w = w.astype(int)
        stay_final = np.zeros(self.N, dtype=int)
        stay_final = np.copy(np.diagonal(w))
        np.fill_diagonal(w, 0) #Drivers self-relocating are only those leaving the location.
        return stay_final, w
    
    def GenerateArrivals(self, p):
        prob_pay = 1 - p #probability that a customer is willing to pay the price
        lambda_ = self.Lambda[:,:,self.t] * prob_pay
        c = np.random.poisson(lambda_)
        c = c.astype(int)
        return c, lambda_
    
    def DispatchDrivers(self, stay_init, c):
        d = np.zeros([self.N, self.N])
        stay_final = np.zeros(self.N, dtype=int)
        for i in range(self.N): 
            if np.sum(c[i]) <= stay_init[i]:
                d[i] = c[i]
                stay_final[i] = stay_init[i] - np.sum(c[i])
            else:
                #print("c: ", c)
                gen = np.random.Generator(np.random.PCG64(4861946401452))     
                d[i] = gen.multivariate_hypergeometric(c[i], stay_init[i])
                stay_final[i] = 0
        d = d.astype(int)
        return stay_final, d
    
    def CompleteServiceDrivers(self, r, w, d):
        y = self.y + r + w + d
        z = np.random.binomial(y, self.Prob_Trip_Completed)
        return z
    
    def SetState(self, obs_dict):
        self.state = {k: np.array(v, copy=True) for k, v in obs_dict.items()}

        p1 = self.state["p"]
        p2= np.reshape(p1, [self.N, self.N])        
        self.p = p2
        xyt = self.state["xyt"]
        self.x = xyt[0:self.N]
        y1 = xyt[self.N: self.N+self.N**2]
        self.y = np.reshape(y1, [self.N, self.N])      
        self.t = xyt[-1]
        
    def DriversAvailable(self, stay, z):
        self.x = stay + z.sum(axis=0)
        
    def DriversTraveling(self, r, w, d, z):
        self.y += r + w + d - z

    def ComputeRevenue(self, d, p):
        Adjusted_Price = p
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
    
    def IsTerminal(self):
        self.t += 1
        if self.t == self.H:
            return True
        else:
            return False        
        