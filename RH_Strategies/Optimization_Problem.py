# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:15:52 2024

@author: ThomasDM
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

class EVP():
    def __init__(self, Lambda, tau, distance,
                 F = 900, N = 8, opt_horizon = 4,
                 C_q = 0.4, C_r = 0.2):
        #Parameters
        self.Lambda = Lambda
        self.tau = tau
        self.distance = distance
        self.Prob_Trip_Completed = 1 - np.exp(-tau)
        self.F= F
        
        self.C_q = C_q
        self.C_r = C_r
        
        self.N = N
        self.opt_horizon = opt_horizon
        
        self.x_start = np.array([F/N] * N)
        self.y_start = np.zeros([N, N])
        self.zeta = np.zeros([N, N, opt_horizon])
        
    def DefineModel(self, Lambda, x_start, y_start, opt_horizon):
        self.Lambda = Lambda
        self.x_start = x_start
        self.y_start = y_start
        self.opt_horizon = opt_horizon 
        
        self.model = gp.Model('RideHailing')
        self.model.params.LogToConsole = 1
        self.model.params.nonConvex = 2
        self.model.params.MIPGap = 0.01
        self.model.params.TimeLimit = 120
        
        for i in range(self.N):
            for j in range(self.N):
                for t in range(self.opt_horizon):
                    self.zeta[i,j,t] = self.Prob_Trip_Completed[i,j] * (1.0-self.Prob_Trip_Completed[i,j])**(t)

        # Variables
        self.x = self.model.addVars(self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
        self.p = self.model.addVars(self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="p")
        self.d = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="d")
        self.c = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
        self.sigma = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="sigma")
        self.r = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="r")
        
        # Objective 

        self.Revenue = gp.quicksum(self.p[i, t] * self.distance[i, j] * self.d[i, j, t] for i in range(self.N) 
                               for j in range(self.N) for t in range(self.opt_horizon))
        
        self.Rejection = gp.quicksum(self.C_q * self.sigma[i, j, t]  for i in range(self.N) 
                       for j in range(self.N) for t in range(self.opt_horizon))
        
        self.Repositioning = gp.quicksum(self.C_r * self.distance[i, j] * self.r[i, j, t] 
                               for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon))
        
        self.model.setObjective(self.Revenue - self.Repositioning - self.Rejection, GRB.MAXIMIZE)

        # Constraints
        
        self.Dispatch_DiffLoc = self.model.addConstrs(self.d[i, j, t] == self.c[i, j, t] - self.sigma[i, j, t] for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon) if i!=j)
        
        self.Dispatch_SameLoc = self.model.addConstrs(self.d[i, i, t] == self.c[i, i, t] for i in range(self.N) for t in range(self.opt_horizon))

        self.Demand = self.model.addConstrs(self.c[i, j, t] == self.Lambda[i, j, t] * (1 - self.p[i, t]) for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon))
        
        self.Limited_Supply = self.model.addConstrs(self.x[i, t] >= gp.quicksum(self.d[i, j, t] + self.r[i, j, t] for j in range(self.N))
                               for i in range(self.N) for t in range(self.opt_horizon))
        
        self.Supply_Conservation = self.model.addConstrs(self.x[i, t] == self.x[i, t-1] - gp.quicksum(self.d[i, j, t-1] + self.r[i, j, t-1] for j in range(self.N))
                                       + gp.quicksum(self.zeta[j, i, u] * (self.d[j, i, t-u-1] + self.r[j, i, t-u-1]) for j in range(self.N) for u in range(t))
                                       + gp.quicksum(self.zeta[j, i, t-1] * self.y_start[j, i] for j in range(self.N))
                                       for i in range(self.N) for t in range(1, self.opt_horizon))

        self.Supply_Conservation_0 = self.model.addConstrs(self.x[i, 0] == self.x_start[i] for i in range(self.N))
        #self.model.write('RideHailing.lp')
    
    def OptimizeModel(self):
        self.model.reset()
        self.model.optimize()
        
    def ConvertToImplementableAction(self):
        
        #Store the solutions in arrays        
        x_array = np.zeros([self.N, self.opt_horizon])
        p_array = np.zeros([self.N, self.opt_horizon])
        d_array = np.zeros([self.N, self.N, self.opt_horizon])
        c_array = np.zeros([self.N, self.N, self.opt_horizon])
        sigma_array = np.zeros([self.N, self.N, self.opt_horizon])
        r_array = np.zeros([self.N, self.N, self.opt_horizon])
        
        for i in range(self.N):
            for t in range(self.opt_horizon):
                x_array[i, t] = self.x[i, t].x
                p_array[i, t] = self.p[i, t].x
                
        for i in range(self.N):
            for j in range(self.N):
                for t in range(self.opt_horizon):
                    d_array[i, j, t] = self.d[i, j, t].x
                    c_array[i, j, t] = self.c[i, j, t].x
                    sigma_array[i, j, t] = self.sigma[i, j, t].x
                    r_array[i, j, t] = self.r[i, j, t].x

        stay_array = x_array - np.sum(r_array, axis=1) #drivers who are left available
        
        #print(np.sum(r_array) - np.sum(np.maximum(r_array-sigma_array, 0)))
        
        #Rejection decisions 
        Rejection_Perc = np.zeros([self.N, self.N, self.opt_horizon])
        for i in range(self.N):
            for j in range(self.N): 
                for t in range(self.opt_horizon):
                    if c_array[i, j, t] == 0.0:
                        c_array[i, j, t] = 1e-6
                    Rejection_Perc[i, j, t] = sigma_array[i, j, t]/c_array[i, j, t]
                    
        #Repositioning decisions
        for i in range(self.N):
            for t in range(self.opt_horizon):
                r_array[i, i, t] = stay_array[i, t]
                
        Repositioning_Perc = np.random.rand(self.N, self.N, 0)
        for t in range(self.opt_horizon):
            r = r_array[:, :, t]
            for i in range(self.N):
                if r.sum(axis=1, keepdims=True)[i]<1e-4:
                    r[i, i] = 1.0
            perc = r/r.sum(axis=1, keepdims=True)
            Repositioning_Perc = np.dstack((Repositioning_Perc, perc))
        Action_Perc = Repositioning_Perc - Rejection_Perc
        Action_Perc = Action_Perc.reshape(self.N ** 2, self.opt_horizon)
        Action_Perc = Action_Perc.transpose()
                
        #Pricing decision
        Pricing = p_array.transpose()
        Pricing = self.Action_Encoder_Pricing(Pricing)
        #Build matrix of actions for the next periods
        action = np.concatenate([Action_Perc, Pricing], axis=1)  
        return action
    
    def Action_Adapter_Dispatching(self, a):
        return (2*a - 1)/1
    
    def Action_Encoder_Pricing(self, a):
        a = (2*a - 1)/1
        return a