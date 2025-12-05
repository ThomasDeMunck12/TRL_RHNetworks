# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:51:07 2025

@author: thoma
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

class EVP():
    def __init__(self, Lambda, tau, distance, mu,
                 F = 900, N = 8, opt_horizon = 4, p_interval = 2,
                 C_q = 0.4, C_r = 0.2):
        #Parameters
        self.Lambda = Lambda
        self.tau = tau
        self.distance = distance
        self.mu = mu
        self.Prob_Trip_Completed = 1 - np.exp(-tau)
        self.F= F
        
        self.C_r = C_r
        self.C_q = C_q

        self.N = N
        self.opt_horizon = opt_horizon
        
        self.p_interval = p_interval 
        
        self.x_start = np.array([F/N] * N)
        self.y_start = np.zeros([N, N])
        self.zeta = np.zeros([N, N, opt_horizon])
        
    def DefineModel(self, Lambda, x_start, y_start, p_start, t_start, exp_coef, opt_horizon):
        self.Lambda = Lambda
        self.x_start = x_start
        self.y_start = y_start
        #self.t_start = t_start 
        self.p_start = p_start
        self.opt_horizon = opt_horizon 
        self.model = gp.Model('RideHailing')
        self.model.params.LogToConsole = 0
        self.model.params.nonConvex = 2
        self.model.params.MIPGap = 0.05
        self.model.params.TimeLimit = 30
        
        self.n_step_start = (self.p_interval - t_start % self.p_interval) % self.p_interval
        self.n_step_end = (self.opt_horizon - self.n_step_start ) % self.p_interval
        self.n_periods = (self.opt_horizon - self.n_step_start - self.n_step_end) // self.p_interval
        #print("period: ", t_start, " - start: ", self.n_step_start, ", end: ", self.n_step_end, ", between: ", self.n_periods*self.p_interval, " - ", opt_horizon)
        
        for i in range(self.N):
            for j in range(self.N):
                for t in range(self.opt_horizon):
                    self.zeta[i,j,t] = self.Prob_Trip_Completed[i,j] * (1.0-self.Prob_Trip_Completed[i,j])**(t)

        # Self-relocation 
        self.exp_coef = exp_coef
        #print(self.exp_coef[:,:,0])

        # Variables
        self.x = self.model.addVars(self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
        self.p = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="p")
        self.d = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="d")
        self.c = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
        self.sigma = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="sigma")
        self.r = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="r")
        self.w = self.model.addVars(self.N, self.N, self.opt_horizon, vtype=GRB.CONTINUOUS, lb=0.0, name="w")

        # Objective 

        self.Revenue = gp.quicksum(self.p[i, j, t] * self.distance[i, j] * self.d[i, j, t] for i in range(self.N) 
                               for j in range(self.N) for t in range(self.opt_horizon))
                
        self.Repositioning = gp.quicksum(self.C_r * self.distance[i, j] * self.r[i, j, t] 
                               for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon))
        
        self.LostSales = gp.quicksum(self.C_q * self.sigma[i, j, t]  for i in range(self.N) 
                       for j in range(self.N) for t in range(self.opt_horizon))
        
        self.model.setObjective(self.Revenue - self.Repositioning - self.LostSales, GRB.MAXIMIZE)

        # Constraints
        
        self.Demand = self.model.addConstrs(self.c[i, j, t] == self.Lambda[i, j, t] * (1 - self.p[i, j, t]) for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon))
        #self.Demand = self.model.addConstrs(self.c[i, j, t] == self.Lambda[i, j, t] * (1 - self.p[i, t]) for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon))

        self.Dispatch = self.model.addConstrs(self.d[i, j, t] == self.c[i, j, t] - self.sigma[i, j, t] for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon))

        self.Limited_Supply = self.model.addConstrs(self.x[i, t] >= gp.quicksum(self.d[i, j, t] + self.r[i, j, t] + self.w[i, j, t] for j in range(self.N))
                               for i in range(self.N) for t in range(self.opt_horizon))
        #
        self.SelfReloc = self.model.addConstrs(self.w[i, j, t] == (self.x[i, t] - gp.quicksum(self.r[i, j, t] for j in range(self.N))) * self.exp_coef[i, j, t] for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon) if i!=j)
        
        self.SelfReloc0 = self.model.addConstrs(self.w[i, i, t] == 0.0 for i in range(self.N) for t in range(self.opt_horizon))

        self.Supply_Conservation = self.model.addConstrs(self.x[i, t] == self.x[i, t-1] - gp.quicksum(self.d[i, j, t-1] + self.r[i, j, t-1] + self.w[i, j, t-1] for j in range(self.N))
                                       + gp.quicksum(self.zeta[j, i, u] * (self.d[j, i, t-u-1] + self.r[j, i, t-u-1] + self.w[j, i, t-u-1]) for j in range(self.N) for u in range(t))
                                       + gp.quicksum(self.zeta[j, i, t-1] * self.y_start[j, i] for j in range(self.N)) + self.mu[i, t-1] for i in range(self.N) for t in range(1, self.opt_horizon))
        
        self.Supply_Conservation_0 = self.model.addConstrs(self.x[i, 0] == self.x_start[i] for i in range(self.N))
        
        self.Price_0 = self.model.addConstrs(self.p[i, j, t] == p_start[i, j] for i in range(self.N) for j in range(self.N) for t in range(self.n_step_start))
        self.Price_int = self.model.addConstrs(self.p[i, j, self.n_step_start + k*self.p_interval] == self.p[i, j, self.n_step_start + k*self.p_interval + t] for i in range(self.N) for j in range(self.N) for k in range(self.n_periods) for t in range(self.p_interval))
        self.Price_T = self.model.addConstrs(self.p[i, j, t] == self.p[i, j, self.opt_horizon-1] for i in range(self.N) for j in range(self.N) for t in range(self.opt_horizon - self.n_step_end, self.opt_horizon))
        #self.Price_Same = self.model.addConstrs(self.p[i, j, t] == self.p[i, k, t] for i in range(self.N) for j in range(self.N) for k in range(self.N) for t in range(self.opt_horizon))

        #self.model.write('RideHailing.lp')
    
    def OptimizeModel(self):
        self.model.reset()
        self.model.optimize()
        #print(self.model.ObjVal)

    def ConvertToImplementableAction(self):
        
        #Store the solutions in arrays        
        x_array = np.zeros([self.N, self.opt_horizon])
        p_array = np.zeros([self.N, self.N, self.opt_horizon])
        c_array = np.zeros([self.N, self.N, self.opt_horizon])
        r_array = np.zeros([self.N, self.N, self.opt_horizon])
        w_array = np.zeros([self.N, self.N, self.opt_horizon])

        for i in range(self.N):
            for t in range(self.opt_horizon):
                x_array[i, t] = self.x[i, t].x
        for i in range(self.N):
            for j in range(self.N):
                for t in range(self.opt_horizon):
                    p_array[i, j, t] = self.p[i, j, t].x
                    c_array[i, j, t] = self.c[i, j, t].x
                    r_array[i, j, t] = self.r[i, j, t].x            
                    w_array[i, j, t] = self.w[i, j, t].x
    
        stay_array = x_array - np.sum(r_array, axis=1) #- np.sum(w_array, axis=1)
        # #drivers who are left available
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
        Repositioning_Perc = Repositioning_Perc.reshape(self.N ** 2, self.opt_horizon)
        Repositioning_Perc = Repositioning_Perc.transpose()
        Repositioning_Perc = self.Action_Encoder_Repositioning(Repositioning_Perc)
        #print(Repositioning_Perc)
        #Pricing decision
        Pricing = p_array
        Pricing = Pricing.reshape(self.N ** 2, self.opt_horizon)
        Pricing = Pricing.transpose()
        Pricing = self.Action_Encoder_Pricing(Pricing)

        #Build matrix of actions for the next periods
        return p_array, r_array, Repositioning_Perc, Pricing
    
    def Action_Encoder_Repositioning(self, a):
        a = np.where(a > 0.00001, a, -0.1)
        #np.set_printoptions(threshold=1000000)

        return a
    
    def Action_Encoder_Pricing(self, a):
        return (2*a - 1)/1