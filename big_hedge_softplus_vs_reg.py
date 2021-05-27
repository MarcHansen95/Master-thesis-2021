#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:36:31 2021

@author: MarcHansen
"""


#This is a hedge experiment of pol. reg vs. Softplus reg.
# We do not use this in the thesis

from scipy.stats import norm
import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize



poly = np.arange(4, 7, 1)
random.seed(3)
S0 = 1
T = 1
K = 1
vol = 0.2
Nhedge = 52
TruePrice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]
max_power = 9
sn = 6000
sim_num = np.arange(1000, sn+1000, 1000)
big_hedge_error = np.ones((max_power-2,8))
v = 0
for l in range(-1000,sn+1000,1000):  
    if l < -900:
        l=300
    elif l < 10:
        l = 600
    else:
        l = l
    print(l)    
    Npaths = 10000
    N = l
    n =5
    dt = T/Nhedge
    u = 0
    print(v)
    for q in range(1, n+1, 1):
        print(q)
        if q ==1:
            x = np.array([0,0,0,0])
        elif q ==2:
            x = np.array([-0.02346599,  0.12785693,  7.83448937, -7.5980519,0,0,0])
        elif q == 3:
            x = np.array([-0.02346599,  0.12785693,  7.83448937, -7.5980519,  0, 0, 0 ,0,0,0])
        elif q == 4:
            x = np.array([-0.02346599,  0.12785693,  7.83448937, -7.5980519,  0, 0, 0 ,0,0,0,0,0,0])
        elif q == 5:
            x = np.array([-0.02346599,  0.12785693,  7.83448937, -7.5980519,  0, 0, 0 ,0,0,0,0,0,0,0,0,0])

        # Get coefeceints from softplus regress

        Coef = np.zeros((1+3*q,Nhedge))
       
        Sim1 = random.normal(0,1,N)
        Sim2 = random.normal(0,1,N)
        
        CallDelta = np.zeros((N,))
        
        for i in range(0,52):
            
            S1 = S0+vol*np.sqrt(T)*Sim1
            S = S1
            S2 = S1+vol*np.sqrt(T-(i-1)*dt)*Sim2
            CallPayoff = np.maximum(S2-K,0)
            CallPrice = CallPayoff
            for j in range(0,len(CallDelta)):     
                if S2[j]>=K:
                    CallDelta[j] = 1
                else:
                    CallDelta[j] = 0
            tau = np.std(CallPayoff)/np.std(CallDelta)
            w = 1/(1+tau)
            def criterion(beta):
                if len(beta)==4:
                        dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])
                        dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]
                elif len(beta)==7:
                        dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])+beta[4]*fl.SoftPlus(beta[5]*S+beta[6])
                        dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]
                elif len(beta)==10:
                        dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])+beta[4]*fl.SoftPlus(beta[5]*S+beta[6])+beta[7]*fl.SoftPlus(beta[8]*S+beta[9])
                        dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]+beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])*beta[8]
                elif len(beta)==13:
                        dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])+beta[4]*fl.SoftPlus(beta[5]*S+beta[6])+beta[7]*fl.SoftPlus(beta[8]*S+beta[9])+beta[10]*fl.SoftPlus(beta[11]*S+beta[12])
                        dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]+beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])*beta[8]+beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])*beta[11]
                elif len(beta)==16:
                        dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])+beta[4]*fl.SoftPlus(beta[5]*S+beta[6])+beta[7]*fl.SoftPlus(beta[8]*S+beta[9])+beta[10]*fl.SoftPlus(beta[11]*S+beta[12])+beta[13]*fl.SoftPlus(beta[14]*S+beta[15])
                        dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]+beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])*beta[8]+beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])*beta[11]+beta[13]*fl.SoftPlus_dif(beta[14]*S+beta[15])*beta[14]
                return(w*np.sum((CallPrice-dummy)**2)+(1-w)*np.sum((CallDelta-dif_dummy)**2))
                
            def Gradient(beta):
                if len(beta)==4:
                    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])-CallPrice
                    dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]-CallDelta
                    grad = np.zeros((len(beta),))
                    
                    d1 = 1
                    dif_d1 = 0
                    grad[0] = w*2*np.sum(dummy*d1)+(1-w)*2*np.sum(dif_dummy*dif_d1)
                    
                    d2 = fl.SoftPlus(beta[2]*S+beta[3])
                    dif_d2 = fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]
                    grad[1] = w*2*np.sum(dummy*d2)+(1-w)*2*np.sum(dif_dummy*dif_d2)
                    
                    d3 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*S
                    dif_d3 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*S*beta[2]+beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    grad[2] = w*2*np.sum(dummy*d3)+(1-w)*2*np.sum(dif_dummy*dif_d3)
                    
                    d4 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    dif_d4 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*beta[2]
                    grad[3] = w*2*np.sum(dummy*d4)+(1-w)*2*np.sum(dif_dummy*dif_d4)
                        
                elif len(beta)==7:
                    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])-CallPrice
                    dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]-CallDelta
                    grad = np.zeros((len(beta),))
                        
                    d1 = 1
                    dif_d1 = 0
                    grad[0] = w*2*np.sum(dummy*d1)+(1-w)*2*np.sum(dif_dummy*dif_d1)
                        
                    d2 = fl.SoftPlus(beta[2]*S+beta[3])
                    dif_d2 = fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]
                    grad[1] = w*2*np.sum(dummy*d2)+(1-w)*2*np.sum(dif_dummy*dif_d2)
                        
                    d3 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*S
                    dif_d3 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*S*beta[2]+beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    grad[2] = w*2*np.sum(dummy*d3)+(1-w)*2*np.sum(dif_dummy*dif_d3)
                        
                    d4 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    dif_d4 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*beta[2]
                    grad[3] = w*2*np.sum(dummy*d4)+(1-w)*2*np.sum(dif_dummy*dif_d4)
                        
                    d5 = fl.SoftPlus(beta[5]*S+beta[6])
                    dif_d5 = fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]
                    grad[4] = w*2*np.sum(dummy*d5)+(1-w)*2*np.sum(dif_dummy*dif_d5)
                        
                    d6 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*S
                    dif_d6 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*S*beta[5]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    grad[5] = w*2*np.sum(dummy*d6)+(1-w)*2*np.sum(dif_dummy*dif_d6)
                        
                    d7 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    dif_d7 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*beta[5]
                    grad[6] = w*2*np.sum(dummy*d7)+(1-w)*2*np.sum(dif_dummy*dif_d7)
                elif len(beta)==10:
                    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])-CallPrice
                    dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]-CallDelta
                    grad = np.zeros((len(beta),))
                    
                    d1 = 1
                    dif_d1 = 0
                    grad[0] = w*2*np.sum(dummy*d1)+(1-w)*2*np.sum(dif_dummy*dif_d1)
                    
                    d2 = fl.SoftPlus(beta[2]*S+beta[3])
                    dif_d2 = fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]
                    grad[1] = w*2*np.sum(dummy*d2)+(1-w)*2*np.sum(dif_dummy*dif_d2)
                    
                    d3 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*S
                    dif_d3 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*S*beta[2]+beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    grad[2] = w*2*np.sum(dummy*d3)+(1-w)*2*np.sum(dif_dummy*dif_d3)
                    
                    d4 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    dif_d4 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*beta[2]
                    grad[3] = w*2*np.sum(dummy*d4)+(1-w)*2*np.sum(dif_dummy*dif_d4)
                    
                    d5 = fl.SoftPlus(beta[5]*S+beta[6])
                    dif_d5 = fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]
                    grad[4] = w*2*np.sum(dummy*d5)+(1-w)*2*np.sum(dif_dummy*dif_d5)
                    
                    d6 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*S
                    dif_d6 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*S*beta[5]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    grad[5] = w*2*np.sum(dummy*d6)+(1-w)*2*np.sum(dif_dummy*dif_d6)
                    
                    d7 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    dif_d7 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*beta[5]
                    grad[6] = w*2*np.sum(dummy*d7)+(1-w)*2*np.sum(dif_dummy*dif_d7)
                    
                    d8 = fl.SoftPlus(beta[8]*S+beta[9])
                    dif_d8 = fl.SoftPlus_dif(beta[8]*S+beta[9])*beta[8]
                    grad[7] = w*2*np.sum(dummy*d8)+(1-w)*2*np.sum(dif_dummy*dif_d8)
                    
                    d9 = beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])*S
                    dif_d9 = beta[7]*fl.SoftPlus_difdif(beta[8]*S+beta[9])*S*beta[8]+beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])
                    grad[8] = w*2*np.sum(dummy*d9)+(1-w)*2*np.sum(dif_dummy*dif_d9)
                    
                    d10 = beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])
                    dif_d10 = beta[7]*fl.SoftPlus_difdif(beta[8]*S+beta[9])*beta[8]
                    grad[9] = w*2*np.sum(dummy*d10)+(1-w)*2*np.sum(dif_dummy*dif_d10)
                        
                        
                elif len(beta)==13:
                    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])-CallPrice
                    dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]-CallDelta
                    grad = np.zeros((len(beta),))                   
                    d1 = 1
                    dif_d1 = 0
                    grad[0] = w*2*np.sum(dummy*d1)+(1-w)*2*np.sum(dif_dummy*dif_d1)
                    
                    d2 = fl.SoftPlus(beta[2]*S+beta[3])
                    dif_d2 = fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]
                    grad[1] = w*2*np.sum(dummy*d2)+(1-w)*2*np.sum(dif_dummy*dif_d2)
                    
                    d3 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*S
                    dif_d3 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*S*beta[2]+beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    grad[2] = w*2*np.sum(dummy*d3)+(1-w)*2*np.sum(dif_dummy*dif_d3)
                    
                    d4 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    dif_d4 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*beta[2]
                    grad[3] = w*2*np.sum(dummy*d4)+(1-w)*2*np.sum(dif_dummy*dif_d4)
                    
                    d5 = fl.SoftPlus(beta[5]*S+beta[6])
                    dif_d5 = fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]
                    grad[4] = w*2*np.sum(dummy*d5)+(1-w)*2*np.sum(dif_dummy*dif_d5)
                    
                    d6 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*S
                    dif_d6 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*S*beta[5]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    grad[5] = w*2*np.sum(dummy*d6)+(1-w)*2*np.sum(dif_dummy*dif_d6)
                    
                    d7 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    dif_d7 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*beta[5]
                    grad[6] = w*2*np.sum(dummy*d7)+(1-w)*2*np.sum(dif_dummy*dif_d7)
                    
                    d8 = fl.SoftPlus(beta[8]*S+beta[9])
                    dif_d8 = fl.SoftPlus_dif(beta[8]*S+beta[9])*beta[8]
                    grad[7] = w*2*np.sum(dummy*d8)+(1-w)*2*np.sum(dif_dummy*dif_d8)
                    
                    d9 = beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])*S
                    dif_d9 = beta[7]*fl.SoftPlus_difdif(beta[8]*S+beta[9])*S*beta[8]+beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])
                    grad[8] = w*2*np.sum(dummy*d9)+(1-w)*2*np.sum(dif_dummy*dif_d9)
                    
                    d10 = beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])
                    dif_d10 = beta[7]*fl.SoftPlus_difdif(beta[8]*S+beta[9])*beta[8]
                    grad[9] = w*2*np.sum(dummy*d10)+(1-w)*2*np.sum(dif_dummy*dif_d10)
                    
                    d11 = fl.SoftPlus(beta[11]*S+beta[12])
                    dif_d11 = fl.SoftPlus_dif(beta[11]*S+beta[12])*beta[11]
                    grad[10] = w*2*np.sum(dummy*d11)+(1-w)*2*np.sum(dif_dummy*dif_d11)
                    
                    d12 = beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])*S
                    dif_d12 = beta[10]*fl.SoftPlus_difdif(beta[11]*S+beta[12])*S*beta[11]+beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])
                    grad[11] = w*2*np.sum(dummy*d12)+(1-w)*2*np.sum(dif_dummy*dif_d12)
                    
                    d13 = beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])
                    dif_d13 = beta[10]*fl.SoftPlus_difdif(beta[11]*S+beta[12])*beta[11]
                    grad[12] = w*2*np.sum(dummy*d13)+(1-w)*2*np.sum(dif_dummy*dif_d13)
                        
                elif len(beta)==16:
                    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])-CallPrice
                    dif_dummy = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]-CallDelta
                    grad = np.zeros((len(beta),))
                        
                    d1 = 1
                    dif_d1 = 0
                    grad[0] = w*2*np.sum(dummy*d1)+(1-w)*2*np.sum(dif_dummy*dif_d1)
                        
                    d2 = fl.SoftPlus(beta[2]*S+beta[3])
                    dif_d2 = fl.SoftPlus_dif(beta[2]*S+beta[3])*beta[2]
                    grad[1] = w*2*np.sum(dummy*d2)+(1-w)*2*np.sum(dif_dummy*dif_d2)
                        
                    d3 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*S
                    dif_d3 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*S*beta[2]+beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    grad[2] = w*2*np.sum(dummy*d3)+(1-w)*2*np.sum(dif_dummy*dif_d3)
                        
                    d4 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
                    dif_d4 = beta[1]*fl.SoftPlus_difdif(beta[2]*S+beta[3])*beta[2]
                    grad[3] = w*2*np.sum(dummy*d4)+(1-w)*2*np.sum(dif_dummy*dif_d4)
                        
                    d5 = fl.SoftPlus(beta[5]*S+beta[6])
                    dif_d5 = fl.SoftPlus_dif(beta[5]*S+beta[6])*beta[5]
                    grad[4] = w*2*np.sum(dummy*d5)+(1-w)*2*np.sum(dif_dummy*dif_d5)
                        
                    d6 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*S
                    dif_d6 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*S*beta[5]+beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    grad[5] = w*2*np.sum(dummy*d6)+(1-w)*2*np.sum(dif_dummy*dif_d6)
                        
                    d7 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
                    dif_d7 = beta[4]*fl.SoftPlus_difdif(beta[5]*S+beta[6])*beta[5]
                    grad[6] = w*2*np.sum(dummy*d7)+(1-w)*2*np.sum(dif_dummy*dif_d7)
                        
                    d8 = fl.SoftPlus(beta[8]*S+beta[9])
                    dif_d8 = fl.SoftPlus_dif(beta[8]*S+beta[9])*beta[8]
                    grad[7] = w*2*np.sum(dummy*d8)+(1-w)*2*np.sum(dif_dummy*dif_d8)
                        
                    d9 = beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])*S
                    dif_d9 = beta[7]*fl.SoftPlus_difdif(beta[8]*S+beta[9])*S*beta[8]+beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])
                    grad[8] = w*2*np.sum(dummy*d9)+(1-w)*2*np.sum(dif_dummy*dif_d9)
                        
                    d10 = beta[7]*fl.SoftPlus_dif(beta[8]*S+beta[9])
                    dif_d10 = beta[7]*fl.SoftPlus_difdif(beta[8]*S+beta[9])*beta[8]
                    grad[9] = w*2*np.sum(dummy*d10)+(1-w)*2*np.sum(dif_dummy*dif_d10)
                           
                    d11 = fl.SoftPlus(beta[11]*S+beta[12])
                    dif_d11 = fl.SoftPlus_dif(beta[11]*S+beta[12])*beta[11]
                    grad[10] = w*2*np.sum(dummy*d11)+(1-w)*2*np.sum(dif_dummy*dif_d11)
                        
                    d12 = beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])*S
                    dif_d12 = beta[10]*fl.SoftPlus_difdif(beta[11]*S+beta[12])*S*beta[11]+beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])
                    grad[11] = w*2*np.sum(dummy*d12)+(1-w)*2*np.sum(dif_dummy*dif_d12)
                        
                    d13 = beta[10]*fl.SoftPlus_dif(beta[11]*S+beta[12])
                    dif_d13 = beta[10]*fl.SoftPlus_difdif(beta[11]*S+beta[12])*beta[11]
                    grad[12] = w*2*np.sum(dummy*d13)+(1-w)*2*np.sum(dif_dummy*dif_d13)  
                        
                    d14 = fl.SoftPlus(beta[14]*S+beta[15])
                    dif_d14 = fl.SoftPlus_dif(beta[14]*S+beta[15])*beta[14]
                    grad[13] = w*2*np.sum(dummy*d14)+(1-w)*2*np.sum(dif_dummy*dif_d14)
                        
                    d15 = beta[13]*fl.SoftPlus_dif(beta[14]*S+beta[15])*S
                    dif_d15 = beta[13]*fl.SoftPlus_difdif(beta[14]*S+beta[15])*S*beta[14]+beta[13]*fl.SoftPlus_dif(beta[14]*S+beta[15])
                    grad[14] = w*2*np.sum(dummy*d15)+(1-w)*2*np.sum(dif_dummy*dif_d15)
                        
                    d16 = beta[13]*fl.SoftPlus_dif(beta[14]*S+beta[15])
                    dif_d16 = beta[13]*fl.SoftPlus_difdif(beta[14]*S+beta[15])*beta[14]
                    grad[15] = w*2*np.sum(dummy*d16)+(1-w)*2*np.sum(dif_dummy*dif_d16)
                return(grad)


            res = minimize(criterion, x, method='BFGS', jac=Gradient,
               options={'maxiter':10000000000000,'disp': False})
            #print(res.success)
            Coef[:,i] = res.x
         
            ##Week 0
        Initialprice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]
        start = np.zeros(2)
        start[0] = fl.SoftPlus_reg(Coef[:,0],S0)
        start[1] = Coef[1,0]*fl.SoftPlus_dif(Coef[2,0]*S0+Coef[3,0])*Coef[2,0]
        S = np.ones(Npaths)
        Vpf = np.repeat(Initialprice,Npaths)
        a = np.repeat(start[1],Npaths)
        b = Vpf-a*S
        
        
        random.seed(1)
        e = np.ones((Npaths,Nhedge))
        e[:,0] = a*S+b-Initialprice
        for i in range(1,Nhedge):
            S = S+vol*np.sqrt(dt)*random.normal(0,1,Npaths)
            tau = T-dt*(i-1)
            dummy = np.asarray(fl.ZeroRateBachelierCall(S,tau,K,vol)) 
            #With or without reg. weights
            for k in range(0,Npaths):
                dummy[1,k] = Coef[1,i]*fl.SoftPlus_dif(Coef[2,i]*S[k]+Coef[3,i])*Coef[2,i]
                
            Vpf = a*S+b
                
            a = dummy[1]
            b = Vpf - a*S
                
            Vpf = b+a*S
            #Error for hedge port
            e[:,i] = Vpf - dummy[0]
            
            
        S = S+vol*np.sqrt(dt)*random.normal(0,1,Npaths)   
        CallPayoff = np.maximum(S-K,0)
        Vpf = a*S+b
        e[:,51] = Vpf - CallPayoff
        
        z = np.std(CallPayoff-Vpf)/Initialprice
        big_hedge_error[u,v] = z
        u = u+1
    v = v+1        
  




##### DUMMY WEIGHTS

poly = np.arange(4, 7, 1)
random.seed(1)
S0 = 1
T = 1
K = 1
vol = 0.2

TruePrice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]
max_power = 9
sn = 6000
sim_num = np.arange(1000, sn+1000, 1000)
big_hedge_error_1 = np.ones((max_power-2,8))
v = 0
for l in range(-1000,sn+1000,1000):  
    if l < -900:
        l=300
    elif l < 10:
        l = 600
    else:
        l = l
    print(l)    
    Npaths = 10000
    N = l
    Nhedge = 52
    dt = T/Nhedge
    u = 0
    print(v)
    for p in range(3, max_power+1, 1):
        #print(p)

        w = 1
        # Get coefeceints from polynmial regress

        Powers = np.arange(0, p, 1)
        M = len(Powers)
        Coef = np.zeros((M,Nhedge))
        
        Sim1 = random.normal(0,1,N)
        Sim2 = random.normal(0,1,N)
        
        CallDelta = np.zeros((N,))
        
        for i in range(0,52):
            S1 = S0+vol*np.sqrt(T)*Sim1
            S2 = S1+vol*np.sqrt(T-(i-1)*dt)*Sim2
            CallPayoff = np.maximum(S2-K,0)
            for j in range(0,len(CallDelta)):     
                if S2[j]>=K:
                    CallDelta[j] = 1
                else:
                    CallDelta[j] = 0
            tau = np.std(CallPayoff)/np.std(CallDelta)
            w = 1/(1+tau)
            X1 = np.ones(N)
            X2 = np.zeros(len(X1))
            for k in range(1,len(Powers)):
                X1 = np.column_stack((X1,S1**Powers[k]))
                X2 = np.column_stack((X2,Powers[k]*S1**(Powers[k]-1)))
            OLSCoef = np.linalg.solve((w*np.dot(X1.T,X1)+(1-w)*np.dot(X2.T,X2)),(
                                       w*np.dot(X1.T,CallPayoff))+(1-w)*np.dot(X2.T,CallDelta))
        
            #Error functions
            def error_wd(beta):
                b = beta
                return(w*np.dot((np.dot(X1,b)-CallPayoff).T,np.dot(X1,b)-CallPayoff)
                   +(1-w)*np.dot((np.dot(X2,b)-CallDelta).T,np.dot(X2,b)-CallDelta))
        
            def errordif_wd(beta):
                b = beta
                return(2*w*np.dot(np.dot(X1.T,X1),b)-2*w*np.dot(X1.T,CallPayoff)+
                   2*(1-w)*np.dot(np.dot(X2.T,X2),b)-2*(1-w)*np.dot(X2.T,CallDelta))
            res = minimize(error_wd, OLSCoef, method='BFGS', jac=errordif_wd,
                       options={'disp': False})
            Coef[:,i] = res.x
         
            ##Week 0
        Initialprice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]
        start = np.zeros(2)
        start[0] = np.dot(Coef[:,0],S0**Powers)
        start[1] = np.dot(Coef[1:M,0],Powers[1:M]*S0**(Powers[1:M]-1))
        S = np.ones(Npaths)
        Vpf = np.repeat(Initialprice,Npaths)
        a = np.repeat(start[1],Npaths)
        b = Vpf-a*S
        
        
        random.seed(1)
        e = np.ones((Npaths,Nhedge))
        e[:,0] = a*S+b-Initialprice
        for i in range(1,Nhedge):
            S = S+vol*np.sqrt(dt)*random.normal(0,1,Npaths)
            tau = T-dt*(i-1)
            dummy = np.asarray(fl.ZeroRateBachelierCall(S,tau,K,vol)) 
            #With or without reg. weights
            for k in range(0,Npaths):
                dummy[1,k] = np.dot(Coef[1:M,i],Powers[1:M]*S[k]**(Powers[1:M]-1))
                
            Vpf = a*S+b
                
            a = dummy[1]
            b = Vpf - a*S
                
            Vpf = b+a*S
            #Error for hedge port
            e[:,i] = Vpf - dummy[0]
            
            
        S = S+vol*np.sqrt(dt)*random.normal(0,1,Npaths)   
        CallPayoff = np.maximum(S-K,0)
        Vpf = a*S+b
        e[:,51] = Vpf - CallPayoff
        
        z = np.std(CallPayoff-Vpf)/Initialprice
        big_hedge_error_1[u,v] = z
        u = u+1
    v = v+1        
 




          
sim_num = np.array([300, 600, 1000, 2000, 3000, 4000, 5000, 6000])
plt.figure(0)
plt.plot(sim_num,big_hedge_error[0,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error[1,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error[2,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error[3,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error[4,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error[5,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error[6,:],'.-', color='red',markersize=2)
plt.plot(sim_num,big_hedge_error_1[0,:],'.-', color='black',markersize=2)
plt.plot(sim_num,big_hedge_error_1[1,:],'.-', color='black',markersize=2)
plt.plot(sim_num,big_hedge_error_1[2,:],'.-', color='black',markersize=2)
plt.plot(sim_num,big_hedge_error_1[3,:],'.-', color='black',markersize=2)
plt.plot(sim_num,big_hedge_error_1[4,:],'.-', color='black',markersize=2)
plt.plot(sim_num,big_hedge_error_1[5,:],'.-', color='black',markersize=2)
plt.plot(sim_num,big_hedge_error_1[6,:],'.-', color='black',markersize=2)


plt.text(7000,0.3,'3., 4.',horizontalalignment='right')
plt.text(7000,0.2,'5., 6. ',horizontalalignment='right')
plt.text(7000,0.17,'7., 8. ',horizontalalignment='right')
plt.text(7000,0.14,'9. ',horizontalalignment='right')



plt.ylim(0.09,0.8)
plt.hlines(y=0.1209, xmin=0, xmax=6000, colors='gray', linestyles='-', lw=2)
plt.xlabel('No. simulations in regression')
plt.ylabel('Standard deviation of relative hedge error')
plt.title('Call hedhing in Bacheliers model')
red_patch = mpatches.Patch(color='red', label='Half/Half (w = 0.5)')
black_patch = mpatches.Patch(color='black', label='Equal variances (w = w_sigma)')
gray_patch = mpatches.Patch(color='gray', label='True delta')
white_patch = mpatches.Patch(color='white', label='No. indicates deg. of. pol')

plt.legend(handles=[red_patch,black_patch,gray_patch, white_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Big_hedge.png')
    
