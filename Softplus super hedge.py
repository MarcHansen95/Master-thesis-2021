#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:36:31 2021

@author: MarcHansen
"""

# A hedge experiment in Softplus regression. Change the max order in line 68. We only use this for max_order=3
import Functions as fl
from numpy import random
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize
random.seed(3)
#S0 = 1
T = 1
K = 1
vol = 0.2
c = 0.0001
S = np.arange(.25, 2, c)
TruePrice = fl.ZeroRateBachelierCall(S,T,K,vol)[0]

Nhedge = 52
dt = T/Nhedge
# Get coefeceints from polynmial regress
N = 10000


Npaths = len(S)
Sim1 = random.normal(0,1,Npaths)
Sim2 = random.normal(0,1,len(S))

def criterion(a):
    Dummy = a[0]
    dif_dummy = 0
    for j in range(1,1+(int((len(a)-1)/3))):
      Dummy = Dummy+a[3*j-2]*fl.SoftPlus(a[3*j-1]*S + a[3*j])
      dif_dummy = dif_dummy + a[3*j-2]*fl.SoftPlus_dif(a[3*j-1]*S+a[3*j])*a[3*j-1]
    return(w*np.sum((CallPrice-Dummy)**2)+(1-w)*np.sum((CallDelta-dif_dummy)**2))

def Gradient(a):
  Dummy = a[0]
  dif_dummy = 0
  for j in range(1,1+(int((len(a)-1)/3))):
      Dummy = Dummy+a[3*j-2]*fl.SoftPlus(a[3*j-1]*S + a[3*j])
      dif_dummy = dif_dummy+a[3*j-2]*fl.SoftPlus_dif(a[3*j-1]*S + a[3*j])*a[3*j-1]
  Dummy = Dummy-CallPrice
  dif_dummy = dif_dummy-CallDelta
  Grad = np.zeros(len(a))
  Grad[0] = w*2*np.sum(Dummy)
  for j in range(1,1+(int((len(a)-1)/3))):
    d1 = fl.SoftPlus(a[3*j-1]*S + a[3*j])
    dif_d1 = fl.SoftPlus_dif(a[3*j-1]*S+a[3*j])*a[3*j-1]
    Grad[3*j-2] = w*2*np.sum(Dummy*d1)+(1-w)*2*np.sum(dif_dummy*dif_d1)
    
    d2 = a[3*j-2]*fl.SoftPlus_dif(a[3*j-1]*S + a[3*j])*S
    dif_d2 = a[3*j-2]*fl.SoftPlus_difdif(a[3*j-1]*S+a[3*j])*S*a[3*j-1]+a[3*j-2]*fl.SoftPlus_dif(a[3*j-1]*S+a[3*j])
    Grad[3*j-1] = w*2*np.sum(Dummy*d2)+(1-w)*2*np.sum(dif_dummy*dif_d2)
    
    d3 = a[3*j-2]*fl.SoftPlus_dif(a[3*j-1]*S + a[3*j])
    dif_d3 = a[3*j-2]*fl.SoftPlus_difdif(a[3*j-1]*S+a[3*j])*a[3*j-1]
    Grad[3*j] = w*2*np.sum(Dummy*d3)+(1-w)*2*np.sum(dif_dummy*dif_d3)
  
  return(Grad)

max_order = 1
Hedge_error = np.zeros((max_order,))
for v in range(1,max_order+1):
    if v == 1:
        x = np.zeros(4)
    else:
        z = random.uniform(0,1,3)
        x = np.r_[res.x,z]
    Coef = np.zeros((len(x),Nhedge))
    S0 = 1    
    #CallDelta = np.zeros((len(S),))
    print("Hej11")
    u=0
    for i in range(0,52): 
        #print(u)
        S = np.arange(.25, 2, c)
        S2 = S+vol*np.sqrt(T-(i-1)*dt)*Sim2     
        CallPrice = np.maximum(S2-K,0)
        CallDelta = np.zeros((len(S),))
        for y in range(0,len(S2)):
            if S2[y]>=K:
                CallDelta[y] = 1
            else:
                CallDelta[y] = 0

        tau = np.std(CallPrice)/np.std(CallDelta)
        w = 1/(1+tau)
        w = 1
        #print(np.mean(CallDelta))      
        #CallPrice = Bach_CallPrice = fl.ZeroRateBachelierCall(S,T-(i-1)*dt,1,0.2)[0]
        #Bach_CallDelta = fl.ZeroRateBachelierCall(S,T-(i-1)*dt,1,0.2)[1]
        #print("din") 
        res = minimize(criterion, x, method='BFGS', jac=Gradient,
                   options={'maxiter':10000000000000,'disp': False})
        x = res.x
        print(res.success)
        #print(res.x)
        Coef[:,i] = res.x
        u=u+1
    
    ##Week 0
    Initialprice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]
    Initialdelta = fl.ZeroRateBachelierCall(S0,T,K,vol)[1]
    start = np.zeros(2)
    start[0] = fl.SoftPlus_reg(res.x,S0)
    start[1] = res.x[1]*fl.SoftPlus_dif(res.x[2]*S0+res.x[3])*res.x[2]
    Stock = np.ones(N)
    Vpf = np.repeat(Initialprice,N)
    a = np.repeat(start[1],N)
    
    b = Vpf-a*Stock
    
    random.seed(1)
    e = np.ones((N,Nhedge))
    e[:,0] = a*Stock+b-Initialprice
    S = Stock
    for i in range(1,Nhedge):
        S = S+vol*np.sqrt(dt)*random.normal(0,1,N)
        tau = T-dt*(i-1)
        dummy = np.asarray(fl.ZeroRateBachelierCall(S,tau,K,vol)) 

        for k in range(0,N):
            dummy[1,k] = 0
            for j in range(1,1+(int((len(res.x)-1)/3))):       
                dummy[1,k] = dummy[1,k] + Coef[3*j-2,i]*fl.SoftPlus_dif(Coef[3*j-1,i]*S[k]+Coef[3*j,i])*Coef[3*j-1,i]
        Vpf = a*S+b
        a = dummy[1]
        
        bach = fl.ZeroRateBachelierCall(S,tau,K,vol)[1]
        #print(bach-dummy[1])
        b = Vpf - a*S
    
    
        #Error for hedge port
        e[:,i] = Vpf - dummy[0]
        #print(np.mean(e,axis = 0))
    
    
    S = S+vol*np.sqrt(dt)*random.normal(0,1,N)   
    CallPayoff = np.maximum(S-K,0)
    Vpf = a*S+b
    e[:,51] = Vpf - CallPayoff
    Hedge_error[v-1] = np.std(CallPayoff-Vpf)/Initialprice
    

x_axis = np.arange(1, max_order+1, 1)
plt.figure(0)
plt.plot(x_axis,Hedge_error,'.-',color = 'Black')
#plt.ylim(0,0.0006)
plt.xlabel('Softplus order')
plt.ylabel('Relative hedge error')
plt.title('Hedge experiment')

        

