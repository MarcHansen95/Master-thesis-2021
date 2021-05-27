#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:36:31 2021

@author: MarcHansen
"""
#this i a hedge experiment in a polynomial regression.
#One need to cahnge the weights in line 62 to get the different results. 



from scipy.stats import norm
import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize

random.seed(3)
S0 = 1
T = 1
K = 1

vol = 0.2

TruePrice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]

Npaths = 10000
Nhedge = 52
dt = T/Nhedge

# Get coefeceints from polynmial regress
N = 10000
Powers = np.arange(0, 9, 1)
#15 er max, da begunder den at chrashe pga høje polynomier 
#Starting point in approx of coef is a bit diff to make sure, 
#that the poly doenst divergence
M = len(Powers)
Coef = np.zeros((M,Nhedge))

Sim1 = random.normal(0,1,N)
Sim2 = random.normal(0,1,N)

CallDelta = np.zeros((N,))

dum_coef = np.arange(3,12,1)

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
S0 = 1
Initialprice = fl.ZeroRateBachelierCall(S0,T,K,vol)[0]
Initialdelta = fl.ZeroRateBachelierCall(S0,T,K,vol)[1]
start = np.zeros(2)
start[0] = np.dot(Coef[:,0],S0**Powers)
start[1] = np.dot(Coef[1:M,0],Powers[1:M]*S0**(Powers[1:M]-1))
S = np.ones(Npaths)
Vpf = np.repeat(Initialprice,Npaths)
a = np.repeat(start[1],Npaths)
#without regression
#a = np.repeat(Initialdelta,Npaths)
b = Vpf-a*S

random.seed(1)
e = np.ones((Npaths,Nhedge))
e[:,0] = a*S+b-Initialprice
S = S0
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


    #Error for hedge port
    e[:,i] = Vpf - dummy[0]

    """
    print(np.mean(dummy,axis=1)[0])
    print(np.mean(Vpf))
    print(np.mean(e[:,i]))
    """
    """
    plt.figure(i)
    plt.plot(S,Vpf , 'o', color='grey')
    plt.plot(S,dummy[0,:] , 'o', color='red')
"""


S = S+vol*np.sqrt(dt)*random.normal(0,1,Npaths)   
CallPayoff = np.maximum(S-K,0)
Vpf = a*S+b
e[:,51] = Vpf - CallPayoff
"""
print("Sidste uge")
print(np.mean(CallPayoff))
print(np.mean(Vpf))
print(np.mean(e[:,51]))
"""

plt.figure(1)
#simulated payoff
plt.plot(S,Vpf , 'o', color='grey', markersize=2)
#regression payoff
plt.plot(S,CallPayoff,'o',color = 'red',markersize=2) 
plt.xlabel('Stock price')
plt.ylabel('Value')
plt.title('Value of hedge and option at expire')
red_patch = mpatches.Patch(color='red', label='Call option payoff')
grey_patch = mpatches.Patch(color='grey', label='Hedge portfolio')
plt.legend(handles=[grey_patch,red_patch])
plt.savefig('Value_hedge.png')
error = np.mean(e,axis=0)
xx = np.arange(0,52,1)
plt.figure(2)
plt.plot(xx,error,'o')
plt.ylim(-0.02,0.1)
plt.xlabel('Week no.')
plt.ylabel('Running hedge error')
plt.title('Running hedge error with 10,000 paths')
plt.savefig('error_hedge.png')

z = np.std(CallPayoff-Vpf)/Initialprice
print(z)
        






a = 0.00021886892318254394
b = 0.00020013974330207704


"""S = np.zeros((Npaths,T))
S[:,0] = 1

for j in range(0,Npaths):
    for i in range (1,T):
        S1 = S[j-1,i-1]+vol*math.sqrt(1)*random.normal(0,1,1)
        S[j,i] = S1+vol*math.sqrt(1)*random.normal(0,1,1)"""

