#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:02:32 2021

@author: MarcHansen
"""

#This code can be used to run the model with an interest rate. This is not used directly in the thesis

import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize

S = 1
T = 1
K = 1
vol = 0.2
r = 0.01

random.seed(3)
N = 10000
Sim1 = random.normal(0,1,N)
Sim2 = random.normal(0,1,N)

S1 = S+r*S*T+vol*math.sqrt(T)*Sim1
S2 = S1+r*S1*T+vol*math.sqrt(T)*Sim2
CallPayoff = np.maximum(S2-K,0)

Powers = list(range(0,9))
Powers = np.array(Powers)

X = np.ones((N,))

for i in range(1,9):
    X = np.column_stack((X,S1**Powers[i]))


Srange = SimEstDelta = np.arange(0.1, 2.01, 0.01)

true = fl.BachelierCall_w_rate(Srange,T,K,r,vol)

######### NUMERICAL OPTIMIZATION
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2, 3.3,4,3,2])

def error(beta):
    b = beta
    return(np.dot(b.T,np.dot(X.T,np.dot(X,b)))-
           2*np.dot(b.T,np.dot(X.T,CallPayoff))+np.dot(CallPayoff.T,CallPayoff))

def errordif(beta):
    b = beta
    return(2*np.dot(X.T,np.dot(X,b))-2*np.dot(X.T,CallPayoff))
    
def errorhess_wd(beta):
    b = beta
    return(2*np.dot(X.T,X))



res = minimize(error, x0, method='BFGS', jac=errordif,
               options={'disp': False})


Coef = res.x

EstPrice = np.zeros(len(Srange))
EstDelta = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    EstPrice[i] = np.dot(Coef,Srange[i]**Powers)
    EstDelta[i] = np.dot(Coef[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))



plt.figure(1)
#simulated payoff
plt.plot(S1,CallPayoff , 'o', color='grey')
#regression payoff
plt.plot(Srange,EstPrice,'-',color = 'red')
#Bachelier payoff
plt.plot(Srange,true[0],'-',color = 'black')
plt.xlim(0.8,1.5)
plt.ylim(0,1.2)
plt.xlabel('Stock Price')
plt.ylabel('Call Price')
grey_patch = mpatches.Patch(color='grey', label = 'Simulated Prices')
red_patch = mpatches.Patch(color='red', label='8th degree pol. reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[grey_patch,red_patch,black_patch])
plt.title('Option Price with numerical optimization')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_price.png')

plt.figure(2)
plt.plot(Srange,EstDelta,'-',color = 'red')
plt.plot(Srange,true[1],'-',color = 'black')
plt.xlim(0.6,1.6)
plt.ylim(0,1.2)
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Option Delta with numerical optimization')
red_patch = mpatches.Patch(color='red', label='8th degree pol. reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[red_patch,black_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_delta.png')

# Wtih derivatives and weights

Y = np.zeros((N,))

for i in range(1,9):
    Y = np.column_stack((Y,i*S1**(Powers[i]-1)))

CallDelta = np.zeros((N,))

for i in range(0,len(CallDelta)):
    if S2[i]>=K:
        CallDelta[i] = 1
    else:
        CallDelta[i] = 0

tau = np.std(CallPayoff)/np.std(CallDelta)
w = 1/(1+tau)
def error_wd(beta):
    b = beta
    return(w*np.dot((np.dot(X,b)-CallPayoff).T,np.dot(X,b)-CallPayoff)
           +(1-w)*np.dot((np.dot(Y,b)-CallDelta).T,np.dot(Y,b)-CallDelta))

def errordif_wd(beta):
    b = beta
    return(2*w*np.dot(np.dot(X.T,X),b)-2*w*np.dot(X.T,CallPayoff)+
           2*(1-w)*np.dot(np.dot(Y.T,Y),b)-2*(1-w)*np.dot(Y.T,CallDelta))
    
def errorhess_wd(beta):
    b = beta
    return(2*w*np.dot(X.T,X)+2*(1-w)*np.dot(Y.T,Y))

res1 = minimize(error_wd, x0, method='BFGS', jac=errordif_wd,
               options={'disp': True})

Coef_wd = res1.x

EstPrice_wd = np.zeros(len(Srange))
EstDelta_wd = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    EstPrice_wd[i] = np.dot(Coef_wd,Srange[i]**Powers)
    EstDelta_wd[i] = np.dot(Coef_wd[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))
    
    
w = 1
def error_wd_1(beta):
    b = beta
    return(w*np.dot((np.dot(X,b)-CallPayoff).T,np.dot(X,b)-CallPayoff)
           +(1-w)*np.dot((np.dot(Y,b)-CallDelta).T,np.dot(Y,b)-CallDelta))

def errordif_wd_1(beta):
    b = beta
    return(2*w*np.dot(np.dot(X.T,X),b)-2*w*np.dot(X.T,CallPayoff)+
           2*(1-w)*np.dot(np.dot(Y.T,Y),b)-2*(1-w)*np.dot(Y.T,CallDelta))
    
def errorhess_wd_1(beta):
    b = beta
    return(2*w*np.dot(X.T,X)+2*(1-w)*np.dot(Y.T,Y))

res1 = minimize(error_wd_1, x0, method='BFGS', jac=errordif_wd_1,
               options={'disp': True})

Coef_wd_1 = res1.x    
    

EstPrice_wd_1 = np.zeros(len(Srange))
EstDelta_wd_1 = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    EstPrice_wd_1[i] = np.dot(Coef_wd_1,Srange[i]**Powers)
    EstDelta_wd_1[i] = np.dot(Coef_wd_1[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))   
    
w = 0.01
def error_wd_0(beta):
    b = beta
    return(w*np.dot((np.dot(X,b)-CallPayoff).T,np.dot(X,b)-CallPayoff)
           +(1-w)*np.dot((np.dot(Y,b)-CallDelta).T,np.dot(Y,b)-CallDelta))

def errordif_wd_0(beta):
    b = beta
    return(2*w*np.dot(np.dot(X.T,X),b)-2*w*np.dot(X.T,CallPayoff)+
           2*(1-w)*np.dot(np.dot(Y.T,Y),b)-2*(1-w)*np.dot(Y.T,CallDelta))
    
def errorhess_wd_0(beta):
    b = beta
    return(2*w*np.dot(X.T,X)+2*(1-w)*np.dot(Y.T,Y))

res1 = minimize(error_wd_0, x0, method='BFGS', jac=errordif_wd_0,
               options={'disp': True})

Coef_wd_0 = res1.x    
    

EstPrice_wd_0 = np.zeros(len(Srange))
EstDelta_wd_0 = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    EstPrice_wd_0[i] = np.dot(Coef_wd_0,Srange[i]**Powers)
    EstDelta_wd_0[i] = np.dot(Coef_wd_0[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))   


plt.figure(3)
plt.plot(Srange,EstDelta_wd_0,'-',color = 'orange')
plt.plot(S1, CallDelta, 'o', color='green')
plt.plot(Srange,EstDelta_wd,'-',color = 'blue')
plt.plot(Srange,EstDelta_wd_1,'-',color = 'red')
plt.plot(Srange,true[1],'-',color = 'black')
plt.xlim(0.2,2)
plt.ylim(0,2)
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Option Delta with numerical optimization and d.w')
green_patch = mpatches.Patch(color='green', label = 'Simulated Deltas')
red_patch = mpatches.Patch(color='red', label='Price only-regression')
blue_patch = mpatches.Patch(color='blue', label='Half/half-regression')
orange_patch = mpatches.Patch(color='orange', label='Delta only-regression')
plt.legend(handles=[green_patch,red_patch,blue_patch,orange_patch,black_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_delta.png')





"""
plt.figure(3)
#simulated payoff
plt.plot(S1,CallPayoff , 'o', color='grey')
#regression payoff
plt.plot(Srange,EstPrice_wd,'-',color = 'red')
#Bachelier payoff
plt.plot(Srange,true[0],'-',color = 'black')
plt.xlim(0.8,1.5)
plt.ylim(0,1.2)
plt.xlabel('Stock Price')
plt.ylabel('Call Price')
grey_patch = mpatches.Patch(color='grey', label = 'Simulated Prices')
red_patch = mpatches.Patch(color='red', label='8th degree pol. reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[grey_patch,red_patch,black_patch])
plt.title('Option Price with numerical optimization and d.w')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_price.png')
"""