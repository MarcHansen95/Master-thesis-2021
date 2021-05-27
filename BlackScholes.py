#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:41:02 2021

@author: MarcHansen
"""
from scipy.stats import norm
import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#S = 1
T = .1
K = 100
r = 0.0
vol = 0.4

random.seed(3)
N = 10000
Sim1 = random.normal(0,1,N)
Sim2 = random.normal(0,1,N)

S0 = K*np.exp(-(vol**2)*T/2+np.sqrt(T)*Sim1)
S2 = S0+np.exp(-(vol**2)*T/2+np.sqrt(T)*Sim2)
CallPayoff = np.maximum(S2-K,0)

Powers = list(range(0,9))
Powers = np.array(Powers)

X = np.ones((N,))

for i in range(1,9):
    X = np.column_stack((X,S0**Powers[i]))

A = np.linalg.inv(np.dot(X.T,X))
B = np.dot(X.T,CallPayoff)
Coef = np.dot(A,B)

Srange = SimEstDelta = np.arange(0.1, 180, 1)

EstPrice = EstDelta = np.zeros(len(Srange))

for i in range(0,len(Srange)):
    EstPrice[i] = np.dot(Coef,Srange[i]**Powers)
    EstDelta[i] = np.dot(Coef[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))


true = fl.BlackScholesCall(Srange,T,K,r,vol)

plt.figure(0)
#simulated payoff
plt.plot(S0,CallPayoff , 'o', color='grey')
#regression payoff
plt.plot(Srange,EstPrice,'-',color = 'red')
#Bachelier payoff
plt.plot(Srange,true[0],'-',color = 'black')
plt.xlim(0,180)
plt.ylim(0,80)
plt.xlabel('Stock Price')
plt.ylabel('Call Price')
grey_patch = mpatches.Patch(color='grey', label = 'Simulated Prices')
red_patch = mpatches.Patch(color='red', label='8th degree pol. reg.')
black_patch = mpatches.Patch(color='black', label='Black Scholes')
plt.legend(handles=[grey_patch,red_patch,black_patch])
plt.title('Option Price')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_price.png')

plt.figure(1)
plt.plot(Srange,EstDelta,'-',color = 'red')
plt.plot(Srange,true[1],'-',color = 'black')
plt.xlim(0,180)
plt.ylim(-1,1.2)
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Option Delta')
red_patch = mpatches.Patch(color='red', label='8th degree pol. reg.')
black_patch = mpatches.Patch(color='black', label='Black Scholes')
plt.legend(handles=[red_patch,black_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_delta.png')





#############

Y = np.zeros((N,))
for i in range(1,9):
    Y = np.column_stack((Y,i*S0**(Powers[i]-1)))


CallDelta = np.zeros((N,))

for i in range(0,len(CallDelta)):
    if S2[i]>=K:
        CallDelta[i] = S2[i]/S0[i]
    else:
        CallDelta[i] = 0
        
plt.figure(3)
plt.plot(S0, CallDelta, 'o', color='green')
w = 1
A = np.linalg.inv(w*np.dot(X.T,X)+(1-w)*np.dot(Y.T,Y))
B = w*np.dot(X.T,CallPayoff)+(1-w)*np.dot(Y.T,CallDelta)
OLSCoef = np.dot(A,B)

EstPrice1 = np.zeros(len(Srange))
EstDelta1_2 = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    #EstPrice1[i] = np.dot(OLSCoef,Srange[i]**Powers1)
    EstDelta1_2[i] = np.dot(OLSCoef[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))


plt.plot(Srange,EstDelta1_2,'-',color = 'red')

#new weights
tau = np.std(CallPayoff)/np.std(CallDelta)
w = 1/(1+tau)


A = np.linalg.inv(w*np.dot(X.T,X)+(1-w)*np.dot(Y.T,Y))
B = w*np.dot(X.T,CallPayoff)+(1-w)*np.dot(Y.T,CallDelta)
OLSCoef = np.dot(A,B)

EstPrice1 = np.zeros(len(Srange))
EstDelta1 = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    #EstPrice1[i] = np.dot(OLSCoef,Srange[i]**Powers1)
    EstDelta1[i] = np.dot(OLSCoef[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))


plt.plot(Srange,EstDelta1,'-',color = 'blue')
    
w = 0.01
A = np.linalg.inv(w*np.dot(X.T,X)+(1-w)*np.dot(Y.T,Y))
B = w*np.dot(X.T,CallPayoff)+(1-w)*np.dot(Y.T,CallDelta)
OLSCoef = np.dot(A,B)

EstPrice1 = np.zeros(len(Srange))
EstDelta1_1 = np.zeros(len(Srange))
for i in range(0,len(Srange)):
    #EstPrice1[i] = np.dot(OLSCoef,Srange[i]**Powers1)
    EstDelta1_1[i] = np.dot(OLSCoef[1:len(Powers)]
    ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))


plt.plot(Srange,EstDelta1_1,'-',color = 'orange')
plt.plot(Srange,true[1],'-',color = 'black')

plt.xlim(0,180)
plt.ylim(-1,1.2)
plt.title('Option Delta with derivatives')
plt.xlabel('Stock Price')
plt.ylabel('Delta')
green_patch = mpatches.Patch(color='green', label = 'Simulated Deltas')
red_patch = mpatches.Patch(color='red', label='Price only-regression')
blue_patch = mpatches.Patch(color='blue', label='Half/half-regression')
orange_patch = mpatches.Patch(color='orange', label='Delta only-regression')
plt.legend(handles=[green_patch,red_patch,blue_patch,orange_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_delta_w_dev.png')

