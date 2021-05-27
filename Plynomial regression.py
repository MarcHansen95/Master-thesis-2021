#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:29:46 2021

@author: MarcHansen
"""

#This creates polynomial  regression. One can change the maximal polynomial dregree in line 54.
# The value and delta graph of the maximal degree and results of all degrees + run time will be plottet
import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.lines as mlines
import time
random.seed(3)
T = 1
K = 1
vol = 0.2

N = 10000
Sim1 = random.normal(0,1,N)
Sim2 = random.normal(0,1,N)

S = K+vol*math.sqrt(T)*Sim1
S2 = S+vol*math.sqrt(T)*Sim2
CallPrice = CallPayoff = np.maximum(S2-K,0)

Srange = SimEstDelta = np.arange(0.1, 2.01, 0.01)
Bach_CallPrice = fl.ZeroRateBachelierCall(Srange,1,1,0.2)[0]
Bach_CallDelta = fl.ZeroRateBachelierCall(Srange,1,1,0.2)[1]

CallDelta = np.zeros((len(S),))

for i in range(0,len(CallDelta)):
    if S2[i]>=K:
        CallDelta[i] = 1
    else:
        CallDelta[i] = 0
tau = np.std(CallPrice)/np.std(CallDelta)
w = 1/(1+tau)



#### REGRESSION ###
print("")
print("Polynomial Regression")
print("")
CallPayoff = CallPrice
# Works for polynomials up to 16th degree
NN = 5
Price_error = np.zeros((NN-2,))
Delta_error = np.zeros((NN-2,))
runtime = np.zeros((NN-2,))
for J in range(3,NN+1):
    start_time = time.time()
    if J == 3:
        x = np.zeros(4)
    else:
        z = random.uniform(0,1,1)
        x = np.r_[res1.x,z]
    p = len(x)
    Powers = list(range(0,p))
    Powers = np.array(Powers)
    N = len(S)
    X = np.ones((N,))
    
    for i in range(1,p):
        X = np.column_stack((X,S**Powers[i]))
    
    ######### NUMERICAL OPTIMIZATION
    x0 = x
    #print(x0)
    
    Y = np.zeros((N,))
    
    for i in range(1,p):
        Y = np.column_stack((Y,i*S**(Powers[i]-1)))
    
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
    
    res1 = minimize(error_wd, x0, method='BFGS', jac=errordif_wd, options={'disp': False})
    print("Regression order:", J)
    print("Success Regression:", res1.success)
    Coef_wd = res1.x 
    EstPrice_wd = np.zeros(len(Srange))
    EstDelta_wd = np.zeros(len(Srange))
    for i in range(0,len(Srange)):
        EstPrice_wd[i] = np.dot(Coef_wd,Srange[i]**Powers)
        EstDelta_wd[i] = np.dot(Coef_wd[1:len(Powers)]
        ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))
        
    Price_error[J-3] = round(np.mean(abs(EstPrice_wd-Bach_CallPrice)),8)
    Delta_error[J-3] = round(np.mean(abs(EstDelta_wd-Bach_CallDelta)),8)
    runtime[J-3] = time.time() - start_time
    print("--- %s seconds ---" % round((time.time() - start_time),4))
print("")
print("No regularization")
print(w)
w = 1
Price_error_no_reg = np.zeros((NN-2,))
Delta_error_no_reg = np.zeros((NN-2,))
runtime_no_reg = np.zeros((NN-2,))
for J in range(3,NN+1):
    start_time = time.time()
    if J == 3:
        x = np.zeros(4)
    else:
        z = random.uniform(0,1,1)
        x = np.r_[res1.x,z]
    p = len(x)
    Powers = list(range(0,p))
    Powers = np.array(Powers)
    N = len(S)
    X = np.ones((N,))  
    for i in range(1,p):
        X = np.column_stack((X,S**Powers[i]))   
    ######### NUMERICAL OPTIMIZATION
    x0 = x
    Y = np.zeros((N,))
    for i in range(1,p):
        Y = np.column_stack((Y,i*S**(Powers[i]-1)))    
    def error_wd(beta):
        b = beta
        return(w*np.dot((np.dot(X,b)-CallPayoff).T,np.dot(X,b)-CallPayoff)
               +(1-w)*np.dot((np.dot(Y,b)-CallDelta).T,np.dot(Y,b)-CallDelta))
    
    def errordif_wd(beta):
        b = beta
        return(2*w*np.dot(np.dot(X.T,X),b)-2*w*np.dot(X.T,CallPayoff)+
               2*(1-w)*np.dot(np.dot(Y.T,Y),b)-2*(1-w)*np.dot(Y.T,CallDelta))
    res1 = minimize(error_wd, x0, method='BFGS', jac=errordif_wd, options={'disp': False})
    print("Regression order:", J)
    print("Success Regression:", res1.success)
    Coef_wd = res1.x 
    EstPrice = np.zeros(len(Srange))
    EstDelta = np.zeros(len(Srange))
    for i in range(0,len(Srange)):
        EstPrice[i] = np.dot(Coef_wd,Srange[i]**Powers)
        EstDelta[i] = np.dot(Coef_wd[1:len(Powers)]
        ,Powers[1:len(Powers)]*Srange[i]**(Powers[1:len(Powers)]-1))      
    Price_error_no_reg[J-3] = round(np.mean(abs(EstPrice-Bach_CallPrice)),8)
    Delta_error_no_reg[J-3] = round(np.mean(abs(EstDelta-Bach_CallDelta)),8)
    runtime_no_reg[J-3] = time.time() - start_time
    print("--- %s seconds ---" % round((time.time() - start_time),4))


plt.figure(0)
plt.plot(S,CallPayoff,',', color='grey')
plt.plot(Srange,Bach_CallPrice,'-',color = 'black')
plt.plot(Srange,EstPrice_wd,'-',color = 'deepskyblue')
plt.plot(Srange,EstPrice,'-',color = 'red')
#Bachelier payoff

plt.xlim(0.4,1.75)
plt.ylim(-0.1,1.2)
plt.xlabel('Stock Price')
plt.ylabel('Call Value')
grey_patch = mpatches.Patch(color='grey', label='Simulated payoffs')
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
red_patch = mpatches.Patch(color='red', label='No regularization')
black_patch = mpatches.Patch(color='black', label='Bachelier')

plt.legend(handles=[grey_patch,red_patch,deepskyblue_patch,black_patch])
plt.title('Option value with polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.figure(1)
#regression payoff
plt.plot(S, CallDelta, ',', color='green')
plt.plot(Srange,EstDelta_wd,'-',color = 'deepskyblue')
plt.plot(Srange,EstDelta,'-',color = 'red')
#Bachelier payoff
plt.plot(Srange,Bach_CallDelta,'-',color = 'black')
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')
plt.xlim(0.25,2)
plt.ylim(-0.1,1.7)
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
red_patch = mpatches.Patch(color='red', label='No regularization')
black_patch = mpatches.Patch(color='black', label='Bachelier')
green_patch = mpatches.Patch(color='green', label='Simulated deltas')
plt.legend(handles=[green_patch,red_patch,deepskyblue_patch,black_patch])
plt.title('Option delta with polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')



plt.figure(2)
x = np.arange(3, NN+1, 1)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x,Price_error,color = 'deepskyblue')
ax1.plot(x,Price_error_no_reg,color = 'red')
ax2.plot(x, runtime,':',color = 'deepskyblue')
ax2.plot(x, runtime_no_reg,':',color = 'red')
ax1.set_xlabel('Polynomial order')
ax1.set_ylabel('Value error', color='black')
ax2.set_ylabel('Runtime', color='black')
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
red_patch = mpatches.Patch(color='red', label='No regularization')

dashed_line = mlines.Line2D([], [], color='black', linestyle =':', markersize=15, label='Runtime')


plt.legend(handles=[red_patch,deepskyblue_patch,dashed_line])
plt.title('Option value with polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.figure(3)
x = np.arange(3, NN+1, 1)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x,Delta_error,color = 'deepskyblue')
ax1.plot(x,Delta_error_no_reg,color = 'red')
ax2.plot(x, runtime,':',color = 'deepskyblue')
ax2.plot(x, runtime_no_reg,':',color = 'red')
ax1.set_xlabel('Polynomial order')
ax1.set_ylabel('Delta error', color='black')
ax2.set_ylabel('Runtime', color='black')
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
red_patch = mpatches.Patch(color='red', label='No regularization')
plt.legend(handles=[red_patch,deepskyblue_patch,dashed_line])
plt.title('Option delta with polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')