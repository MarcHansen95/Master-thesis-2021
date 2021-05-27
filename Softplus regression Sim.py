#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:29:46 2021

@author: MarcHansen
"""

#This code creates a Softplus regression with simulated payoffs as labels and with differential regularization.
#One can change the order of the reg. by changing N in line 84. The plots shows approx value and deltas + runtime and error
#The plots are not really used in the thesis

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

c = 0.0001
S = np.arange(.25, 2, c)
#With sim. CallPayoff as independent var. 
Sim2 = random.normal(0,1,len(S))
S2 = S+vol*math.sqrt(T)*Sim2
CallPrice = np.maximum(S2-K,0) 

Bach_CallPrice = fl.ZeroRateBachelierCall(S,1,1,0.2)[0]
Bach_CallDelta = fl.ZeroRateBachelierCall(S,1,1,0.2)[1]

CallDelta = np.zeros((len(S),))

for i in range(0,len(CallDelta)):
    if S2[i]>=K:
        CallDelta[i] = 1
    else:
        CallDelta[i] = 0

tau = np.std(CallPrice)/np.std(CallDelta)
w = 1/(1+tau)


#### Konklussion #####
# Softplus konvergerer for alle N når c = 0.01
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

N = 2
Price_error = np.zeros((N,))
Delta_error = np.zeros((N,))
Price_var = np.zeros((N,))
Delta_var = np.zeros((N,))
runtime = np.zeros((N,))
for h in range(1,N+1):
    start_time = time.time()
    print(h)
    if h == 1:
        x = np.zeros(4)
    else:
        z = random.uniform(0,1,3)
        x = np.r_[res.x,z]
    #print(x)
    res = minimize(criterion, x, method='BFGS', jac=Gradient, options={'maxiter':100000000,'disp': True})
    #print(res.x)
    print("Success Softplus:", res.success)
    print(res.x)
    EstPrice = fl.SoftPlus_reg(res.x,S)
    delta = 0
    for j in range(1,1+(int((len(res.x)-1)/3))):       
        delta = delta + res.x[3*j-2]*fl.SoftPlus_dif(res.x[3*j-1]*S+res.x[3*j])*res.x[3*j-1]
    #Bachelier payoff
    Price_error[h-1] = round(np.mean(abs(EstPrice-Bach_CallPrice)),8)
    Delta_error[h-1] = round(np.mean(abs(delta-Bach_CallDelta)),8)
    Price_var[h-1] = round(np.var(abs(EstPrice-Bach_CallPrice)),8)
    Delta_var[h-1] = round(np.var(abs(delta-Bach_CallDelta)),8)
    runtime[h-1] = time.time() - start_time
    print("--- %s seconds ---" % round((time.time() - start_time),4))
    #print("SoftPlus Price Error:", round(np.mean(abs(y-CallPrice)),8))
    #print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
w = 1
Price_error_no_reg = np.zeros((N,))
Delta_error_no_reg = np.zeros((N,))

Price_var_no_reg = np.zeros((N,))
Delta_var_no_reg = np.zeros((N,))
runtime_no_reg = np.zeros((N,))
for h in range(1,N+1):
    start_time = time.time()
    print(h)
    if h == 1:
        x = np.zeros(4)
    else:
        z = random.uniform(0,1,3)
        x = np.r_[res.x,z]
    #print(x)
    res = minimize(criterion, x, method='BFGS', jac=Gradient, options={'maxiter':100000000,'disp': True})
    #print(res.x)
    print("Success Softplus:", res.success)
    EstPrice_no_reg = fl.SoftPlus_reg(res.x,S)
    delta_no_reg = 0
    for j in range(1,1+(int((len(res.x)-1)/3))):       
        delta_no_reg = delta_no_reg + res.x[3*j-2]*fl.SoftPlus_dif(res.x[3*j-1]*S+res.x[3*j])*res.x[3*j-1]
    #Bachelier payoff
    Price_error_no_reg[h-1] = round(np.mean(abs(EstPrice_no_reg-Bach_CallPrice)),8)
    Delta_error_no_reg[h-1] = round(np.mean(abs(delta_no_reg-Bach_CallDelta)),8)
    Price_var_no_reg[h-1] = round(np.var(abs(EstPrice_no_reg-Bach_CallPrice)),8)
    Delta_var_no_reg[h-1] = round(np.var(abs(delta_no_reg-Bach_CallDelta)),8)
    runtime_no_reg[h-1] = time.time() - start_time
    print("--- %s seconds ---" % round((time.time() - start_time),4))
    #print("SoftPlus Price Error:", round(np.mean(abs(y-CallPrice)),8))
    #print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
  
plt.figure(0)
plt.plot(S,EstPrice,'-',color = 'deepskyblue')
plt.plot(S,EstPrice_no_reg,'-',color = 'red')
#Bachelier payoff
plt.plot(S,Bach_CallPrice,'-',color = 'black')
plt.xlim(0.1,2)
plt.ylim(-0.1,1.2)
plt.xlabel('Stock Price')
plt.ylabel('Call Value')
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
red_patch = mpatches.Patch(color='red', label='No regularization')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[red_patch,deepskyblue_patch,black_patch])
plt.title('Option Price from polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.figure(1)
#regression payoff
plt.plot(S,delta,'-',color = 'deepskyblue')
plt.plot(S,delta_no_reg,'-',color = 'red')
#Bachelier payoff
plt.plot(S,Bach_CallDelta,'-',color = 'black')
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
red_patch = mpatches.Patch(color='red', label='No regularization')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[red_patch,deepskyblue_patch,black_patch])
plt.title('Option Price from polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')


plt.figure(2)
x = np.arange(1, N+1, 1)
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
plt.title('Option Price from polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.figure(3)
x = np.arange(1, N+1, 1)
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
plt.title('Option Price from polynomial regression')
plt.grid(b=True, which='major', color='#666666', linestyle='-')