#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:29:46 2021

@author: MarcHansen
"""
#This code creates a Softplus regression with Bachelier prices as labels and with differential regularization.
#One can change the order of the reg. by changing N in line 81. The plots shows approx values and deltas
#We furthermore compare the softplus to a polynomial regression.



import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    y = fl.SoftPlus_reg(res.x,S)
    delta = 0
    for j in range(1,1+(int((len(res.x)-1)/3))):       
        delta = delta + res.x[3*j-2]*fl.SoftPlus_dif(res.x[3*j-1]*S+res.x[3*j])*res.x[3*j-1]
    #Bachelier payoff
    Price_error[h-1] = round(np.mean(abs(y-Bach_CallPrice)),8)
    Delta_error[h-1] = round(np.mean(abs(delta-Bach_CallDelta)),8)
    print("--- %s seconds ---" % round((time.time() - start_time),4))
    #print("SoftPlus Price Error:", round(np.mean(abs(y-CallPrice)),8))
    #print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
x_axis = S


x_values1 = np.arange(1, N+1, 1)
#### REGRESSION ###
print("")
print("Polynomial Regression")
print("")
CallPayoff = CallPrice
# Works for polynomials up to 16th degree

NN = 6
reg_Price_error = np.zeros((NN-2,))
reg_Delta_error = np.zeros((NN-2,))
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

    #print(res1.x)
    Coef_wd = res1.x
    
    EstPrice_wd = np.zeros(len(S))
    EstDelta_wd = np.zeros(len(S))
    for i in range(0,len(S)):
        EstPrice_wd[i] = np.dot(Coef_wd,S[i]**Powers)
        EstDelta_wd[i] = np.dot(Coef_wd[1:len(Powers)]
        ,Powers[1:len(Powers)]*S[i]**(Powers[1:len(Powers)]-1))
        
    reg_Price_error[J-3] = round(np.mean(abs(EstPrice_wd-Bach_CallPrice)),8)
    reg_Delta_error[J-3] = round(np.mean(abs(EstDelta_wd-Bach_CallDelta)),8)
    print("--- %s seconds ---" % round((time.time() - start_time),4))


plt.figure(0)
#Bachelier payoff
plt.plot(S,Bach_CallPrice,'.-',color = 'black')
plt.plot(S,y,'-',color = 'yellow')

##regression payoff
plt.plot(S,EstPrice_wd,'-',color = 'red')
plt.xlabel('Stock Price')
plt.ylabel('Call Value')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
red_patch = mpatches.Patch(color='red', label='6th deg. pol. reg.')
plt.legend(handles=[yellow_patch,red_patch,black_patch])
plt.title('Option value w. different basis functions')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

print("SoftPlus Price Error:", round(np.mean(abs(y-CallPrice)),8))

print("Regression Price Error:", round(np.mean(abs(EstPrice_wd-CallPrice)),8))
print("")


plt.figure(1)

plt.plot(S,Bach_CallDelta,'.-',color = 'black')
plt.plot(S,delta,'-',color = 'yellow')
plt.plot(S,EstDelta_wd,'-',color = 'red')
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')
plt.ylim(-0.2,2)
plt.title('Option delta w. different basis functions')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
red_patch = mpatches.Patch(color='red', label='6th deg. pol. reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[yellow_patch,red_patch,black_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
print("Regression Delta Error:", round(np.mean(abs(EstDelta_wd-Bach_CallDelta)),8))