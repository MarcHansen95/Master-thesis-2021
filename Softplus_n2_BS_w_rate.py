#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:29:46 2021

@author: MarcHansen
"""
import Functions as fl
from numpy import random
import numpy as np
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize


random.seed(3)
T = 1
K = 1
vol = 0.2
r = 0.01
c = 0.0001
S = np.arange(.25, 2, c)
#With sim. CallPayoff as independent var. 
Sim2 = random.normal(0,1,len(S))
S2 = S*np.exp((r-vol**2*1/2)*T+vol*np.sqrt(T)*Sim2)


CallPrice = np.maximum(S2-K,0) 
BS_CallPrice = fl.BlackScholesCall(S,1,1,0.01,0.2)[0]
BS_CallDelta = fl.BlackScholesCall(S,1,1,0.01,0.2)[1]



def criterion(beta):
    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])+beta[4]*fl.SoftPlus(beta[5]*S+beta[6])
    return(np.sum((CallPrice-dummy)**2))

def Gradient(beta):
    dummy = beta[0]+beta[1]*fl.SoftPlus(beta[2]*S+beta[3])+beta[4]*fl.SoftPlus(beta[5]*S+beta[6])-CallPrice
    grad = np.zeros((len(beta),))
    d1 = 1
    grad[0] = 2*np.sum(dummy*d1)
    d2 = fl.SoftPlus(beta[2]*S+beta[3])
    grad[1] = 2*np.sum(dummy*d2)
    d3 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])*S
    grad[2] = 2*np.sum(dummy*d3)
    d4 = beta[1]*fl.SoftPlus_dif(beta[2]*S+beta[3])
    grad[3] = 2*np.sum(dummy*d4)
    d5 = fl.SoftPlus(beta[5]*S+beta[6])
    grad[4] = 2*np.sum(dummy*d5)
    d6 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])*S
    grad[5] = 2*np.sum(dummy*d6)
    d7 = beta[4]*fl.SoftPlus_dif(beta[5]*S+beta[6])
    grad[6] = 2*np.sum(dummy*d7)
    return(grad)

def rolf_criterion(beta):
    dummy = fl.SoftPlus(beta[0]*S - beta[1])+beta[2]*fl.SoftPlus(beta[3]*S - beta[4])
    return(np.sum((CallPrice-dummy)**2))

def rolf_Gradient(beta):
    dummy = fl.SoftPlus(beta[0]*S - beta[1])+beta[2]*fl.SoftPlus(beta[3]*S 
    - beta[4])-CallPrice
    grad = np.zeros((len(beta),))
    d1 = fl.SoftPlus_dif(beta[0]*S-beta[1])*S
    grad[0] = 2*np.sum(dummy*d1)
    d2 = -fl.SoftPlus_dif(beta[0]*S-beta[1])
    grad[1] = 2*np.sum(dummy*d2)
    d3 = fl.SoftPlus_dif(beta[3]*S-beta[4])
    grad[2] = 2*np.sum(dummy*d3)
    d4 = beta[2]*fl.SoftPlus_dif(beta[3]*S-beta[4])*S
    grad[3] = 2*np.sum(dummy*d4)
    d5 = -beta[2]*fl.SoftPlus_dif(beta[3]*S-beta[4])
    grad[4] = 2*np.sum(dummy*d5)
    return(grad)

x = np.array([0,0,0,0,0,0,0])

res = minimize(criterion, x, method='BFGS', jac=Gradient,
               options={'disp': True})
#Nelder-Mead giver dårligere resultat
"""res = minimize(criterion, x, method="Nelder-Mead",
               options={'maxiter':100000,'disp': True})
"""

y = fl.SoftPlus_2reg(res.x,S)
delta = res.x[1]*fl.SoftPlus_dif(res.x[2]*S+res.x[3])*res.x[2]+res.x[4]*fl.SoftPlus_dif(res.x[5]*S+res.x[6])*res.x[5]

"""print("ROLF")

x0 = np.array([0,0,0,0,0])

rolf_res = minimize(rolf_criterion, x0, method="Nelder-Mead",
               options={'maxiter':1000000,'disp': False})

print("hej")
rolf_y = fl.rolf_SoftPlus_reg(rolf_res.x,S)"""
print("hej")
plt.figure(0)
#plt.plot(S,rolf_y,'-',color = 'cyan')
#BSelier payoff
plt.plot(S,BS_CallPrice,'.-',color = 'black')
plt.plot(S,y,'-',color = 'yellow')

#### REGRESSION ###

CallPayoff = CallPrice

p=8
Powers = list(range(0,p))
Powers = np.array(Powers)

X = np.ones((len(S),))
N = len(S)
for i in range(1,p):
    X = np.column_stack((X,S**Powers[i]))

######### NUMERICAL OPTIMIZATION
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2,1.2,1.1,0.7])

Y = np.zeros((N,))

for i in range(1,p):
    Y = np.column_stack((Y,i*S**(Powers[i]-1)))

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

res1 = minimize(error_wd, x0, method='BFGS', jac=errordif_wd,
               options={'disp': False})

Coef_wd = res1.x

EstPrice_wd = np.zeros(len(S))
EstDelta_wd = np.zeros(len(S))
for i in range(0,len(S)):
    EstPrice_wd[i] = np.dot(Coef_wd,S[i]**Powers)
    EstDelta_wd[i] = np.dot(Coef_wd[1:len(Powers)]
    ,Powers[1:len(Powers)]*S[i]**(Powers[1:len(Powers)]-1))
    


#regression payoff
plt.plot(S,EstPrice_wd,'-',color = 'blue')
plt.xlabel('Stock Price')
plt.ylabel('Call Price')
blue_patch = mpatches.Patch(color='blue', label='7th deg. pol. reg.')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
black_patch = mpatches.Patch(color='black', label='Black-Scholes')

plt.legend(handles=[yellow_patch,black_patch,blue_patch])

plt.title('Option Price with N = 2')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_price_soft_n2_BS.png')
#print("Rolf Error:",round(np.mean(abs(rolf_y-CallPrice)),4))
print("My SoftPlus Error:", round(np.mean(abs(y-CallPrice)),5))
print("Regression Error:", round(np.mean(abs(EstPrice_wd-CallPrice)),5))


plt.figure(1)

#BSelier payoff
plt.plot(S,BS_CallDelta,'.-',color = 'black')
plt.plot(S,EstDelta_wd,'-',color = 'blue')
plt.plot(S,delta,'-',color = 'yellow')
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')
plt.ylim(-0.2,2)
plt.title('Option Delta w. N = 2')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
blue_patch = mpatches.Patch(color='blue', label='7th. deg. poly reg.')
black_patch = mpatches.Patch(color='black', label='Black-Scholes')
plt.legend(handles=[yellow_patch,black_patch,blue_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.savefig('Option_delta_soft_n2_BS.png')
print("My SoftPlus Delta Error:",round(np.mean(abs(delta-BS_CallDelta)),5))
print("Regression Delta Error:", round(np.mean(abs(EstDelta_wd-BS_CallDelta)),5))
