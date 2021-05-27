#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:29:46 2021

@author: MarcHansen
"""

#This code creates a Softplus regression with Bachelier prices as labels and with differential regularization.
#One can change the order of the reg. by changing the number of dimensions in the initial point in line 73

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

c = 0.01
S = np.arange(.25, 2, c)
#With Bachelier price as independent var.
CallPrice = Bach_CallPrice = fl.ZeroRateBachelierCall(S,1,1,0.2)[0]
CallDelta = Bach_CallDelta = fl.ZeroRateBachelierCall(S,1,1,0.2)[1]


tau = np.std(CallPrice)/np.std(CallDelta)
w = 1/(1+tau)
w = 1

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


### KONKLUSSION
# Vi viser her at der muligvis er en ide i at bruge Softplus med vægte. Bruger kun c = 0.01, ellers bliver det for grimt. 


# c = 0.01
x = np.array([0,0,0,0])
z = random.uniform(0,1,3)
#x = np.array([-2.07621953e-03,  1.19018663e-01,  8.43636417e+00, -8.44961647e+00,z[0],z[1],z[2]])
#z = random.uniform(0,1,3)
#x = np.array([1.27991283e-03,  1.93530403e+00,  6.35843879e+00, -6.39125547e+00, -1.82388835e+00,  6.20036225e+00, -6.23646435e+00,z[0],z[1],z[2]])
#z = random.uniform(0,1,3)
#x = np.array([-0.01512412,  2.01124029,  6.12069584, -6.09774013, -1.90665693, 5.93993312, -5.91455241,  0.01211815,  0.88516106,  0.89241648,z[0],z[1],z[2]])
#x = np.array([6.06553113e-04,  1.65699423e+00,  5.63947385e+00, -5.61392957e+00, -1.62772161e+00,  5.27013424e+00, -5.23833385e+00, -1.51596242e-01, 1.86258703e+00,  9.20737572e-01,  4.13265364e-01,  1.35748575e+00, -5.20155632e-01,z[0],z[1],z[2]])

# c = 0.001
#x = np.array([0,0,0,0])
#z = random.uniform(0,1,3)
#x = np.array([-2.08147134e-03,  1.19010413e-01,  8.43672192e+00, -8.44985041e+00,z[0],z[1],z[2]])
#z = random.uniform(0,1,3)
#x = np.array([-3.8030038 ,  0.12239256,  8.27501655, -8.26686977,  0.31127256, -0.02463037, 12.22383747,z[0],z[1],z[2]])
#z = random.uniform(0,1,3)
#x = np.array([-1.68032953e+00,  1.22392552e-01,  8.27501671e+00, -8.26686993e+00, 5.39667735e-02, -1.42032851e-01,  1.19490447e+01,  1.04495882e+00, -2.60668752e-06,  5.29888756e-01,z[0],z[1],z[2]])
#z = random.uniform(0,1,3)
#x = np.array([-2.27390878e+00,  1.22392549e-01,  8.27501681e+00, -8.26687002e+00,        3.10093217e-02, -2.47136527e-01,  1.18580853e+01,  1.09291395e+00, -2.37866446e-06,  4.49211767e-01,  7.24514209e-01, -3.24965312e-06, 8.58087581e-01,z[0],z[1],z[2]])


# c = 0.0001 n = 4
#x = np.array([2.12586983e+00,  1.21737535e-01,  8.32636415e+00, -8.31601181e+00, -5.96434672e-01,  1.40590144e-02,  4.49229030e+00,  1.00633729e+00,       2.66546282e-04, -2.89410107e-01,0,0,0])
# c = 0.0001 n = 5
#x = np.array([2.12586983e+00,  1.21737535e-01,  8.32636415e+00, -8.31601181e+00, -5.96434671e-01,  1.40590144e-02,  4.49229030e+00,  1.00633729e+00, 2.66546312e-04, -2.89410107e-01,  1.00063386e-10,  4.46734635e-21, 1.21117921e-21,0,0,0])


res = minimize(criterion, x, method='BFGS', jac=Gradient, options={'maxiter':100000000,'disp': True})
print("Success Softplus:", res.success)

if len(x)==4:
    y = fl.SoftPlus_reg(res.x,S)
    delta = res.x[1]*fl.SoftPlus_dif(res.x[2]*S+res.x[3])*res.x[2]
elif len(x)==7:
    y = fl.SoftPlus_2reg(res.x,S)
    delta = res.x[1]*fl.SoftPlus_dif(res.x[2]*S+res.x[3])*res.x[2]+res.x[4]*fl.SoftPlus_dif(res.x[5]*S+res.x[6])*res.x[5]
elif len(x)==10:
    y = fl.SoftPlus_3reg(res.x,S)
    delta = res.x[1]*fl.SoftPlus_dif(res.x[2]*S+res.x[3])*res.x[2]+res.x[4]*fl.SoftPlus_dif(res.x[5]*S+res.x[6])*res.x[5]+res.x[7]*fl.SoftPlus_dif(res.x[8]*S+res.x[9])*res.x[8]
elif len(x)==13:
    y = fl.SoftPlus_4reg(res.x,S)
    delta = res.x[1]*fl.SoftPlus_dif(res.x[2]*S+res.x[3])*res.x[2]+res.x[4]*fl.SoftPlus_dif(res.x[5]*S+res.x[6])*res.x[5]+res.x[7]*fl.SoftPlus_dif(res.x[8]*S+res.x[9])*res.x[8]+res.x[10]*fl.SoftPlus_dif(res.x[11]*S+res.x[12])*res.x[11]
elif len(x)==16:
    y = fl.SoftPlus_5reg(res.x,S)
    delta = res.x[1]*fl.SoftPlus_dif(res.x[2]*S+res.x[3])*res.x[2]+res.x[4]*fl.SoftPlus_dif(res.x[5]*S+res.x[6])*res.x[5]+res.x[7]*fl.SoftPlus_dif(res.x[8]*S+res.x[9])*res.x[8]+res.x[10]*fl.SoftPlus_dif(res.x[11]*S+res.x[12])*res.x[11]+res.x[13]*fl.SoftPlus_dif(res.x[14]*S+res.x[15])*res.x[14]

plt.figure(0)
#Bachelier payoff
plt.plot(S,Bach_CallPrice,'.-',color = 'black')
plt.plot(S,y,'-',color = 'yellow')
#### REGRESSION ###
random.seed(3)
N = 10000
Sim1 = random.normal(0,1,len(S))
Sim2 = random.normal(0,1,len(S))
S = np.arange(.25, 2, c)

S2 = S+vol*math.sqrt(T)*Sim2
CallPayoff = CallPrice



p = len(x)
Powers = list(range(0,p))
Powers = np.array(Powers)
N = len(S)

X = np.ones((N,))


for i in range(1,p):
    X = np.column_stack((X,S**Powers[i]))

######### NUMERICAL OPTIMIZATION
x0 = x
x0 = np.zeros((len(x),))
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
print("Success Regression:", res1.success)
Coef_wd = res1.x

EstPrice_wd = np.zeros(len(S))
EstDelta_wd = np.zeros(len(S))
for i in range(0,len(S)):
    EstPrice_wd[i] = np.dot(Coef_wd,S[i]**Powers)
    EstDelta_wd[i] = np.dot(Coef_wd[1:len(Powers)]
    ,Powers[1:len(Powers)]*S[i]**(Powers[1:len(Powers)]-1))
    


#regression payoff
plt.plot(S,EstPrice_wd,'-',color = 'red')
plt.xlabel('Stock Price')
plt.ylabel('Call Value')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
red_patch = mpatches.Patch(color='red', label='3rd deg. pol. reg.')
plt.legend(handles=[yellow_patch,red_patch,black_patch])
plt.title('Option value w. different basis functions')
plt.grid(b=True, which='major', color='#666666', linestyle='-')

print("SoftPlus Price Error:", round(np.mean(abs(y-CallPrice)),8))

print("Regression Price Error:", round(np.mean(abs(EstPrice_wd-CallPrice)),8))
print("")


plt.figure(1)

#Bachelier payoff
plt.plot(S,Bach_CallDelta,'.-',color = 'black')
plt.plot(S,delta,'-',color = 'yellow')
plt.plot(S,EstDelta_wd,'-',color = 'red')
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')
plt.ylim(-0.2,2)
plt.title('Option delta w. different basis functions')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
red_patch = mpatches.Patch(color='red', label='3rd deg. poly. reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[yellow_patch,red_patch,black_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
print("Regression Delta Error:", round(np.mean(abs(EstDelta_wd-Bach_CallDelta)),8))