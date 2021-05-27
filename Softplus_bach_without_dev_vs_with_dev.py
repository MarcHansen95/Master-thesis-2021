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
from Softplus_super_Bachprice import Price_error, Delta_error


#Use this to get Softplus reg. with Bachelier prices with. and without. diff. regularization. Change the order in line 70


random.seed(3)
T = 1
K = 1
vol = 0.2

c = 0.001
S = np.arange(.25, 2.0, c)
#With Bachelier price as independent var.

CallPrice = Bach_CallPrice = fl.ZeroRateBachelierCall(S,1,1,0.2)[0]
CallDelta = Bach_CallDelta = fl.ZeroRateBachelierCall(S,1,1,0.2)[1]

tau = np.std(CallPrice)/np.std(CallDelta)
w = 1/(1+tau)


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


#### KONKLUSSION #####
#Brug denne kode til at vise der er en ide med Softplus + Bach. Vi har ikke tilføjet vægte, så denne model, skal sammenlignes med den med vægte
N = 16
Price_error_w_dev = np.zeros((N,))
Delta_error_w_dev = np.zeros((N,))
for N in range(1,N+1):
    if N == 1:
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
    plt.figure(0)
    #Bachelier payoff
    Price_error_w_dev[N-1] = round(np.mean(abs(y-CallPrice)),8)
    Delta_error_w_dev[N-1] = round(np.mean(abs(delta-Bach_CallDelta)),8)
    #print("SoftPlus Price Error:", round(np.mean(abs(y-CallPrice)),8))
    #print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
x_axis_1 = np.arange(1, N+1, 1)
x_values1 = np.arange(1, N+1, 1)
y_values1_w_dev = Price_error_w_dev

fig=plt.figure(0)
ax=fig.add_subplot(111)
ax2=ax.twiny()
    
ax.plot(x_axis_1, Price_error,'.-', color="black")
ax.set_xlabel("Without derivatives", color="black")
ax.set_ylabel("Error", color="black")
ax.tick_params(axis='x', colors="black")
ax.tick_params(axis='y', colors="black")

ax2.plot(x_values1, y_values1_w_dev,'.-', color="deepskyblue")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('With derivatives', color="deepskyblue") 
ax2.set_ylabel('Error', color="deepskyblue")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="deepskyblue")
ax2.tick_params(axis='y', colors="deepskyblue")

plt.title('Option Price Error')
plt.ylim(0,0.0007)
ax.grid(b=True, which='major', color='#666666', linestyle='-')
blue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
black_patch = mpatches.Patch(color='black', label='No regularization')

plt.legend(handles=[blue_patch,black_patch])
plt.show()


fig=plt.figure(1)

x_values1 = x_axis_1
y_values1 = Delta_error_w_dev

x_values2 = x_axis_1

ax=fig.add_subplot(111)
ax2=ax.twiny()
    
ax.plot(x_values1, Delta_error,'.-', color="black")
ax.set_xlabel("Softplus order", color="black")
ax.set_ylabel("Error", color="black")
ax.tick_params(axis='x', colors="black")
ax.tick_params(axis='y', colors="black")

ax2.plot(x_values2, y_values1,'.-', color="deepskyblue")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('Softplus order', color="deepskyblue") 
ax2.set_ylabel('Error', color="deepskyblue")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="deepskyblue")
ax2.tick_params(axis='y', colors="deepskyblue")

plt.title('Option Delta Error')
plt.ylim(0,0.006)
ax.grid(b=True, which='major', color='#666666', linestyle='-')
blue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
black_patch = mpatches.Patch(color='black', label='No regularization')
plt.legend(handles=[blue_patch,black_patch])
plt.show()


"""plt.figure(0)
plt.plot(x_axis_1,Price_error,'.-',color = 'black')
plt.figure(1)
plt.plot(x_axis_1,Delta_error,'.-',color = 'black')"""


"""
plt.plot(S,Bach_CallPrice,'.-',color = 'black')
plt.plot(S,y,'-',color = 'yellow')"""
"""#### REGRESSION ###
print("HER")
CallPayoff = CallPrice
NN = 16
reg_Price_error = np.zeros((NN-2,))
reg_Delta_error = np.zeros((NN-2,))
for J in range(3,NN+1):
    if J == 3:
        x = np.zeros(4)
    else:
        z = random.uniform(0,1,1)
        x = np.r_[res1.x,z]
    p = len(x)
    print(p)
    Powers = list(range(0,p))
    Powers = np.array(Powers)
    N = len(S)
    X = np.ones((N,))
    
    for i in range(1,p):
        X = np.column_stack((X,S**Powers[i]))
    
    ######### NUMERICAL OPTIMIZATION
    x0 = x
    print(x0)
    
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
    #print("Success Regression:", res1.success)
    #print(res1.x)
    Coef_wd = res1.x
    
    EstPrice_wd = np.zeros(len(S))
    EstDelta_wd = np.zeros(len(S))
    for i in range(0,len(S)):
        EstPrice_wd[i] = np.dot(Coef_wd,S[i]**Powers)
        EstDelta_wd[i] = np.dot(Coef_wd[1:len(Powers)]
        ,Powers[1:len(Powers)]*S[i]**(Powers[1:len(Powers)]-1))
        
    reg_Price_error[J-3] = round(np.mean(abs(EstPrice_wd-CallPrice)),8)
    reg_Delta_error[J-3] = round(np.mean(abs(EstDelta_wd-Bach_CallDelta)),8)
x_axis = np.arange(3, NN+1, 1)

x_values2 = x_axis
y_values2 = reg_Price_error

fig=plt.figure(0)
ax=fig.add_subplot(111)
ax2=ax.twiny()
    
ax.plot(x_axis_1, y_values1,'.-', color="black")
ax.set_xlabel("SoftPlus order", color="black")
ax.set_ylabel("Error", color="black")
ax.tick_params(axis='x', colors="black")
ax.tick_params(axis='y', colors="black")

ax2.plot(x_values2, y_values2,'.-', color="lawngreen")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('Polynomial order', color="red") 
ax2.set_ylabel('Error', color="red")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="red")
ax2.tick_params(axis='y', colors="red")

plt.title('Option Price Error')
plt.ylim(0,0.004)
ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()

fig=plt.figure(1)

x_values1 = x_axis_1
y_values1 = Delta_error

x_values2 = x_axis
y_values2 = reg_Delta_error
ax=fig.add_subplot(111)
ax2=ax.twiny()
    
ax.plot(x_values1, y_values1,'.-', color="black")
ax.set_xlabel("SoftPlus order", color="black")
ax.set_ylabel("Error", color="black")
ax.tick_params(axis='x', colors="black")
ax.tick_params(axis='y', colors="black")

ax2.plot(x_values2, y_values2,'.-', color="red")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('Polynomial order', color="red") 
ax2.set_ylabel('Error', color="red")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="red")
ax2.tick_params(axis='y', colors="red")

plt.title('Option Delta Error')
plt.ylim(0,0.04)
ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()





plt.figure(0)
plt.plot(x_axis,reg_Price_error,'.-',color = 'red')
#plt.ylim(0,0.0006)
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')
"""
"""plt.figure(2)
plt.plot(x_axis,reg_Delta_error,'.-',color = 'red')
#plt.ylim(0,0.006)
#regression payoff
plt.xlabel('Stock Price')
plt.ylabel('Call Delta')

plt.plot(S,EstPrice_wd,'-',color = 'red')
plt.xlabel('Stock Price')
plt.ylabel('Call Price')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
red_patch = mpatches.Patch(color='red', label='6th deg. pol. reg.')
plt.legend(handles=[yellow_patch,black_patch,red_patch])
plt.title('Option Price from polynomial regression')
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
plt.title('Option Delta w. different basis functions')
yellow_patch = mpatches.Patch(color='yellow', label='SoftPlus reg.')
red_patch = mpatches.Patch(color='red', label='6th. deg. poly reg.')
black_patch = mpatches.Patch(color='black', label='Bachelier')
plt.legend(handles=[yellow_patch,black_patch,red_patch])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
print("SoftPlus Delta Error:",round(np.mean(abs(delta-Bach_CallDelta)),8))
print("Regression Delta Error:", round(np.mean(abs(EstDelta_wd-Bach_CallDelta)),8))"""