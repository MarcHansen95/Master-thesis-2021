#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:02:44 2021

@author: MarcHansen
"""

#Here we define functions used in all other programs, hence it is important
# to place this in that same folder as the program one wish to run


from scipy.stats import norm
import math
import numpy as np

def ZeroRateBachelierCall(S,T,K,vol):
    d = (S-K)/(vol*math.sqrt(T))
    CallDelta = norm.cdf(d,0,1)
    CallPrice = (S-K)*norm.cdf(d,0,1) + vol*math.sqrt(T)*norm.pdf(d,0,1)
    return(CallPrice, CallDelta)

def BachelierCall_w_rate(S,T,K,r,vol):
    d = (S * np.exp(r*T) - K) / np.sqrt(vol**2/(2 * r) * (np.exp(2*r*T)-1) )  
    CallPrice = np.exp(-r * T) * (S * np.exp(r * T) - K) * norm.cdf(d) + \
        np.exp(-r * T) * np.sqrt(vol**2/(2*r) * (np.exp(2*r*T)-1) ) * norm.pdf(d)
    CallDelta = norm.cdf(d,0,1)
    return (CallPrice, CallDelta)
    
    
def BlackScholesCall(S,T,K,r,vol):
    d1 = 1/(vol*np.sqrt(T))*(np.log(S/K)+(r+1/2*vol**2)*T)
    d2 = d1 - vol*np.sqrt(T)
    CallDelta = norm.cdf(d1,0,1)
    CallPrice = S*norm.cdf(d1,0,1)-np.exp(-r*T)*K*norm.cdf(d2,0,1)
    return (CallPrice, CallDelta)

def BlackScholesCall_w_div(S,T,K,r, div,vol):
    q = div
    d1 = 1/(vol*np.sqrt(T))*(np.log(S/K)+(r-q+1/2*vol**2)*T)
    d2 = d1 - vol*np.sqrt(T)
    CallDelta = norm.cdf(d1,0,1)
    CallPrice = S*np.exp(-q*T)*norm.cdf(d1,0,1)-np.exp(-r*T)*K*norm.cdf(d2,0,1)
    return (CallPrice, CallDelta)

## N=1
def SoftPlus(x):
    a = np.log(1+np.exp(x))
    return(a)

def SoftPlus_dif(x):
    a = np.exp(x)/(1+np.exp(x))
    return(a)

def SoftPlus_difdif(x):
    a = np.exp(x)/((1+np.exp(x))**2)
    return(a)

def SoftPlus_reg(beta,X):
    y = beta[0]
    for i in range(1,1+(int((len(beta)-1)/3))):
        y = y+beta[3*i-2]*SoftPlus(beta[3*i-1]*X+beta[3*i])
    return(y)

## N=2
def SoftPlus_2reg(beta,X):
    y = beta[0]+beta[1]*SoftPlus(beta[2]*X+beta[3])+beta[4]*SoftPlus(beta[5]*X+beta[6])
    return(y)

## N=3
def SoftPlus_3reg(beta,X):
    y = beta[0]+beta[1]*SoftPlus(beta[2]*X+beta[3])+beta[4]*SoftPlus(beta[5]*X+beta[6])+beta[7]*SoftPlus(beta[8]*X+beta[9])
    return(y)

## N=4
def SoftPlus_4reg(beta,X):
    y = beta[0]+beta[1]*SoftPlus(beta[2]*X+beta[3])+beta[4]*SoftPlus(beta[5]*X+beta[6])+beta[7]*SoftPlus(beta[8]*X+beta[9])+beta[10]*SoftPlus(beta[11]*X+beta[12])
    return(y)
## N=5
def SoftPlus_5reg(beta,X):
    y = beta[0]+beta[1]*SoftPlus(beta[2]*X+beta[3])+beta[4]*SoftPlus(beta[5]*X+beta[6])+beta[7]*SoftPlus(beta[8]*X+beta[9])+beta[10]*SoftPlus(beta[11]*X+beta[12])+beta[13]*SoftPlus(beta[14]*X+beta[15])
    return(y)
def SoftPlus_6reg(beta,X):
    y = beta[0]+beta[1]*SoftPlus(beta[2]*X+beta[3])+beta[4]*SoftPlus(beta[5]*X+beta[6])+beta[7]*SoftPlus(beta[8]*X+beta[9])+beta[10]*SoftPlus(beta[11]*X+beta[12])+beta[13]*SoftPlus(beta[14]*X+beta[15])+beta[16]*SoftPlus(beta[17]*X+beta[18])
    return(y)

def rolf_SoftPlus_reg(beta,X):
    y = SoftPlus(beta[0]*X-beta[1])+beta[2]*SoftPlus(beta[3]*X-beta[4])
    return(y)