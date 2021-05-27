#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:32 2021

@author: MarcHansen
"""

#These creates the plots of Softplusfunction and the derivative of Softplus

import numpy as np
import matplotlib.pyplot as plt 
# evenly sampled time at 200ms intervals
t = np.arange(-5, 5., 0.2)

# red dashes, blue squares and green triangles
plt.figure(0)
plt.plot(t, np.log(1+np.exp(t)))
plt.xlabel('x', fontsize=15)
plt.ylim(-.5,5)
plt.ylabel('f(x)', fontsize=15)
plt.grid(b=False, which='major', color='#666666', linestyle='-',zorder=3)  
plt.savefig('plot of softplus.png')
plt.figure(1)
plt.plot(t, np.exp(t)/(1+np.exp(t)))
plt.xlabel('x', fontsize=15)
plt.ylim(-.5,2)
plt.ylabel('f(x)$', fontsize=15)
plt.grid(b=False, which='major', color='#666666', linestyle='-',zorder=3)  
plt.savefig('plot of softplus_dev.png')