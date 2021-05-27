#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 08:24:53 2021

@author: MarcHansen
"""

#This creates Figure 5.15 (b)


import Functions as fl
from numpy import random
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize


x = np.array([0.12413542, 0.13153904, 0.41923348, 0.53891355])
y = np.array([0.12462922, 0.12445072, 0.12572268, 1])
x_axis = np.arange(1, len(x)+1, 1)
plt.figure(0)
plt.plot(x_axis,x,'.-',color = 'Black')
plt.plot(x_axis,y,'.-',color = 'red')
plt.ylim(0.1,0.55)
plt.xlabel('Softplus order')
plt.ylabel('Relative hedge error')
plt.title('Hedge experiment')