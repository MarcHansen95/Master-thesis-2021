#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:44:47 2021

@author: MarcHansen
"""
#This creates Figure 6.3. The values are extracted from other programs.




import Functions as fl
from numpy import random
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import minimize
l1 = np.array([16.61,14.27,13.50,12.96,13.15,12.58])
l2 = np.array([13.72,12.53,12.53,12.14,12.74,11.86])
l3 = np.array([13.44,12.94,12.54,12.34,12.90,12.80])
l4 = np.array([13.38,12.56,12.57,12.47,13.10,12.50])

y1 = np.array([22.35,18.98,17.27,15.69,16.69,16.23])
y2 = np.array([17.34,15.98,14.18,14.22,16.27,14.03])
y3 = np.array([17.49,15.48,14.60,15.28,16.16,13.50])
y4 = np.array([17.25,17.37,16.73,16.73,19.35,14.90])

per = np.array([1000,2000,3000,4000,5000,6000])

plt.plot(per,l1,'.-',color = 'deepskyblue')
plt.plot(per,l2,'.-',color = 'deepskyblue')
plt.plot(per,l3,'.-',color = 'deepskyblue')
plt.plot(per,l4,'.-',color = 'deepskyblue')



plt.plot(per,y1,'.-',color = 'Black')
plt.plot(per,y2,'.-',color = 'Black')
plt.plot(per,y3,'.-',color = 'Black')
plt.plot(per,y4,'.-',color = 'Black')
#plt.ylim(0.1,0.55)
plt.xlim(0,6000)
plt.xlabel('Training examples')
plt.ylabel('Relative hedge error')
plt.title('Hedge experiment')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.text(760,22.2,'l = 1',horizontalalignment='right')
plt.text(900,17.4,'l = 2, 3, 4',horizontalalignment='right')
plt.text(760,16.5,'l = 1',horizontalalignment='right')
plt.text(960,13.5,'l = 2, 3, 4 ',horizontalalignment='right')
black_patch = mpatches.Patch(color='black', label='No regularization')
deepskyblue_patch = mpatches.Patch(color='deepskyblue', label='Differential regularization')
plt.legend(handles=[black_patch,deepskyblue_patch])