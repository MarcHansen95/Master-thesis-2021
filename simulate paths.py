#This creates Figure 2.1 


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8)

fig = plt.figure()
        
T = 1
N = 50 # Number of points, number of subintervals = N-1
dt = T/(N-1) # Time step
t = np.linspace(0,T,N)
M = 20 # Number of walkers/paths

dX = np.sqrt(dt) * np.random.randn(M, N)
X = np.cumsum(dX, axis=1)
xx = np.arange(0.35,0.85,0.025)
#X = X + 10

for i in range(M):
    plt.plot(t, X[i,:])
  

plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('Stock prices', fontsize=14)
plt.title('Simulated paths of stock prices', fontsize=14)
axes = plt.gca()
axes.set_xlim([0,1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks([])
plt.tight_layout()
plt.show()


K = np.zeros(M)
for i in range(0,len(X[:,49])):
    if X[i,49]>=0:
        K[i] = X[i,49]
    else:
        K[i] = 0
plt.figure(1)


plt.plot(xx,K,'.')
plt.xlabel('Training input - Initial states', fontsize=14)
plt.ylabel('Training labels - Final payoffs', fontsize=14)
plt.title('Option payoff', fontsize=14)
axes = plt.gca()
#axes.set_xlim([0,.351])
axes.set_ylim([-0.09,1.75])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xticks([])
#plt.yticks([])
plt.tight_layout()
plt.show()