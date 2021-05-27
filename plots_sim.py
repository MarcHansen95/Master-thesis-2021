import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#This creates plot of results. The results are extracted from other files
# and this is just used for visualization of the results


plt.figure(0)
x_values1=[1,2,3,4,5,6]
y_values1=[0.08844029,0.08696662,0.08700127,.08700129,.08700126,.08700127]

x_values2=[3,6,9,12,15,18]
y_values2=[0.09412611,0.08717234,0.08705981,.0870701,.08714164,1.0599643]


fig=plt.figure()
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
plt.ylim(0.08,.12)
plt.title('Option Price Error')

ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()



plt.figure(1)
x_values1=[1,2,3,4,5,6]
y_values1=[0.02074606,0.02795418,0.02866388,.02866366,.02866397,.02866388]

x_values2=[3,6,9,12,15,18]
y_values2=[0.10430225,0.03056811,0.02584395,.02988953,.02819718,1.28075722]


fig=plt.figure()
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
plt.ylim(-0.0012,.12)
plt.title('Option Delta Error')

ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()

