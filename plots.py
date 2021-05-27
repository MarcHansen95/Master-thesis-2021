import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Softplus_super_Bachprice import Price_error

#This creates plot of results. The results are extracted from other files
# and this is just used for visualization of the results

plt.figure(0)
x_values1=[1,2,3,4,5]
y_values1=[0.00054252,0.00016172,0.00010228,.0001021,.00010266]

x_values2=[3,6,9,12,15]
y_values2=[0.01929204,0.00227684,0.00023712,.00014735,.00014618]


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

plt.title('Option Price Error')

ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()



plt.figure(1)
x_values1=[1,2,3,4,5]
y_values1=[0.00494707,0.00213441,0.001429,.00142917,.00143747]

x_values2=[3,6,9,12,15]
y_values2=[0.10637878,0.02253825,0.00356919,.00234638,.00235429]


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

plt.title('Option Delta Error')

ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()

plt.figure(2)
x_values1=[1,2,3,4,5]
y_values1=[0.00054252,0.00016172,0.00010228,.0001021,.00010266]

x_values2=[1,2,3,4,5]
y_values2=[0.00064275,0.00012221,8.814e-05,2.813e-05,5.42e-06]


fig=plt.figure()
ax=fig.add_subplot(111)
ax2=ax.twiny()
    
ax.plot(x_values1, y_values1,'.-', color="black")
ax.set_xlabel("Without derivatives", color="black")
ax.set_ylabel("Error", color="black")
ax.tick_params(axis='x', colors="black")
ax.tick_params(axis='y', colors="black")

ax2.plot(x_values2, y_values2,'.-', color="blue")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('With derivatives', color="blue") 
ax2.set_ylabel('Error', color="blue")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="blue")
ax2.tick_params(axis='y', colors="blue")

plt.title('Option Price Error')

ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()


plt.figure(3)
x_values1=[1,2,3,4,5]
y_values1=[0.00494707,0.00213441,0.001429,.00142917,.00143747]

x_values2=[1,2,3,4,5]
y_values2=[0.00478555,0.00147609,0.00102156,.00039413,9.745e-05]


fig=plt.figure()
ax=fig.add_subplot(111)
ax2=ax.twiny()
    
ax.plot(x_values1, y_values1,'.-', color="black")
ax.set_xlabel("Without derivatives", color="black")
ax.set_ylabel("Error", color="black")
ax.tick_params(axis='x', colors="black")
ax.tick_params(axis='y', colors="black")

ax2.plot(x_values2, y_values2,'.-', color="blue")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel('With derivatives', color="blue") 
ax2.set_ylabel('Error', color="blue")       
ax2.xaxis.set_label_position('top') 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='x', colors="blue")
ax2.tick_params(axis='y', colors="blue")

plt.title('Option Delta Error')

ax.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()

