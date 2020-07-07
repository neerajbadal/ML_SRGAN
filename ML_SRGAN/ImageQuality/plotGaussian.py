'''
Created on 13-Jun-2020

@author: Neeraj Badal
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 12 
sigma = 4
x1 = 5
x2 = 20

# calculate the z-transform
z1 = ( x1 - mu ) / sigma
z2 = ( x2 - mu ) / sigma

# x = np.arange(z1, z2, 0.001) # range of x in spec
x = np.arange(-2, 10, 0.001) # range of x in spec
# mean = 0, stddev = 1, since Z-transform was calculated
y = norm.pdf(x,3,1)

mu = 28 
sigma = 2
x1 = 15
x2 = 40

# calculate the z-transform
z1 = ( x1 - mu ) / sigma
z2 = ( x2 - mu ) / sigma

# x_2 = np.arange(z1, z2, 0.001) # range of x in spec
x_2 = np.arange(0, 12, 0.001) # range of x in spec
y_2 = norm.pdf(x_2,6,1)


# build the plot
fig, ax = plt.subplots(figsize=(9,6))
plt.style.use('fivethirtyeight')
ax.plot(x,y)
ax.plot(x_2,y_2)
# ax.fill_between(x,y,0, alpha=0.3, color='b')
ax.fill_between(x,y,0, alpha=0.3)
ax.fill_between(x_2,y_2,0, alpha=0.3)
ax.set_xlim([-10,10])
ax.set_xlabel('s_i')
ax.set_ylabel('p')
ax.set_yticklabels([])

plt.show()