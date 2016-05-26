from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt


import numpy as np
from mf import TriangularMF
from encoder import FuzzyAutoencoder


def get_data(data,i,j):
    x = data[:,i]
    y = data[:,j]
    
    x = x / max(x)
    y = y / max(y)
    
    return (x,y)
    
boston = datasets.load_boston()
print(boston.data.shape)

i = 4
j = 8
x0, x1 = get_data(boston.data, i, j)
x = np.array([x0,x1]).transpose()
#x = x[0:1,:]
print(x.shape)

    
    
encoder = FuzzyAutoencoder(num_inputs=2)

encoder.set_partition([
    TriangularMF([-0.1, 0, 0.5], 'lo'),
    TriangularMF([0, 0.5, 1], 'med'),
    TriangularMF([0.5, 1, 1.1], 'hi')
])

encoder.add_rule(np.array([
    [1, 0, 0], # x0 is lo
    [1, 0, 0]  # x1 is lo
]))
encoder.add_rule(np.array([
    [1, 0, 0], # x0 is hi
    [0, 1, 0]  # x1 is hi
]))
encoder.add_rule(np.array([
    [1, 0, 0], # x0 is hi
    [0, 0, 1]  # x1 is hi
]))

"""
encoder.add_rule(np.array([
    [0, 1, 0], # x0 is lo
    [1, 0, 0]  # x1 is lo
]))
encoder.add_rule(np.array([
    [0, 1, 0], # x0 is hi
    [0, 1, 0]  # x1 is hi
]))
encoder.add_rule(np.array([
    [0, 1, 0], # x0 is hi
    [0, 0, 1]  # x1 is hi
]))
"""

encoder.add_rule(np.array([
    [0, 0, 1], # x0 is lo
    [1, 0, 0]  # x1 is lo
]))
encoder.add_rule(np.array([
    [0, 0, 1], # x0 is hi
    [0, 1, 0]  # x1 is hi
]))
encoder.add_rule(np.array([
    [0, 0, 1], # x0 is hi
    [0, 0, 1]  # x1 is hi
]))

#encoder.add_random_rules(100, 0.5)

y = encoder.encode(x)
x_hat = encoder.decode(y)

con = encoder.consequents()

fig = plt.figure()
ax1 = fig.add_subplot(111)


ax1.scatter(con[:,0], con[:,1], s=100, c='y', marker='s', label='rules')
ax1.scatter(x[:,0], x[:,1], c='g', marker='o', label='original')
ax1.scatter(x_hat[:,0], x_hat[:,1], c='r', marker='o', label='recon')
plt.xlabel(i)
plt.ylabel(j)
plt.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()

