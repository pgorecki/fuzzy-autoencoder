# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:11:04 2016

@author: Przemek
"""

import numpy as np
from mf import TriangularMF
from encoder import FuzzyAutoencoder

encoder = FuzzyAutoencoder(num_inputs=2)

encoder.set_partition([
    TriangularMF([-0.1, 0, 0.5], 'lo'),
    TriangularMF([0, 0.5, 1], 'med'),
    TriangularMF([0.5, 1, 1.1], 'hi')
])

encoder.add_rule(np.array([
    [1, 0, 0], # x0 is lo
    [0, 0, 1]  # x1 is hi
]))

encoder.add_rule(np.array([
    [1, 0, 0], # x0 is lo
    [1, 0, 0]  # x1 is lo
]))

encoder.add_rule(np.array([
    [0, 1, 0], # x0 is med
    [0, 1, 0]  # x1 is med
]))

#encoder.add_random_rules(3, 0.5)
print(encoder)

# in this case, input is reconstructed without an error
x = np.array([0,0])
y = encoder.encode(x)
x_prime = encoder.decode(y)
print x, y, x_prime, encoder.loss(x, x_prime)

# in this case, input is reconstructed with two weak rules
x = np.array([0.2,0.2])
y = encoder.encode(x)
x_prime = encoder.decode(y)
print x,y, x_prime, encoder.loss(x, x_prime)

# in this case, none of the rules had fired
# so this input cannot be reconstructed
x = np.array([[1,1],[1,1]])
y = encoder.encode(x)
x_prime = encoder.decode(y)
print x,y, x_prime, encoder.loss(x, x_prime)


# in this case, none of the rules had fired
# but we are using nan_replacer
x = np.array([[1,1],[1,1]])
y = encoder.encode(x)
x_prime = encoder.decode(y)
print x,y, x_prime, encoder.loss(x, x_prime)


