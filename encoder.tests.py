from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import sys

import numpy as np
from mf import TriangularMF
from encoder import FuzzyAutoencoder

# both rule systems code the same knowledge

# test 1

encoder = FuzzyAutoencoder(num_inputs=2)
encoder.set_partition([
    TriangularMF([-0.1, 0, 0.5], 'lo'),
    TriangularMF([0, 0.5, 1], 'med'),
    TriangularMF([0.5, 1, 1.1], 'hi')
])
encoder.add_rule(np.array([
    [0, 0, 1], # x0 is hi
    [0, 0, 1]  # x1 is hi
]))

x = np.array([[1, 1]])
y = encoder.encode(x)
x_hat = encoder.decode(y)

assert np.array_equal(y, np.array([[1]]))
assert np.array_equal(x_hat, x)


# test 2

encoder = FuzzyAutoencoder(num_inputs=2)
encoder.set_partition([
    TriangularMF([-0.1, 0, 0.5], 'lo'),
    TriangularMF([0, 0.5, 1], 'med'),
    TriangularMF([0.5, 1, 1.1], 'hi')
])
encoder.add_rule(np.array([
    [0, 0, 1], # x0 is hi
    [0, 0, 0]
]))
encoder.add_rule(np.array([
    [0, 0, 0],
    [0, 0, 1]  # x1 is hi
]))

x = np.array([[1, 1]])
y = encoder.encode(x)
x_hat = encoder.decode(y)

assert np.array_equal(y, np.array([[1, 1]]))
assert np.array_equal(x_hat, x)

# test 3

encoder = FuzzyAutoencoder(num_inputs=2)
encoder.set_partition([
    TriangularMF([-0.1, 0, 0.5], 'lo'),
    TriangularMF([0, 0.5, 1], 'med'),
    TriangularMF([0.5, 1, 1.1], 'hi')
])
encoder.add_rule(np.array([
    [0, 1, 1], # x0 is med or hi
    [0, 0, 0]
]))
encoder.add_rule(np.array([
    [0, 0, 0],
    [0, 1, 1]  # x1 is med or hi
]))

x = np.array([[0.75, 0.75]])
y = encoder.encode(x)
x_hat = encoder.decode(y)

assert np.array_equal(y, np.array([[1, 1]]))
assert np.array_equal(x_hat, x)

