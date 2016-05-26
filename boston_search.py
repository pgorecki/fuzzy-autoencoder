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

# ------------------------------------------------------------    
    
encoder = FuzzyAutoencoder(num_inputs=2)

encoder.set_partition([
    TriangularMF([-0.1, 0, 0.5], 'lo'),
    TriangularMF([0, 0.5, 1], 'med'),
    TriangularMF([0.5, 1, 1.1], 'hi')
])

encoder.add_random_rules(3, 0.5)

def plot_encoder(x, encoder, title=None):
    y = encoder.encode(x)
    x_hat = encoder.decode(y)
    con = encoder.consequents()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(con[:,0], con[:,1], s=100, c='y', marker='s', label='rules')
    ax1.scatter(x[:,0], x[:,1], c='g', marker='o', label='original')
    ax1.scatter(x_hat[:,0], x_hat[:,1], c='r', marker='o', label='recon')
    if title:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.3, 1.0))
    plt.show()


# this is how the random model is doing
model = encoder.get_state()
score = encoder.loss(x, x_hat)

print "initial"
print model
print score
plot_encoder(x, encoder, 'Initial: %f' % score)

# the goal is to select all the values that will 
# maximize reward (positive ones)
def random_search(x, encoder):
    # get current score
    y = encoder.encode(x)
    x_hat = encoder.decode(y)
    score = encoder.loss(x, x_hat)
    original = encoder.get_state()
    modified = np.array(original)
    
    # choose random index and change its value
    i = int(np.random.rand()*modified.shape[0])
    modified[i] = not modified[i]
    encoder.set_state(modified)
    y = encoder.encode(x)
    x_hat = encoder.decode(y)
    modified_score = encoder.loss(x, x_hat)
    
    if modified_score < score:
        print score
        return modified_score
    else:
        encoder.set_state(original)
        return score


num_iterations = 100
history = []
for i in range(0, num_iterations):
    score = random_search(x, encoder)
    history.append(score)
    
plt.plot(history)

print "optimized by random search"
print model
print score
plot_encoder(x, encoder, 'Optimized: %f' % score)
