# a toy for random search optimization

import numpy as np
import matplotlib.pyplot as plt

num_params = 1000
num_iterations = 10000

def evaluate(model, rewards):
    return np.sum(model * rewards)


model = np.zeros(num_params)
rewards = np.floor(np.random.rand(num_params)*200 - 100)

# the goal is to select all the values that will 
# maximize reward (positive ones)

def random_search(model, best_score, rewards):
    current = np.array(model)
    # choose random index and change its value
    i = int(np.random.rand()*model.shape[0])
    current[i] = not current[i]
    current_score = evaluate(current, rewards)
    if current_score > best_score:
        return (current, current_score)
    else:
        return (model, best_score)
    
    
    
best_true = ((rewards > 0).astype('float') * rewards).sum()

score = evaluate(model, rewards)

history = []

for i in range(0, num_iterations):
    model, score = random_search(model, score, rewards)
    history.append(score)
    
plt.plot(history)

print np.vstack([rewards, model])
print model, score, best_true