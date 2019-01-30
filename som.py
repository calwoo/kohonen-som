import numpy as np
from utils import *

"""
Here we'll build a basic Kohonen self-organizing map. We'll create a dataset of random RGB colors 
(3-vectors with values in [0,255]) and fit a SOM to it to figure out the organizational features of
the dataset. SOMs are usually used as a form of dimensionality reduction to 2d-feature spaces.
"""

# create dataset with 100 samples
data = np.random.randint(0, 255, (3, 100))
data = normalize_data(data, by_cols=True)

# hyperparameters
grid_dims = np.array([5,5])
weight_dim = data.shape[0]
num_examples = data.shape[1]
num_iters = 2000
lr_init = 0.01
# training radius hyperparams
radius_init = grid_dims[0] / 2
time_constant = num_iters / np.log(radius_init)

# SOM matrix
som = np.random.random((*grid_dims, weight_dim))

"""
The learning process for a SOM is as follows: for each example,
    1) Find neuron on SOM that is the closest in distance (by whatever metric we choose) to the example.
    2) Move this neuron (the BMU)'s weights close to that of the point.
    3) Within the training radius, move those neurons slightly closer to the point as well.
    4) Update the learning rate.
"""
# step 0: pick random example
example = data[:, np.random.randint(0, num_examples)].reshape(-1, 1)
# step 1: pick best matching unit (BMU) on SOM
bmu, bmu_id = bmu(example, som)
# step 2: move the neuron closer to point







# step 4: update learning rate
# both the radius and learning rate will decay according to exponential decay
radius = radius_init * np.exp(-it / time_constant)
lr = lr_init * np.exp(-it / num_iters)

