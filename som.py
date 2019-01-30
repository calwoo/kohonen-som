import numpy as np
from tqdm import tqdm
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
# initialize training loop
radius = radius_init
lr = lr_init

for it in tqdm(range(num_iters)):
    # step 0: pick random example

    example = data[:, np.random.randint(0, num_examples)].reshape(-1, 1)
    # step 1: pick best matching unit (BMU) on SOM
    bmu, bmu_id = search_bmu(example, som)

    # step 2+3: move the neuron(s) closer to the point
    # points will be moved proportional to a gaussian decay centered around the bmu
    for x in range(grid_dims[0]):
        for y in range(grid_dims[1]):
            neuron_weight = som[x,y,:].reshape(weight_dim, 1)
            # get distance between neuron and bmu
            neuron_bmu_dist = np.sum((np.array([x,y]) - bmu_id)**2)
            # if neuron is within training radius of bmu, move it
            if neuron_bmu_dist <= radius**2:
                theta = gaussian_decay(neuron_bmu_dist, radius)
                # update the weight using "gradient descent"
                neuron_weight += lr * theta * (example - neuron_weight)
                som[x,y,:] = neuron_weight.reshape(1, weight_dim)

    # step 4: update learning rate
    # both the radius and learning rate will decay according to exponential decay
    radius = radius_init * np.exp(-it / time_constant)
    lr = lr_init * np.exp(-it / num_iters)

graph_som(som)
