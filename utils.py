import numpy as np 


# data normalize to approx. [0,1]
def normalize_data(data, by_cols=True):
    max_nums = np.max(data, axis=1)
    if by_cols:
        normed_data = data / max_nums[:, np.newaxis]
    else:
        normed_data = data / np.max(data)
    return normed_data

# find best matching unit
def bmu(example, som):
    first_flag = True
    dim = example.shape[0]
    # get distance between neurons and example, get min
    for x in range(som.shape[0]):
        for y in range(som.shape[1]):
            neuron_weight = som[x,y,:].reshape(dim, 1)
            dist = np.sum((example-neuron_weight)**2)
            if first_flag:
                min_dist = dist
                bmu_id = np.array([x,y])
                first_flag = False
            elif dist < min_dist:
                min_dist = dist
                bmu_id = np.array([x,y])
    bmu = som[bmu_id[0], bmu_id[1], :].reshape(dim, 1)
    return bmu, bmu_id

# gaussian decay from center of bmu
def gaussian_decay(d, radius):
    exponent = -d**2 / (2 * radius**2)
    return np.exp(exponent)

