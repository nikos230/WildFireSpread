import numpy as np

# normalize data in range (0, 1)
def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max == data_min:
        return np.zeros_like(data)
    else:    
        return (data - np.min(data) / (np.max(data) - np.min(data)))