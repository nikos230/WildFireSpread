import numpy

# normalize data in range (0, 1)
def normalize(data):
    return (data - np.min(data) / (np.max(data) - np.min(data)))