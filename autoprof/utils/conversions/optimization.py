import numpy as np

def boundaries(val, limits):
    if limits[0] is None:
        return val - 1 / (val - limits[1])
    elif limits[1] is None:
        return val - 1 / (val - limits[0])
    return np.arctanh((val - limits[0]) * 2 / (limits[1] - limits[0]) - 1) #limits[0] + (limits[1] - limits[0]) * 0.5 * (np.arctanh(val) + 1)

def inv_boundaries(val, limits):
    if limits[0] is None:
        return (val + limits[1] - np.sqrt((val - limits[1])**2 + 4)) * 0.5
    elif limits[1] is None:
        return (val + limits[0] + np.sqrt((val - limits[0])**2 + 4)) * 0.5
    return (np.tanh(val) + 1) * 0.5 *(limits[1] - limits[0]) + limits[0]
