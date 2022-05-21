import numpy as np

def boundaries(val, limits):
    """
    val in limits expanded to range -inf to inf
    """
    if limits[0] is None:
        return val - 1 / (val - limits[1])
    elif limits[1] is None:
        return val - 1 / (val - limits[0])
    return np.tan((val - limits[0]) * np.pi / (limits[1] - limits[0]) - np.pi/2)

def inv_boundaries(val, limits):
    """
    val in range -inf to inf compressed to within the limits
    """
    
    if limits[0] is None:
        return (val + limits[1] - np.sqrt((val - limits[1])**2 + 4)) * 0.5
    elif limits[1] is None:
        return (val + limits[0] + np.sqrt((val - limits[0])**2 + 4)) * 0.5
    return (np.arctan(val) + np.pi/2) * (limits[1] - limits[0]) / np.pi + limits[0]

def cyclic_boundaries(val, limits):
    return limits[0] + ((val - limits[0]) % (limits[1] - limits[0]))

def cyclic_difference(val1, val2, period):
    return np.arcsin(np.sin((val1 - val2) * np.pi / period)) * period / np.pi
