import numpy as np


def Rotate_Cartesian(theta, X, Y):
    return X * np.cos(theta) - Y * np.sin(theta), Y * np.cos(theta) + X * np.sin(theta)

