import numpy as np


def Rotate_Cartesian(theta, X, Y):
    return X * np.cos(theta) - Y * np.sin(theta), Y * np.cos(theta) + X * np.sin(theta)


def coord_to_index(x, y, image):

    return y - image.origin[0], x - image.origin[1]

def index_to_coord(i, j, image):

    return j + image.origin[1], i + image.origin[0]
