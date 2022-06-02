import numpy as np


def Rotate_Cartesian(theta, X, Y):
    return X * np.cos(theta) - Y * np.sin(theta), Y * np.cos(theta) + X * np.sin(theta)


def coord_to_index(x, y, image):
    """
    input: x,y in arcsec of real position
    output: i,j in image array indexing units (note that the values will be float an need to be rounded)
    """
    
    return (y - image.origin[0]) / image.pixelscale, (x - image.origin[1]) / image.pixelscale

def index_to_coord(i, j, image):
    """
    input: i,j in image array indexing
    output: x,y in arcsec of real position
    """

    return j*image.pixelscale + image.origin[1], i*image.pixelscale + image.origin[0]
