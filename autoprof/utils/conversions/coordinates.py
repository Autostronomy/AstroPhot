import torch
import numpy as np

def Rotate_Cartesian(theta, X, Y):
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = torch.sin(theta)
    c = torch.cos(theta) 
    return c*X - s*Y, c*Y + s*X

def Rotate_Cartesian_np(theta, X, Y):
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = np.sin(theta)
    c = np.cos(theta) 
    return c*X - s*Y, c*Y + s*X

def Axis_Ratio_Cartesian(q, X, Y, theta = 0., inv_scale = False):
    """
    Applies the transformation: R(theta) Q R(-theta)
    where R is the rotation matrix and Q is the matrix which scales the y component by 1/q.
    This effectively counter-rotates the coordinates so that the angle theta is along the x-axis
    then applies the y-axis scaling, then re-rotates everything back to where it was.
    """
    if inv_scale:
        scale = (1 / q) - 1
    else:
        scale = q - 1
    ss = 1 + scale * torch.pow(torch.sin(theta),2)
    cc = 1 + scale * torch.pow(torch.cos(theta),2)
    s2 = scale * torch.sin(2*theta)
    return ss*X - s2*Y/2, -s2*X/2 + cc*Y

def Axis_Ratio_Cartesian_np(q, X, Y, theta = 0., inv_scale = False):
    """
    Applies the transformation: R(theta) Q R(-theta)
    where R is the rotation matrix and Q is the matrix which scales the y component by 1/q.
    This effectively counter-rotates the coordinates so that the angle theta is along the x-axis
    then applies the y-axis scaling, then re-rotates everything back to where it was.
    """
    if inv_scale:
        scale = (1 / q) - 1
    else:
        scale = q - 1
    ss = 1 + scale * np.sin(theta)**2
    cc = 1 + scale * np.cos(theta)**2
    s2 = scale * np.sin(2*theta)
    return ss*X - s2*Y/2, -s2*X/2 + cc*Y

def coord_to_index(x, y, image):
    """
    input: x,y in arcsec of real position
    output: i,j in image array indexing units (note that the values will be float an need to be rounded)
    """
    
    return (y - image.origin[1]) / image.pixelscale, (x - image.origin[0]) / image.pixelscale

def index_to_coord(i, j, image):
    """
    input: i,j in image array indexing
    output: x,y in arcsec of real position
    """

    return j*image.pixelscale + image.origin[0], i*image.pixelscale + image.origin[1]
