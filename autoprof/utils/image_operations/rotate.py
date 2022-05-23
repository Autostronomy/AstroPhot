import numpy as np

def right_rotate(img, angle):
    assert angle % 90 == 0

def rotate_coordinates(X, Y, theta):
    return X * np.cos(theta) - Y * np.sin(theta), Y * np.cos(theta) + X * np.sin(theta)
    
def arbitrary_rotate(img, angle, crop = 'zero pad'):
    pass
