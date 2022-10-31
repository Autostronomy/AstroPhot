import numpy as np
import torch
import matplotlib.pyplot as plt
from autoprof.utils.interpolate import point_Lanczos

def center_of_mass(center, image, window = None):
    """Iterative light weighted center of mass optimization. Each step
    determines the light weighted center of mass within a small
    window. The new center is used to create a new window. This
    continues until the center no longer updates or an image boundary
    is reached.

    """
    if window is None:
        window = max(min(int(min(image.shape)/10), 30), 6)
    init_center = center
    window += window % 2
    xx, yy = np.meshgrid(np.arange(window), np.arange(window))
    for iteration in range(100):
        # Determine the image window to calculate COM
        ranges = [
            [int(round(center[0] - window/2)), int(round(center[0] + window/2))],
            [int(round(center[1] - window/2)), int(round(center[1] + window/2))]
        ]
        # Avoid edge of image
        if ranges[0][0] < 0 or ranges[1][0] < 0 or ranges[0][1] >= image.shape[0] or ranges[1][1] >= image.shape[1]:
            print('Image edge!')
            return init_center

        # Compute COM
        denom = np.sum(image[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]])
        new_center = [
            ranges[0][0] + np.sum(image[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]] * yy) / denom, 
            ranges[1][0] + np.sum(image[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]] * xx) / denom, 
        ]
        new_center = np.array(new_center)
        # Check for convergence
        if np.sum(np.abs(np.array(center) - new_center)) < 0.1:
            break
        
        center = new_center
        
    return center

def Lanczos_peak(center, image, L_scale = 3):

    res = minimize(lambda x: -point_Lanczos(image, x[0], x[1], scale = L_scale), x0 = center)

    return res.x
    
