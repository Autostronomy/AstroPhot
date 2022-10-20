import numpy as np
import torch
import matplotlib.pyplot as plt

def center_of_mass(center, image, window = 10):
    init_center = center
    window += window % 2
    xx, yy = np.meshgrid(np.arange(window)+1, np.arange(window)+1)
    for iteration in range(100):
        # Determine the image window to calculate COM
        ranges = [
            [int(round(center[0] - window/2)), int(round(center[0] + window/2))],
            [int(round(center[1] - window/2)), int(round(center[1] + window/2))]
        ]
        # Avoid edge of image
        if ranges[0][0] < 0 or ranges[1][0] < 0 or ranges[0][1] >= image.shape[1] or ranges[1][1] >= image.shape[0]:
            print('Image edge!')
            return init_center

        # Compute COM
        denom = np.sum(image[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]])
        new_center = [
            ranges[0][0] + np.sum(image[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] * xx) / denom, 
            ranges[1][0] + np.sum(image[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]] * yy) / denom, 
        ]
        new_center = np.array(new_center)
        # Check for convergence
        if np.sum(np.abs(np.array(center) - new_center)) < 0.1:
            break
        
        center = new_center
        
    return center
