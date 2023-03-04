import numpy as np
from astropy.io import fits
from copy import deepcopy

def windows_from_segmentation_map(seg_map, hdul_index = 0, skip_index = (0,)):
    """Takes a segmentation map as input and uses the segmentation ids to
    determine bounding boxes for every object. Scales the bounding
    boxes according to given factors and returns the coordinates.

    each window is formatted as a list of lists with:
    window = [[xmin,xmax],[ymin,ymax]]

    expand_scale changes the base window by the given
    factor. expand_border is added afterwards on all sides (so an
    expand border of 1 will add 2 to the total width of the window.

    """
    
    if isinstance(seg_map, str):
        if seg_map.endswith(".fits"):
            hdul = fits.open(seg_map)
            seg_map = hdul[hdul_index].data
        elif seg_map.endswith(".npy"):
            seg_map = np.load(seg_map)
        else:
            raise ValueError(f"unrecognized file type, should be one of: fits, npy\n{seg_map}")

    windows = {}

    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        Yid, Xid = np.where(seg_map == index)
        # Get window from segmap
        windows[index] = [
            [np.min(Xid), np.max(Xid)],
            [np.min(Yid), np.max(Yid)]
        ]
        
    return windows

def scale_windows(windows, image_shape = None, expand_scale = 1., expand_border = 0.):
    new_windows = {}
    for index in list(windows.keys()):
        new_window = deepcopy(windows[index])
        # Get center and shape of the window
        center = (
            (new_window[0][0] + new_window[0][1])/2,
            (new_window[1][0] + new_window[1][1])/2,
        )
        shape = (
            new_window[0][1] - new_window[0][0],
            new_window[1][1] - new_window[1][0],
        )
        # Update the window with any expansion coefficients
        new_window = [
            [int(center[0] - expand_scale*shape[0]/2 - expand_border), int(center[0] + expand_scale*shape[0]/2 + expand_border)],
            [int(center[1] - expand_scale*shape[1]/2 - expand_border), int(center[1] + expand_scale*shape[1]/2 + expand_border)],
        ]
        # Ensure the window does not exceed the borders of the image
        if not image_shape is None:
            new_window = [
                [max(0,new_window[0][0]), min(image_shape[1], new_window[0][1])],
                [max(0,new_window[1][0]), min(image_shape[0], new_window[1][1])],
            ]
        new_windows[index] = new_window
    return new_windows

def filter_windows(windows, min_size = None, max_size = None, min_area = None, max_area = None, min_flux = None, max_flux = None, image = None):

    new_windows = {}
    for w in list(windows.keys()):
        if min_size is not None:
            if min(windows[w][0][1] - windows[w][0][0], windows[w][1][1] - windows[w][1][0]) < min_size:
                continue
        if max_size is not None:
            if max(windows[w][0][1] - windows[w][0][0], windows[w][1][1] - windows[w][1][0]) > max_size:
                continue
        if min_area is not None:
            if ((windows[w][0][1] - windows[w][0][0])*(windows[w][1][1] - windows[w][1][0])) < min_area:
                continue
        if max_area is not None:
            if ((windows[w][0][1] - windows[w][0][0])*(windows[w][1][1] - windows[w][1][0])) > max_area:
                continue
        if min_flux is not None:
            if np.sum(image[windows[w][1][0]:windows[w][1][1],windows[w][0][0]:windows[w][0][1]]) < min_flux:
                continue
        if max_flux is not None:
            if np.sum(image[windows[w][1][0]:windows[w][1][1],windows[w][0][0]:windows[w][0][1]]) > max_flux:
                continue
        new_windows[w] = windows[w]
    return new_windows
