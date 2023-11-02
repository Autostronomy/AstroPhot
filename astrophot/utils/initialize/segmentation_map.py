from copy import deepcopy
from typing import Union

import numpy as np
from astropy.io import fits
from ..angle_operations import Angle_COM_PA

__all__ = (
    "centroids_from_segmentation_map",
    "PA_from_segmentation_map",
    "q_from_segmentation_map",
    "windows_from_segmentation_map",
    "scale_windows",
    "filter_windows",
)

def _select_img(img, hduli):
    if isinstance(img, str):
        if img.endswith(".fits"):
            hdul = fits.open(img)
            img = hdul[hduli].data
        elif img.endswith(".npy"):
            img = np.load(img)
        else:
            raise ValueError(
                f"unrecognized file type, should be one of: fits, npy\n{img}"
            )
    return img

def centroids_from_segmentation_map(
    seg_map: Union[np.ndarray, str],
    image: Union[np.ndarray, str],
    hdul_index_seg: int = 0,
    hdul_index_img: int = 0,
    skip_index: tuple = (0,),
):
    """identify centroid centers for all segments in a segmentation map

    For each segment in the map, computes a flux weighted centroid in
    pixel space. A dictionary of pixel centers is produced where the
    keys of the dictionary correspond to the segment id's.

    Args:
      seg_map (Union[np.ndarray, str]): A segmentation map which gives the object identity for each pixel
      image (Union[np.ndarray, str]): An Image which will be used in the light weighted center of mass calculation
      hdul_index_seg (int): If reading from a fits file this is the hdu list index at which the map is found. Default: 0
      hdul_index_img (int): If reading from a fits file this is the hdu list index at which the image is found. Default: 0
      skip_index (tuple): Lists which identities (if any) in the segmentation map should be ignored. Default (0,)

    Returns:
      centroids (dict): dictionary of centroid positions matched to each segment ID. The centroids are in pixel coordinates
    """

    seg_map = _select_img(seg_map, hdul_index_seg)
    image = _select_img(image, hdul_index_img)

    centroids = {}

    XX, YY = np.meshgrid(np.arange(seg_map.shape[1]), np.arange(seg_map.shape[0]))

    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        N = seg_map == index
        xcentroid = np.sum(XX[N] * image[N]) / np.sum(image[N])
        ycentroid = np.sum(YY[N] * image[N]) / np.sum(image[N])
        centroids[index] = [xcentroid, ycentroid]

    return centroids

def PA_from_segmentation_map(
    seg_map: Union[np.ndarray, str],
    image: Union[np.ndarray, str],
    centroids = None,
    hdul_index_seg: int = 0,
    hdul_index_img: int = 0,
    skip_index: tuple = (0,),
    north = np.pi/2,
):
    
    seg_map = _select_img(seg_map, hdul_index_seg)
    image = _select_img(image, hdul_index_img)
    
    if centroids is None:
        centroids = centroids_from_segmentation_map(seg_map=seg_map, image = image, skip_index=skip_index)
    
    XX, YY = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    PAs = {}
    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        N = seg_map == index
        PA = Angle_COM_PA(image[N], XX[N] - centroids[index][0], YY[N] - centroids[index][1]) + north
        PAs[index] = PA
        
    return PAs

def q_from_segmentation_map(
    seg_map: Union[np.ndarray, str],
    image: Union[np.ndarray, str],
    centroids = None,
    hdul_index_seg: int = 0,
    hdul_index_img: int = 0,
    skip_index: tuple = (0,),
    north = np.pi/2,
):
    
    seg_map = _select_img(seg_map, hdul_index_seg)
    image = _select_img(image, hdul_index_img)
    
    if centroids is None:
        centroids = centroids_from_segmentation_map(seg_map=seg_map, image = image, skip_index=skip_index)
    
    XX, YY = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    qs = {}
    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        N = seg_map == index
        theta = np.arctan2(YY[N] - centroids[index][1], XX[N] - centroids[index][0])
        
        # Ballpark correct, could be better
        ang_com_cos = np.sum(image[N] * np.cos(2*theta)) / np.sum(image[N])
        ang_com_sin = np.sum(image[N] * np.sin(2*theta)) / np.sum(image[N])
        qs[index] = 1. - (np.abs(ang_com_cos) + np.abs(ang_com_sin))
        
    return qs


def windows_from_segmentation_map(seg_map, hdul_index=0, skip_index=(0,)):
    """Convert a segmentation map into boinding boxes

    Takes a segmentation map as input and uses the segmentation ids to
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
            raise ValueError(
                f"unrecognized file type, should be one of: fits, npy\n{seg_map}"
            )

    windows = {}

    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        Yid, Xid = np.where(seg_map == index)
        # Get window from segmap
        windows[index] = [[np.min(Xid), np.max(Xid)], [np.min(Yid), np.max(Yid)]]

    return windows


def scale_windows(windows, image_shape=None, expand_scale=1.0, expand_border=0.0):
    new_windows = {}
    for index in list(windows.keys()):
        new_window = deepcopy(windows[index])
        # Get center and shape of the window
        center = (
            (new_window[0][0] + new_window[0][1]) / 2,
            (new_window[1][0] + new_window[1][1]) / 2,
        )
        shape = (
            new_window[0][1] - new_window[0][0],
            new_window[1][1] - new_window[1][0],
        )
        # Update the window with any expansion coefficients
        new_window = [
            [
                int(center[0] - expand_scale * shape[0] / 2 - expand_border),
                int(center[0] + expand_scale * shape[0] / 2 + expand_border),
            ],
            [
                int(center[1] - expand_scale * shape[1] / 2 - expand_border),
                int(center[1] + expand_scale * shape[1] / 2 + expand_border),
            ],
        ]
        # Ensure the window does not exceed the borders of the image
        if not image_shape is None:
            new_window = [
                [max(0, new_window[0][0]), min(image_shape[1], new_window[0][1])],
                [max(0, new_window[1][0]), min(image_shape[0], new_window[1][1])],
            ]
        new_windows[index] = new_window
    return new_windows


def filter_windows(
    windows,
    min_size=None,
    max_size=None,
    min_area=None,
    max_area=None,
    min_flux=None,
    max_flux=None,
    image=None,
):
    new_windows = {}
    for w in list(windows.keys()):
        if min_size is not None:
            if (
                min(
                    windows[w][0][1] - windows[w][0][0],
                    windows[w][1][1] - windows[w][1][0],
                )
                < min_size
            ):
                continue
        if max_size is not None:
            if (
                max(
                    windows[w][0][1] - windows[w][0][0],
                    windows[w][1][1] - windows[w][1][0],
                )
                > max_size
            ):
                continue
        if min_area is not None:
            if (
                (windows[w][0][1] - windows[w][0][0])
                * (windows[w][1][1] - windows[w][1][0])
            ) < min_area:
                continue
        if max_area is not None:
            if (
                (windows[w][0][1] - windows[w][0][0])
                * (windows[w][1][1] - windows[w][1][0])
            ) > max_area:
                continue
        if min_flux is not None:
            if (
                np.sum(
                    image[
                        windows[w][1][0] : windows[w][1][1],
                        windows[w][0][0] : windows[w][0][1],
                    ]
                )
                < min_flux
            ):
                continue
        if max_flux is not None:
            if (
                np.sum(
                    image[
                        windows[w][1][0] : windows[w][1][1],
                        windows[w][0][0] : windows[w][0][1],
                    ]
                )
                > max_flux
            ):
                continue
        new_windows[w] = windows[w]
    return new_windows
