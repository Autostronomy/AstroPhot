from copy import deepcopy
from typing import Optional, Union

import numpy as np
from astropy.io import fits
from ...backend_obj import backend
from ... import config

__all__ = (
    "centroids_from_segmentation_map",
    "PA_from_segmentation_map",
    "q_from_segmentation_map",
    "windows_from_segmentation_map",
    "scale_windows",
    "filter_windows",
    "transfer_windows",
)


def _select_img(img, hduli):
    if isinstance(img, str):
        if img.endswith(".fits"):
            hdul = fits.open(img)
            img = hdul[hduli].data
        elif img.endswith(".npy"):
            img = np.load(img)
        else:
            raise ValueError(f"unrecognized file type, should be one of: fits, npy\n{img}")
    return img


def centroids_from_segmentation_map(
    seg_map: Union[np.ndarray, str],
    image: "Image",
    sky_level: Optional[float] = None,
    hdul_index_seg: int = 0,
    skip_index: tuple = (0,),
):
    """identify centroid centers for all segments in a segmentation map

    For each segment in the map, computes a flux weighted centroid in
    pixel space. A dictionary of pixel centers is produced where the
    keys of the dictionary correspond to the segment id's.

    **Args:**
    -  `seg_map` (Union[np.ndarray, str]): A segmentation map which gives the object identity for each pixel
    -  `image` (Union[np.ndarray, str]): An Image which will be used in the light weighted center of mass calculation
    -  `sky_level` (float): The sky level to subtract from the image data before calculating centroids. Default: None, which uses the median of the image data.
    -  `hdul_index_seg` (int): If reading from a fits file this is the hdu list index at which the map is found. Default: 0
    -  `skip_index` (tuple): Lists which identities (if any) in the segmentation map should be ignored. Default (0,)
    """

    seg_map = _select_img(seg_map, hdul_index_seg)

    seg_map = seg_map.T
    if sky_level is None:
        sky_level = np.nanmedian(backend.to_numpy(image.data))

    data = backend.to_numpy(image.data) - sky_level
    centroids = {}

    II, JJ = np.meshgrid(np.arange(seg_map.shape[0]), np.arange(seg_map.shape[1]), indexing="ij")

    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        N = seg_map == index
        icentroid = np.sum(II[N] * data[N]) / np.sum(data[N])
        jcentroid = np.sum(JJ[N] * data[N]) / np.sum(data[N])
        xcentroid, ycentroid = image.pixel_to_plane(
            backend.as_array(icentroid, dtype=config.DTYPE, device=config.DEVICE),
            backend.as_array(jcentroid, dtype=config.DTYPE, device=config.DEVICE),
            params=(),
        )
        centroids[index] = [xcentroid.item(), ycentroid.item()]

    return centroids


def PA_from_segmentation_map(
    seg_map: Union[np.ndarray, str],
    image: "Image",
    centroids=None,
    sky_level: Optional[float] = None,
    hdul_index_seg: int = 0,
    skip_index: tuple = (0,),
    softening: float = 1e-3,
):

    seg_map = _select_img(seg_map, hdul_index_seg)

    # reverse to match numpy indexing
    seg_map = seg_map.T
    if sky_level is None:
        sky_level = np.nanmedian(backend.to_numpy(image.data))

    data = backend.to_numpy(image.data) - sky_level

    if centroids is None:
        centroids = centroids_from_segmentation_map(
            seg_map=seg_map, image=image, skip_index=skip_index
        )

    x, y = image.coordinate_center_meshgrid()
    x = backend.to_numpy(x)
    y = backend.to_numpy(y)
    PAs = {}
    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        N = seg_map == index
        xx = x[N] - centroids[index][0]
        yy = y[N] - centroids[index][1]
        mu20 = np.median(data[N] * np.abs(xx))
        mu02 = np.median(data[N] * np.abs(yy))
        mu11 = np.median(data[N] * xx * yy / np.sqrt(np.abs(xx * yy) + softening**2))
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
            PAs[index] = np.pi / 2
        else:
            PAs[index] = (0.5 * np.arctan2(2 * mu11, mu20 - mu02) - np.pi / 2) % np.pi

    return PAs


def q_from_segmentation_map(
    seg_map: Union[np.ndarray, str],
    image: "Image",
    centroids=None,
    sky_level: Optional[float] = None,
    hdul_index_seg: int = 0,
    skip_index: tuple = (0,),
    softening: float = 1e-3,
):

    seg_map = _select_img(seg_map, hdul_index_seg)

    # reverse to match numpy indexing
    seg_map = seg_map.T

    if sky_level is None:
        sky_level = np.nanmedian(backend.to_numpy(image.data))

    data = backend.to_numpy(image.data) - sky_level

    if centroids is None:
        centroids = centroids_from_segmentation_map(
            seg_map=seg_map, image=image, skip_index=skip_index
        )

    x, y = image.coordinate_center_meshgrid()
    x = backend.to_numpy(x)
    y = backend.to_numpy(y)
    qs = {}
    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        N = seg_map == index
        xx = x[N] - centroids[index][0]
        yy = y[N] - centroids[index][1]
        mu20 = np.median(data[N] * np.abs(xx))
        mu02 = np.median(data[N] * np.abs(yy))
        mu11 = np.median(data[N] * xx * yy / np.sqrt(np.abs(xx * yy) + softening**2))
        M = np.array([[mu20, mu11], [mu11, mu02]])
        if np.any(np.iscomplex(M)) or np.any(~np.isfinite(M)):
            qs[index] = 0.7
        else:
            l = np.abs(np.sort(np.linalg.eigvals(M)))
            qs[index] = np.clip(np.sqrt(l[0] / l[1]), 0.1, 0.9)

    return qs


def windows_from_segmentation_map(seg_map, hdul_index=0, skip_index=(0,)):
    """Convert a segmentation map into boinding boxes

    Takes a segmentation map as input and uses the segmentation ids to
    determine bounding boxes for every object. Scales the bounding
    boxes according to given factors and returns the coordinates.

    each window is formatted as a list of lists with:
    window = [[xmin,ymin],[xmax,ymax]]

    expand_scale changes the base window by the given
    factor. expand_border is added afterwards on all sides (so an
    expand border of 1 will add 2 to the total width of the window.

    """

    seg_map = _select_img(seg_map, hdul_index)

    seg_map = seg_map.T

    windows = {}

    for index in np.unique(seg_map):
        if index is None or index in skip_index:
            continue
        Iid, Jid = np.where(seg_map == index)
        # Get window from segmap
        windows[index] = [[np.min(Iid), np.min(Jid)], [np.max(Iid), np.max(Jid)]]

    return windows


def scale_windows(windows, image: "Image" = None, expand_scale=1.0, expand_border=0.0):
    new_windows = {}
    for index in list(windows.keys()):
        new_window = deepcopy(windows[index])
        # Get center and shape of the window
        center = (
            (new_window[0][0] + new_window[1][0]) / 2,
            (new_window[0][1] + new_window[1][1]) / 2,
        )
        shape = (
            new_window[1][0] - new_window[0][0],
            new_window[1][1] - new_window[0][1],
        )
        # Update the window with any expansion coefficients
        new_window = [
            [
                int(center[0] - expand_scale * shape[0] / 2 - expand_border),
                int(center[1] - expand_scale * shape[1] / 2 - expand_border),
            ],
            [
                int(center[0] + expand_scale * shape[0] / 2 + expand_border),
                int(center[1] + expand_scale * shape[1] / 2 + expand_border),
            ],
        ]
        # Ensure the window does not exceed the borders of the image
        if image is not None:
            new_window = [
                [max(0, new_window[0][0]), max(0, new_window[0][1])],
                [
                    min(image.data.shape[0], new_window[1][0]),
                    min(image.data.shape[1], new_window[1][1]),
                ],
            ]
        new_windows[index] = new_window
    return new_windows


def filter_windows(
    windows,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    min_flux: Optional[float] = None,
    max_flux: Optional[float] = None,
    image: "Image" = None,
):
    """
    Filter a set of windows based on a set of criteria.

    **Args:**
    -  `windows`: A dictionary of windows to filter. Each window is formatted as a list of lists with: window = [[xmin,ymin],[xmax,ymax]]
    -  `min_size`: minimum size of the window in pixels
    -  `max_size`: maximum size of the window in pixels
    -  `min_area`: minimum area of the window in pixels
    -  `max_area`: maximum area of the window in pixels
    -  `min_flux`: minimum flux of the window in ADU
    -  `max_flux`: maximum flux of the window in ADU
    -  `image`: the image from which the flux is calculated for min_flux and max_flux
    """
    new_windows = {}
    for w in list(windows.keys()):
        if min_size is not None:
            if (
                min(
                    windows[w][1][0] - windows[w][0][0],
                    windows[w][1][1] - windows[w][0][1],
                )
                < min_size
            ):
                continue
        if max_size is not None:
            if (
                max(
                    windows[w][1][0] - windows[w][0][0],
                    windows[w][1][1] - windows[w][0][1],
                )
                > max_size
            ):
                continue
        if min_area is not None:
            if (
                (windows[w][1][0] - windows[w][0][0]) * (windows[w][1][1] - windows[w][0][1])
            ) < min_area:
                continue
        if max_area is not None:
            if (
                (windows[w][1][0] - windows[w][0][0]) * (windows[w][1][1] - windows[w][0][1])
            ) > max_area:
                continue
        if min_flux is not None:
            if (
                np.sum(
                    backend.to_numpy(image.data)[
                        windows[w][0][0] : windows[w][1][0],
                        windows[w][0][1] : windows[w][1][1],
                    ]
                )
                < min_flux
            ):
                continue
        if max_flux is not None:
            if (
                np.sum(
                    backend.to_numpy(image.data)[
                        windows[w][0][0] : windows[w][1][0],
                        windows[w][0][1] : windows[w][1][1],
                    ]
                )
                > max_flux
            ):
                continue
        new_windows[w] = windows[w]
    return new_windows


def transfer_windows(windows, base_image, new_image):
    """
    Convert a set of windows from one image object to another. This will account
    for the relative adjustments in origin, pixelscale, and rotation between the
    two images.

    **Args:**
    -  `windows`: A dictionary of windows to be transferred. Each window is formatted as a list of lists with: window = [[xmin,ymin],[xmax,ymax]]
    -  `base_image`: The image object from which the windows are being transferred.
    -  `new_image`: The image object to which the windows are being transferred.
    """
    new_windows = {}
    for w in list(windows.keys()):
        four_corners_base = backend.as_array(
            [
                windows[w][0],
                windows[w][1],
                [windows[w][0][0], windows[w][1][1]],
                [windows[w][1][0], windows[w][0][1]],
            ],
            dtype=base_image.data.dtype,
            device=base_image.data.device,
        )  # (4,2)
        four_corners_new = backend.to_numpy(
            backend.stack(
                new_image.plane_to_pixel(*base_image.pixel_to_plane(*four_corners_base.T)), dim=-1
            )
        )  # (4,2)

        bottom_corner = np.floor(np.min(four_corners_new, axis=0)).astype(int)
        bottom_corner = np.clip(bottom_corner, 0, np.array(new_image.shape))
        top_corner = np.ceil(np.max(four_corners_new, axis=0)).astype(int)
        top_corner = np.clip(top_corner, 0, np.array(new_image.shape))
        new_windows[w] = [
            [int(bottom_corner[0]), int(bottom_corner[1])],
            [int(top_corner[0]), int(top_corner[1])],
        ]
    return new_windows
