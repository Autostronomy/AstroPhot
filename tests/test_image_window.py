"""Window and WindowList, the pixel regions of an image (astrophot/image/window.py)."""

import astrophot as ap
import numpy as np
import pytest


@pytest.fixture
def big_image():
    """Blank Image with 100 rows and 110 columns."""
    return ap.Image(data=np.zeros((100, 110)), pixelscale=0.3, zeropoint=1.0, name="test_image")


######################################################################
# Window
######################################################################


def test_window_creation(big_image):
    window = ap.Window((2, 107, 3, 97), big_image)

    assert np.all(window.crpix == big_image.crpix), "Window should inherit crpix from image"
    assert window.identity == big_image.identity, "Window should inherit identity from image"
    assert window.shape == (105, 94), "Window should have correct shape"
    assert window.extent == (2, 107, 3, 97), "Window should have correct extent"
    assert str(window) == "Window(2, 107, 3, 97)", "String representation should match"


def test_window_chunk(big_image):
    window = ap.Window((2, 107, 3, 97), big_image)

    subwindows = window.chunk(10**2)
    reconstitute = subwindows[0]
    for subwindow in subwindows:
        reconstitute |= subwindow
    assert (
        reconstitute.extent == window.extent
    ), "chunked windows should reconstitute to original window"


def test_window_arithmetic(big_image):
    windowbig = ap.Window((2, 107, 3, 97), big_image)
    windowsmall = ap.Window((20, 45, 30, 90), big_image)

    big_or_small = windowbig | windowsmall
    assert big_or_small.extent == (
        2,
        107,
        3,
        97,
    ), "logical or of windows should take largest bounding box"

    big_and_small = windowbig & windowsmall
    assert big_and_small.extent == (
        20,
        45,
        30,
        90,
    ), "logical and of windows should take overlap region"


######################################################################
# WindowList
######################################################################


def test_windowlist_creation(image_pair):
    image1, image2 = image_pair
    window1 = ap.Window([4, 13, 5, 9], image1)
    window2 = ap.Window([0, 7, 1, 8], image2)
    windowlist = ap.WindowList([window1, window2])

    window3 = ap.Window([3, 12, 5, 8], image1)
    assert windowlist.index(window3) == 0, "WindowList should find window by index"
    assert len(windowlist) == 2, "WindowList should have two windows"

    window21 = ap.Window([5, 10, 6, 9], image1)
    window22 = ap.Window([0, 9, 0, 8], image2)
    windowlist2 = ap.WindowList([window21, window22])

    windowlist_and = windowlist & windowlist2
    assert len(windowlist_and) == 2, "WindowList should have two windows after intersection"
    assert windowlist_and[0].image is image1, "First window should be from image1"
    assert windowlist_and[1].image is image2, "Second window should be from image2"
    assert windowlist_and[0].extent == (5, 10, 6, 9), "First window should be the overlap"
