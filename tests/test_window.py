import astrophot as ap
import numpy as np


def test_window_creation():

    image = ap.Image(
        data=np.zeros((100, 110)),
        pixelscale=0.3,
        zeropoint=1.0,
        name="test_image",
    )
    window = ap.Window((2, 107, 3, 97), image)

    assert np.all(window.crpix == image.crpix), "Window should inherit crpix from image"
    assert window.identity == image.identity, "Window should inherit identity from image"
    assert window.shape == (105, 94), "Window should have correct shape"
    assert window.extent == (2, 107, 3, 97), "Window should have correct extent"
    assert str(window) == "Window(2, 107, 3, 97)", "String representation should match"


def test_window_chunk():

    image = ap.Image(
        data=np.zeros((100, 110)),
        pixelscale=0.3,
        zeropoint=1.0,
        name="test_image",
    )
    window1 = ap.Window((2, 107, 3, 97), image)

    subwindows = window1.chunk(10**2)
    reconstitute = subwindows[0]
    for subwindow in subwindows:
        reconstitute |= subwindow
    assert (
        reconstitute.i_low == window1.i_low
    ), "chunked windows should reconstitute to original window"
    assert (
        reconstitute.i_high == window1.i_high
    ), "chunked windows should reconstitute to original window"
    assert (
        reconstitute.j_low == window1.j_low
    ), "chunked windows should reconstitute to original window"
    assert (
        reconstitute.j_high == window1.j_high
    ), "chunked windows should reconstitute to original window"


def test_window_arithmetic():

    image = ap.Image(
        data=np.zeros((100, 110)),
        pixelscale=0.3,
        zeropoint=1.0,
        name="test_image",
    )
    windowbig = ap.Window((2, 107, 3, 97), image)
    windowsmall = ap.Window((20, 45, 30, 90), image)

    # Logical or, size
    ######################################################################
    big_or_small = windowbig | windowsmall
    assert big_or_small.i_low == 2, "logical or of images should take largest bounding box"
    assert big_or_small.i_high == 107, "logical or of images should take largest bounding box"
    assert big_or_small.j_low == 3, "logical or of images should take largest bounding box"
    assert big_or_small.j_high == 97, "logical or of images should take largest bounding box"

    # Logical and, size
    ######################################################################
    big_and_small = windowbig & windowsmall
    assert big_and_small.i_low == 20, "logical and of images should take overlap region"
    assert big_and_small.i_high == 45, "logical and of images should take overlap region"
    assert big_and_small.j_low == 30, "logical and of images should take overlap region"
    assert big_and_small.j_high == 90, "logical and of images should take overlap region"
