import astrophot as ap
import numpy as np


######################################################################
# Window List Object
######################################################################


def test_windowlist_creation():

    image1 = ap.Image(
        data=np.zeros((10, 15)),
        pixelscale=1.0,
        zeropoint=1.0,
        name="image1",
    )
    image2 = ap.Image(
        data=np.ones((15, 10)),
        pixelscale=0.5,
        zeropoint=2.0,
        name="image2",
    )
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
    assert windowlist_and[0].i_low == 5, "First window should have i_low of 5"
    assert windowlist_and[0].i_high == 10, "First window should have i_high of 10"
    assert windowlist_and[0].j_low == 6, "First window should have j_low of 6"
    assert windowlist_and[0].j_high == 9, "First window should have j_high of 9"
