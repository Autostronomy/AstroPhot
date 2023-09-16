import unittest
import astrophot as ap
import numpy as np
import torch


######################################################################
# Window List Object
######################################################################
# fixme window list origin/shape definition may need to change


class TestWindowList(unittest.TestCase):
    def test_windowlist_creation(self):

        window1 = ap.image.Window(origin=(0, 6), shape=(100, 110))
        window2 = ap.image.Window(origin=(0, 6), shape=(100, 110))
        windowlist = ap.image.Window_List([window1, window2])

        windowlist.to(dtype=torch.float64, device="cpu")

        # under review
        # self.assertEqual(windowlist.origin[0], 0, "Window should store origin")
        # self.assertEqual(windowlist.origin[1], 6, "Window should store origin")
        # self.assertEqual(windowlist.shape[0], 100, "Window should store shape")
        # self.assertEqual(windowlist.shape[1], 110, "Window should store shape")
        # self.assertEqual(windowlist.center[0], 50., "Window should determine center")
        # self.assertEqual(windowlist.center[1], 61., "Window should determine center")

        self.assertRaises(AssertionError, ap.image.Window_List)

        x = str(windowlist)

    def test_window_arithmetic(self):

        windowbig = ap.image.Window(origin=(0, 0), shape=(100, 110))
        windowsmall = ap.image.Window(origin=(40, 40), shape=(20, 30))
        windowlistbs = ap.image.Window_List([windowbig, windowsmall])
        windowlistbb = ap.image.Window_List([windowbig, windowbig])
        windowlistsb = ap.image.Window_List([windowsmall, windowbig])

        # Logical or, size
        ######################################################################
        big_or_small = windowlistbs | windowlistsb
        
        self.assertEqual(
            big_or_small.origin[0][0],
            0.,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_small.origin[1][0],
            0.,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_small.shape[0][0],
            100,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowbig.origin[0],
            0,
            "logical or of images should not affect initial images",
        )
        self.assertEqual(
            windowbig.shape[0],
            100,
            "logical or of images should not affect initial images",
        )
        self.assertEqual(
            windowsmall.origin[0],
            40,
            "logical or of images should not affect initial images",
        )
        self.assertEqual(
            windowsmall.shape[0],
            20,
            "logical or of images should not affect initial images",
        )

        # Logical and, size
        ######################################################################
        big_and_small = windowlistbs & windowlistsb
        self.assertEqual(
            big_and_small.origin[0][0],
            40,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_small.shape[0][0],
            20,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_small.shape[0][1],
            30,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            windowbig.origin[0],
            0,
            "logical and of images should not affect initial images",
        )
        self.assertEqual(
            windowbig.shape[0],
            100,
            "logical and of images should not affect initial images",
        )
        self.assertEqual(
            windowsmall.origin[0],
            40,
            "logical and of images should not affect initial images",
        )
        self.assertEqual(
            windowsmall.shape[0],
            20,
            "logical and of images should not affect initial images",
        )

        # Logical or, offset
        ######################################################################
        windowoffset = ap.image.Window(origin=(40, -20), shape=(100, 90))
        windowlistoffset = ap.image.Window_List([windowoffset, windowoffset])
        big_or_offset = windowlistbb | windowlistoffset
        self.assertEqual(
            big_or_offset.origin[0][0],
            0,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_offset.origin[1][1],
            -20,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_offset.shape[0][0],
            140,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_offset.shape[1][1],
            130,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowbig.origin[0],
            0,
            "logical or of images should not affect initial images",
        )
        self.assertEqual(
            windowbig.shape[0],
            100,
            "logical or of images should not affect initial images",
        )
        self.assertEqual(
            windowoffset.origin[0],
            40,
            "logical or of images should not affect initial images",
        )
        self.assertEqual(
            windowoffset.shape[0],
            100,
            "logical or of images should not affect initial images",
        )

        # Logical and,  offset
        ######################################################################
        big_and_offset = windowlistbb & windowlistoffset
        self.assertEqual(
            big_and_offset.origin[0][0],
            40,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_offset.origin[0][1],
            0,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_offset.shape[0][0],
            60,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_offset.shape[0][1],
            70,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            windowbig.origin[0],
            0,
            "logical and of images should not affect initial images",
        )
        self.assertEqual(
            windowbig.shape[0],
            100,
            "logical and of images should not affect initial images",
        )
        self.assertEqual(
            windowoffset.origin[0],
            40,
            "logical and of images should not affect initial images",
        )
        self.assertEqual(
            windowoffset.shape[0],
            100,
            "logical and of images should not affect initial images",
        )

        # # Logical ior, size
        # ######################################################################
        # windowbig |= windowsmall
        # self.assertEqual(windowbig.origin[0], 0, "logical or of images should take largest bounding box")
        # self.assertEqual(windowbig.shape[0], 100, "logical or of images should take largest bounding box")
        # self.assertEqual(windowsmall.origin[0], 40, "logical or of images should not affect input image")
        # self.assertEqual(windowsmall.shape[0], 20, "logical or of images should not affect input image")

        # # Logical ior, offset
        # ######################################################################
        # windowbig |= windowoffset
        # self.assertEqual(windowbig.origin[0], 0, "logical or of images should take largest bounding box")
        # self.assertEqual(windowbig.origin[1], -20, "logical or of images should take largest bounding box")
        # self.assertEqual(windowbig.shape[0], 140, "logical or of images should take largest bounding box")
        # self.assertEqual(windowbig.shape[1], 130, "logical or of images should take largest bounding box")
        # self.assertEqual(windowoffset.origin[0], 40, "logical or of images should not affect input image")
        # self.assertEqual(windowoffset.shape[0], 100, "logical or of images should not affect input image")

        # # Logical iand, offset
        # ######################################################################
        # windowbig = ap.image.Window((0,0), (100,110))
        # windowbig &= windowoffset
        # self.assertEqual(windowbig.origin[0], 40, "logical and of images should take overlap region")
        # self.assertEqual(windowbig.origin[1], 0, "logical and of images should take overlap region")
        # self.assertEqual(windowbig.shape[0], 60, "logical and of images should take overlap region")
        # self.assertEqual(windowbig.shape[1], 70, "logical and of images should take overlap region")
        # self.assertEqual(windowoffset.origin[0], 40, "logical and of images should not affect input image")
        # self.assertEqual(windowoffset.shape[0], 100, "logical and of images should not affect input image")

        # windowbig &= windowsmall

        # self.assertEqual(windowbig, windowsmall, "logical and of images should take overlap region, equality should be internally determined")

    def test_windowlist_buffering(self):

        subwindow = ap.image.Window(origin=(0, 0), shape=(100, 110))
        window = ap.image.Window_List([subwindow, subwindow.copy()])

        # Multiply
        ######################################################################
        window_scaled = window * 2
        self.assertEqual(
            window_scaled.origin[0][0], -50, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0][0], 200, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.origin[0][1], -55, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[1][1], 220, "Window scaling should remain centered"
        )
        self.assertEqual(
            window.origin[0][0], 0, "Window scaling should not affect initial images"
        )
        self.assertEqual(
            window.shape[1][0], 100, "Window scaling should not affect initial images"
        )

        # Divide
        ######################################################################
        window_scaled = window / 2
        self.assertEqual(
            window_scaled.origin[0][0], 25, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0][0], 50, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.origin[1][1], 27.5, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0][1], 55, "Window scaling should remain centered"
        )
        self.assertEqual(
            window.origin[1][0], 0, "Window scaling should not affect initial images"
        )
        self.assertEqual(
            window.shape[1][0], 100, "Window scaling should not affect initial images"
        )

        # Add
        ######################################################################
        window_buffer = window + 10
        self.assertEqual(
            window_buffer.origin[1][0], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0][0], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1][1], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1][1], 130, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[1][0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[1][0], 100, "Window buffering should not affect initial images"
        )

        # Subtract
        ######################################################################
        window_buffer = window - 10
        self.assertEqual(
            window_buffer.origin[0][0], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0][0], 80, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[0][1], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0][1], 90, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0][0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0][0], 100, "Window buffering should not affect initial images"
        )

        # iAdd
        ######################################################################
        window_buffer = window.copy()
        window_buffer += 10
        self.assertEqual(
            window_buffer.origin[0][0], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0][0], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1][1], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1][1], 130, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0][0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0][0], 100, "Window buffering should not affect initial images"
        )

        # iSubtract
        ######################################################################
        window_buffer = window.copy()
        window_buffer -= 10
        self.assertEqual(
            window_buffer.origin[1][0], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1][0], 80, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[0][1], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0][1], 90, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0][0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0][0], 100, "Window buffering should not affect initial images"
        )

        self.assertRaises(NotImplementedError, window.shift_origin, torch.tensor([1.0, 1.0]))

    def test_windowlist_logic(self):

        window1 = ap.image.Window(origin=[0.0, 1.0], shape=[10.2, 11.8])
        window2 = ap.image.Window(origin=[0.0, 1.0], shape=[10.2, 11.8])
        window3 = ap.image.Window(origin=[-0.6, 0.4], shape=[15.2, 18.0])
        windowlist1 = ap.image.Window_List([window1, window1.copy()])
        windowlist2 = ap.image.Window_List([window2, window2.copy()])
        windowlist3 = ap.image.Window_List([window3, window3.copy()])

        self.assertEqual(
            windowlist1, windowlist2, "same origin, shape windows should evaluate equal"
        )
        self.assertNotEqual(
            windowlist1, windowlist3, "Differnt windows should not evaluate equal"
        )
        self.assertTrue(
            windowlist3 > windowlist1,
            "Window3 should be identified as larger than window1",
        )
        self.assertTrue(
            windowlist3 >= windowlist1,
            "Window3 should be identified as larger than window1",
        )
        self.assertTrue(
            windowlist1 < windowlist3,
            "Window1 should be identified as smaller than window3",
        )
        self.assertTrue(
            windowlist1 <= windowlist3,
            "Window1 should be identified as smaller than window3",
        )


if __name__ == "__main__":
    unittest.main()
