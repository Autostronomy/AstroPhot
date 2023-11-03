import unittest
import astrophot as ap
import numpy as np
import torch


######################################################################
# Window List Object
######################################################################

class TestWindowList(unittest.TestCase):
    def test_windowlist_creation(self):

        window1 = ap.image.Window(origin=(0, 6), pixel_shape=(100, 110))
        window2 = ap.image.Window(origin=(0, 6), pixel_shape=(100, 110))
        windowlist = ap.image.Window_List([window1, window2])

        windowlist.to(dtype=torch.float64, device="cpu")

        # under review
        self.assertEqual(windowlist.origin[0][0], 0, "Window list should capture origin")
        self.assertEqual(windowlist.origin[1][1], 6, "Window list should capture origin")
        self.assertEqual(windowlist.shape[0][0], 100, "Window list should capture shape")
        self.assertEqual(windowlist.shape[1][1], 110, "Window list should capture shape")
        self.assertEqual(windowlist.center[1][0], 50., "Window should determine center")
        self.assertEqual(windowlist.center[0][1], 61., "Window should determine center")

        x = str(windowlist)
        x = repr(windowlist)

    def test_window_arithmetic(self):

        windowbig = ap.image.Window(origin=(0, 0), pixel_shape=(100, 110))
        windowsmall = ap.image.Window(origin=(40, 40), pixel_shape=(20, 30))
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
        windowoffset = ap.image.Window(origin=(40, -20), pixel_shape=(100, 90))
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

    def test_windowlist_logic(self):

        window1 = ap.image.Window(origin=[0.0, 1.0], pixel_shape=[10.2, 11.8])
        window2 = ap.image.Window(origin=[0.0, 1.0], pixel_shape=[10.2, 11.8])
        window3 = ap.image.Window(origin=[-0.6, 0.4], pixel_shape=[15.2, 18.0])
        windowlist1 = ap.image.Window_List([window1, window1.copy()])
        windowlist2 = ap.image.Window_List([window2, window2.copy()])
        windowlist3 = ap.image.Window_List([window3, window3.copy()])

        self.assertEqual(
            windowlist1, windowlist2, "same origin, shape windows should evaluate equal"
        )
        self.assertNotEqual(
            windowlist1, windowlist3, "Differnt windows should not evaluate equal"
        )

    def test_image_list_errors(self):
        window1 = ap.image.Window(origin=[0.0, 1.0], pixel_shape=[10.2, 11.8])
        window2 = ap.image.Window(origin=[0.0, 1.0], pixel_shape=[10.2, 11.8])
        windowlist1 = ap.image.Window_List([window1, window2])

        # Bad ra dec reference point
        window2 = ap.image.Window(origin=[0.0, 1.0], reference_radec = np.ones(2), pixel_shape=[10.2, 11.8])
        with self.assertRaises(ap.errors.ConflicingWCS):
            test_image = ap.image.Window_List((window1, window2))

        # Bad tangent plane x y reference point
        window2 = ap.image.Window(origin=[0.0, 1.0], reference_planexy = np.ones(2), pixel_shape=[10.2, 11.8])
        with self.assertRaises(ap.errors.ConflicingWCS):
            test_image = ap.image.Window_List((window1, window2))

        # Bad WCS projection
        window2 = ap.image.Window(origin=[0.0, 1.0], projection = "orthographic", pixel_shape=[10.2, 11.8])
        with self.assertRaises(ap.errors.ConflicingWCS):
            test_image = ap.image.Window_List((window1, window2))
        
        
if __name__ == "__main__":
    unittest.main()
