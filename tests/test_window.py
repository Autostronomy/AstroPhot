import unittest
import astrophot as ap
import numpy as np
import torch


class TestWindow(unittest.TestCase):
    def test_window_creation(self):

        window1 = ap.image.Window(origin = (0, 6), pixel_shape = (100, 110))

        window1.to(dtype=torch.float64, device="cpu")

        self.assertEqual(window1.origin[0], 0, "Window should store origin")
        self.assertEqual(window1.origin[1], 6, "Window should store origin")
        self.assertEqual(window1.shape[0], 100, "Window should store shape")
        self.assertEqual(window1.shape[1], 110, "Window should store shape")
        self.assertEqual(window1.center[0], 50.0, "Window should determine center")
        self.assertEqual(window1.center[1], 61.0, "Window should determine center")

        self.assertRaises(Exception, ap.image.Window)

        x = str(window1)
        x = repr(window1)

        wcs = window1.get_astropywcs()

    def test_window_crop(self):
        
        window1 = ap.image.Window(origin = (0, 6), pixel_shape = (100, 110))

        window1.crop_to_pixel([[10,90],[15,105]])
        self.assertTrue(np.all(window1.origin.detach().cpu().numpy() == np.array([10., 21])), "crop pixels should move origin")
        self.assertTrue(np.all(window1.pixel_shape.detach().cpu().numpy() == np.array([80, 90])), "crop pixels should change shape")

        window2 = ap.image.Window(origin = (0, 6), pixel_shape = (100, 110))
        window2.crop_pixel((5,))
        self.assertTrue(np.all(window2.origin.detach().cpu().numpy() == np.array([5., 11.])), "crop pixels should move origin")
        self.assertTrue(np.all(window2.pixel_shape.detach().cpu().numpy() == np.array([90, 100])), "crop pixels should change shape")
        window2.pad_pixel((5,))

        window2 = ap.image.Window(origin = (0, 6), pixel_shape = (100, 110))
        window2.crop_pixel((5,6))
        self.assertTrue(np.all(window2.origin.detach().cpu().numpy() == np.array([5., 12.])), "crop pixels should move origin")
        self.assertTrue(np.all(window2.pixel_shape.detach().cpu().numpy() == np.array([90, 98])), "crop pixels should change shape")
        window2.pad_pixel((5,6))

        window2 = ap.image.Window(origin = (0, 6), pixel_shape = (100, 110))
        window2.crop_pixel((5,6,7,8))
        self.assertTrue(np.all(window2.origin.detach().cpu().numpy() == np.array([5., 12.])), "crop pixels should move origin")
        self.assertTrue(np.all(window2.pixel_shape.detach().cpu().numpy() == np.array([88, 96])), "crop pixels should change shape")
        window2.pad_pixel((5,6,7,8))

        self.assertTrue(np.all(window2.origin.detach().cpu().numpy() == np.array([0., 6.])), "pad pixels should move origin")
        self.assertTrue(np.all(window2.pixel_shape.detach().cpu().numpy() == np.array([100, 110])), "pad pixels should change shape")
        
    def test_window_get_indices(self):

        window1 = ap.image.Window(origin = (0, 6), pixel_shape = (100, 110))
        xstep, ystep = np.meshgrid(range(100), range(110), indexing = "xy")
        zstep = xstep + ystep
        window2 = ap.image.Window(origin = (15, 15), pixel_shape = (30, 200))

        zsliced = zstep[window1.get_self_indices(window2)]
        self.assertTrue(np.all(zsliced == zstep[9:110,15:45]), "window slices should get correct part of image")
        zsliced = zstep[window2.get_other_indices(window1)]
        self.assertTrue(np.all(zsliced == zstep[9:110,15:45]), "window slices should get correct part of image")

    def test_window_arithmetic(self):

        windowbig = ap.image.Window(origin = (0, 0), pixel_shape = (100, 110))
        windowsmall = ap.image.Window(origin = (40, 40), pixel_shape = (20, 30))

        # Logical or, size
        ######################################################################
        big_or_small = windowbig | windowsmall
        self.assertEqual(
            big_or_small.origin[0],
            0,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_small.shape[0],
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
        big_and_small = windowbig & windowsmall
        self.assertEqual(
            big_and_small.origin[0],
            40,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_small.shape[0],
            20,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_small.shape[1],
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
        windowoffset = ap.image.Window(origin = (40, -20), pixel_shape = (100, 90))
        big_or_offset = windowbig | windowoffset
        self.assertEqual(
            big_or_offset.origin[0],
            0,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_offset.origin[1],
            -20,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_offset.shape[0],
            140,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            big_or_offset.shape[1],
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
        big_and_offset = windowbig & windowoffset
        self.assertEqual(
            big_and_offset.origin[0],
            40,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_offset.origin[1],
            0,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_offset.shape[0],
            60,
            "logical and of images should take overlap region",
        )
        self.assertEqual(
            big_and_offset.shape[1],
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

        # Logical ior, size
        ######################################################################
        windowbig |= windowsmall
        self.assertEqual(
            windowbig.origin[0],
            0,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowbig.shape[0],
            100,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowsmall.origin[0],
            40,
            "logical or of images should not affect input image",
        )
        self.assertEqual(
            windowsmall.shape[0],
            20,
            "logical or of images should not affect input image",
        )

        # Logical ior, offset
        ######################################################################
        windowbig |= windowoffset
        self.assertEqual(
            windowbig.origin[0],
            0,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowbig.origin[1],
            -20,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowbig.shape[0],
            140,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowbig.shape[1],
            130,
            "logical or of images should take largest bounding box",
        )
        self.assertEqual(
            windowoffset.origin[0],
            40,
            "logical or of images should not affect input image",
        )
        self.assertEqual(
            windowoffset.shape[0],
            100,
            "logical or of images should not affect input image",
        )

        # Logical iand, offset
        ######################################################################
        windowbig = ap.image.Window(origin = (0, 0), pixel_shape = (100, 110))
        windowbig &= windowoffset
        self.assertEqual(
            windowbig.origin[0], 40, "logical and of images should take overlap region"
        )
        self.assertEqual(
            windowbig.origin[1], 0, "logical and of images should take overlap region"
        )
        self.assertEqual(
            windowbig.shape[0], 60, "logical and of images should take overlap region"
        )
        self.assertEqual(
            windowbig.shape[1], 70, "logical and of images should take overlap region"
        )
        self.assertEqual(
            windowoffset.origin[0],
            40,
            "logical and of images should not affect input image",
        )
        self.assertEqual(
            windowoffset.shape[0],
            100,
            "logical and of images should not affect input image",
        )

        windowbig &= windowsmall

        self.assertEqual(
            windowbig,
            windowsmall,
            "logical and of images should take overlap region, equality should be internally determined",
        )

    def test_window_state(self):
        window_init = ap.image.Window(origin = [1.0, 2.0], pixel_shape = [10, 15], pixelscale = 1, projection = "orthographic", reference_radec = (0,0))
        window = ap.image.Window(state=window_init.get_state())
        self.assertEqual(
            window.origin[0].item(), 1.0, "Window initialization should read state"
        )
        self.assertEqual(
            window.shape[0].item(), 10.0, "Window initialization should read state"
        )
        self.assertEqual(
            window.pixelscale[0][0].item(), 1.0, "Window initialization should read state"
        )

        state = window.get_state()
        self.assertEqual(
            state["reference_imagexy"][1], 2.0, "Window get state should collect values"
        )
        self.assertEqual(
            state["pixel_shape"][1], 15.0, "Window get state should collect values"
        )
        self.assertEqual(
            state["pixelscale"][1][0], 0.0, "Window get state should collect values"
        )
        self.assertEqual(
            state["projection"], "orthographic", "Window get state should collect values"
        )
        self.assertEqual(
            tuple(state["reference_radec"]), (0.,0.), "Window get state should collect values"
        )

    def test_window_logic(self):

        window1 = ap.image.Window(origin=[0.0, 1.0],  pixel_shape=[10., 11.])
        window2 = ap.image.Window(origin=[0.0, 1.0],  pixel_shape=[10., 11.])
        window3 = ap.image.Window(origin=[-0.6, 0.4], pixel_shape=[15., 18.])

        self.assertEqual(
            window1, window2, "same origin, shape windows should evaluate equal"
        )
        self.assertNotEqual(
            window1, window3, "Differnt windows should not evaluate equal"
        )

    def test_window_errors(self):

        # Initialize with conflicting information
        with self.assertRaises(ap.errors.SpecificationConflict):
            window = ap.image.Window(origin=[0.0, 1.0], origin_radec=[5.,6.],  pixel_shape=[10., 11.])

        

if __name__ == "__main__":
    unittest.main()
