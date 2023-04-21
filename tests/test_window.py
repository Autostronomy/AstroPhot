import unittest
import autoprof as ap
import numpy as np
import torch


class TestWindow(unittest.TestCase):
    def test_window_creation(self):

        window1 = ap.image.Window((0, 6), (100, 110))

        window1.to(dtype=torch.float64, device="cpu")

        self.assertEqual(window1.origin[0], 0, "Window should store origin")
        self.assertEqual(window1.origin[1], 6, "Window should store origin")
        self.assertEqual(window1.shape[0], 100, "Window should store shape")
        self.assertEqual(window1.shape[1], 110, "Window should store shape")
        self.assertEqual(window1.center[0], 50.0, "Window should determine center")
        self.assertEqual(window1.center[1], 61.0, "Window should determine center")

        self.assertRaises(ValueError, ap.image.Window)

        shape = window1.get_shape(torch.tensor(10.0))
        self.assertEqual(
            shape[0].item(), 10, "Window shape in pixels should divide by pixelscale"
        )
        shape = window1.get_shape_flip(torch.tensor(5.0))
        self.assertEqual(
            shape[0].item(), 22, "Window shape in pixels should divide by pixelscale"
        )

        x = str(window1)

    def test_window_arithmetic(self):

        windowbig = ap.image.Window((0, 0), (100, 110))
        windowsmall = ap.image.Window((40, 40), (20, 30))

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
        windowoffset = ap.image.Window((40, -20), (100, 90))
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
        windowbig = ap.image.Window((0, 0), (100, 110))
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

    def test_window_buffering(self):

        window = ap.image.Window((0, 0), (100, 110))

        # Multiply
        ######################################################################
        window_scaled = window * 2
        self.assertEqual(
            window_scaled.origin[0], -50, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0], 200, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.origin[1], -55, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[1], 220, "Window scaling should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window scaling should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window scaling should not affect initial images"
        )
        window_scaled = window * (2, 1)
        self.assertEqual(
            window_scaled.origin[0], -50, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0], 200, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.origin[1], 0, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[1], 110, "Window scaling should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window scaling should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window scaling should not affect initial images"
        )

        # Divide
        ######################################################################
        window_scaled = window / 2
        self.assertEqual(
            window_scaled.origin[0], 25, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0], 50, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.origin[1], 27.5, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[1], 55, "Window scaling should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window scaling should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window scaling should not affect initial images"
        )
        window_scaled = window / (2, 1)
        self.assertEqual(
            window_scaled.origin[0], 25, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[0], 50, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.origin[1], 0, "Window scaling should remain centered"
        )
        self.assertEqual(
            window_scaled.shape[1], 110, "Window scaling should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window scaling should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window scaling should not affect initial images"
        )

        # Add
        ######################################################################
        window_buffer = window + 10
        self.assertEqual(
            window_buffer.origin[0], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 130, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )
        window_buffer = window + (10.0, 5.0)
        self.assertEqual(
            window_buffer.origin[0], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], -5, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )

        # Subtract
        ######################################################################
        window_buffer = window - 10
        self.assertEqual(
            window_buffer.origin[0], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 80, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 90, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )
        window_buffer = window - (10.0, 5.0)
        self.assertEqual(
            window_buffer.origin[0], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 80, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], 5, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 100, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )

        # iAdd
        ######################################################################
        window_buffer = window.copy()
        window_buffer += 10
        self.assertEqual(
            window_buffer.origin[0], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 130, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )
        window_buffer = window.copy()
        window_buffer += (10.0, 5.0)
        self.assertEqual(
            window_buffer.origin[0], -10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], -5, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 120, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )

        # iSubtract
        ######################################################################
        window_buffer = window.copy()
        window_buffer -= 10
        self.assertEqual(
            window_buffer.origin[0], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 80, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 90, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )
        window_buffer = window.copy()
        window_buffer -= (10.0, 5.0)
        self.assertEqual(
            window_buffer.origin[0], 10, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[0], 80, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.origin[1], 5, "Window buffer should remain centered"
        )
        self.assertEqual(
            window_buffer.shape[1], 100, "Window buffer should remain centered"
        )
        self.assertEqual(
            window.origin[0], 0, "Window buffering should not affect initial images"
        )
        self.assertEqual(
            window.shape[0], 100, "Window buffering should not affect initial images"
        )

        window.shift_origin(torch.tensor(1.0))
        self.assertEqual(window.origin[0].item(), 1.0, "Origin should be moved")

    def test_window_state(self):

        window = ap.image.Window(state={"origin": [1.0, 2.0], "shape": [10, 15]})
        self.assertEqual(
            window.origin[0].item(), 1.0, "Window initialization should read state"
        )
        self.assertEqual(
            window.shape[0].item(), 10.0, "Window initialization should read state"
        )

        state = window.get_state()
        self.assertEqual(
            state["origin"][1], 2.0, "Window get state should collect values"
        )
        self.assertEqual(
            state["shape"][1], 15.0, "Window get state should collect values"
        )

    def test_window_logic(self):

        window1 = ap.image.Window(origin=[0.0, 1.0], shape=[10.2, 11.8])
        window2 = ap.image.Window(origin=[0.0, 1.0], shape=[10.2, 11.8])
        window3 = ap.image.Window(origin=[-0.6, 0.4], shape=[15.2, 18.0])

        self.assertEqual(
            window1, window2, "same origin, shape windows should evaluate equal"
        )
        self.assertNotEqual(
            window1, window3, "Differnt windows should not evaluate equal"
        )
        self.assertTrue(
            window3 > window1, "Window3 should be identified as larger than window1"
        )
        self.assertTrue(
            window3 >= window1, "Window3 should be identified as larger than window1"
        )
        self.assertTrue(
            window1 < window3, "Window1 should be identified as smaller than window3"
        )
        self.assertTrue(
            window1 <= window3, "Window1 should be identified as smaller than window3"
        )


if __name__ == "__main__":
    unittest.main()
