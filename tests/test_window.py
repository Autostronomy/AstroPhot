import unittest
from autoprof import image
import numpy as np


class TestWindow(unittest.TestCase):
    def test_window_creation(self):

        window1 = image.AP_Window((0,0), (100,100))

        self.assertEqual(window1.origin[0], 0, "Window should store origin")
        self.assertEqual(window1.origin[1], 0, "Window should store origin")
        self.assertEqual(window1.shape[0], 100, "Window should store shape")
        self.assertEqual(window1.shape[1], 100, "Window should store shape")
        self.assertEqual(window1.center[0], 50., "Window should determine center")
        self.assertEqual(window1.center[1], 50., "Window should determine center")

    def test_window_arithmetic(self):

        windowbig = image.AP_Window((0,0), (100,100))
        windowsmall = image.AP_Window((40,40), (20,20))

        big_add_small = windowbig + windowsmall

        self.assertEqual(big_add_small.origin[0], 0, "Addition of images should take largest bounding box")
        self.assertEqual(big_add_small.shape[0], 100, "Addition of images should take largest bounding box")
        self.assertEqual(windowbig.origin[0], 0, "Addition of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "Addition of images should not affect initial images")
        self.assertEqual(windowsmall.origin[0], 40, "Addition of images should not affect initial images")
        self.assertEqual(windowsmall.shape[0], 20, "Addition of images should not affect initial images")

        big_mul_small = windowbig * windowsmall
        
        self.assertEqual(big_mul_small.origin[0], 40, "Product of images should take overlap region")
        self.assertEqual(big_mul_small.shape[0], 20, "Product of images should take overlap region")
        self.assertEqual(windowbig.origin[0], 0, "Addition of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "Addition of images should not affect initial images")
        self.assertEqual(windowsmall.origin[0], 40, "Addition of images should not affect initial images")
        self.assertEqual(windowsmall.shape[0], 20, "Addition of images should not affect initial images")
        
        windowoffset = image.AP_Window((40,-20), (100,100))
        
        big_add_offset = windowbig + windowoffset

        self.assertEqual(big_add_offset.origin[0], 0, "Addition of images should take largest bounding box")
        self.assertEqual(big_add_offset.origin[1], -20, "Addition of images should take largest bounding box")
        self.assertEqual(big_add_offset.shape[0], 140, "Addition of images should take largest bounding box")
        self.assertEqual(big_add_offset.shape[1], 120, "Addition of images should take largest bounding box")
        self.assertEqual(windowbig.origin[0], 0, "Addition of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "Addition of images should not affect initial images")
        self.assertEqual(windowoffset.origin[0], 40, "Addition of images should not affect initial images")
        self.assertEqual(windowoffset.shape[0], 100, "Addition of images should not affect initial images")
        
        big_mul_offset = windowbig * windowoffset

        self.assertEqual(big_mul_offset.origin[0], 40, "Product of images should take overlap region")
        self.assertEqual(big_mul_offset.origin[1], 0, "Product of images should take overlap region")
        self.assertEqual(big_mul_offset.shape[0], 60, "Product of images should take overlap region")
        self.assertEqual(big_mul_offset.shape[1], 80, "Product of images should take overlap region")
        self.assertEqual(windowbig.origin[0], 0, "Product of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "Product of images should not affect initial images")
        self.assertEqual(windowoffset.origin[0], 40, "Product of images should not affect initial images")
        self.assertEqual(windowoffset.shape[0], 100, "Product of images should not affect initial images")
        
        windowbig += windowsmall

        self.assertEqual(windowbig.origin[0], 0, "Addition of images should take largest bounding box")
        self.assertEqual(windowbig.shape[0], 100, "Addition of images should take largest bounding box")
        self.assertEqual(windowsmall.origin[0], 40, "Addition of images should not affect input image")
        self.assertEqual(windowsmall.shape[0], 20, "Addition of images should not affect input image")

        windowbig += windowoffset

        self.assertEqual(windowbig.origin[0], 0, "Addition of images should take largest bounding box")
        self.assertEqual(windowbig.origin[1], -20, "Addition of images should take largest bounding box")
        self.assertEqual(windowbig.shape[0], 140, "Addition of images should take largest bounding box")
        self.assertEqual(windowbig.shape[1], 120, "Addition of images should take largest bounding box")
        self.assertEqual(windowoffset.origin[0], 40, "Addition of images should not affect input image")
        self.assertEqual(windowoffset.shape[0], 100, "Addition of images should not affect input image")
        
        windowbig = image.AP_Window((0,0), (100,100))

        windowbig *= windowoffset

        self.assertEqual(windowbig.origin[0], 40, "Product of images should take overlap region")
        self.assertEqual(windowbig.origin[1], 0, "Product of images should take overlap region")
        self.assertEqual(windowbig.shape[0], 60, "Product of images should take overlap region")
        self.assertEqual(windowbig.shape[1], 80, "Product of images should take overlap region")
        self.assertEqual(windowoffset.origin[0], 40, "Product of images should not affect input image")
        self.assertEqual(windowoffset.shape[0], 100, "Product of images should not affect input image")

        windowbig *= windowsmall

        self.assertEqual(windowbig, windowsmall, "Product of images should take overlap region, equality should be internally determined")

    def test_window_buffering(self):

        window = image.AP_Window((0,0), (100,100))

        window_scaled = window.scaled_window(2)
        self.assertEqual(window_scaled.origin[0], -50, "Window scaling should remain centered")
        self.assertEqual(window_scaled.shape[0], 200, "Window scaling should remain centered")
        self.assertEqual(window.origin[0], 0, "Window scaling should not affect initial images")
        self.assertEqual(window.shape[0], 100, "Window scaling should not affect initial images")

        window_buffer = window.buffer_window(10)
        self.assertEqual(window_buffer.origin[0], -10, "Window buffer should remain centered")
        self.assertEqual(window_buffer.shape[0], 120, "Window buffer should remain centered")
        self.assertEqual(window.origin[0], 0, "Window buffering should not affect initial images")
        self.assertEqual(window.shape[0], 100, "Window buffering should not affect initial images")
                
if __name__ == "__main__":
    unittest.main()
        
