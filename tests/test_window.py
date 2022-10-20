import unittest
from autoprof import image
import numpy as np


class TestWindow(unittest.TestCase):
    def test_window_creation(self):

        window1 = image.AP_Window((0,6), (100,110))

        self.assertEqual(window1.origin[0], 0, "Window should store origin")
        self.assertEqual(window1.origin[1], 6, "Window should store origin")
        self.assertEqual(window1.shape[0], 100, "Window should store shape")
        self.assertEqual(window1.shape[1], 110, "Window should store shape")
        self.assertEqual(window1.center[0], 50., "Window should determine center")
        self.assertEqual(window1.center[1], 61., "Window should determine center")

    def test_window_arithmetic(self):

        windowbig = image.AP_Window((0,0), (100,110))
        windowsmall = image.AP_Window((40,40), (20,30))

        big_or_small = windowbig | windowsmall

        self.assertEqual(big_or_small.origin[0], 0, "logical or of images should take largest bounding box")
        self.assertEqual(big_or_small.shape[0], 100, "logical or of images should take largest bounding box")
        self.assertEqual(windowbig.origin[0], 0, "logical or of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "logical or of images should not affect initial images")
        self.assertEqual(windowsmall.origin[0], 40, "logical or of images should not affect initial images")
        self.assertEqual(windowsmall.shape[0], 20, "logical or of images should not affect initial images")

        big_and_small = windowbig & windowsmall
        
        self.assertEqual(big_and_small.origin[0], 40, "logical and of images should take overlap region")
        self.assertEqual(big_and_small.shape[0], 20, "logical and of images should take overlap region")
        self.assertEqual(big_and_small.shape[1], 30, "logical and of images should take overlap region")
        self.assertEqual(windowbig.origin[0], 0, "logical and of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "logical and of images should not affect initial images")
        self.assertEqual(windowsmall.origin[0], 40, "logical and of images should not affect initial images")
        self.assertEqual(windowsmall.shape[0], 20, "logical and of images should not affect initial images")
        
        windowoffset = image.AP_Window((40,-20), (100,90))
        
        big_or_offset = windowbig | windowoffset
        self.assertEqual(big_or_offset.origin[0], 0, "logical or of images should take largest bounding box")
        self.assertEqual(big_or_offset.origin[1], -20, "logical or of images should take largest bounding box")
        self.assertEqual(big_or_offset.shape[0], 140, "logical or of images should take largest bounding box")
        self.assertEqual(big_or_offset.shape[1], 130, "logical or of images should take largest bounding box")
        self.assertEqual(windowbig.origin[0], 0, "logical or of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "logical or of images should not affect initial images")
        self.assertEqual(windowoffset.origin[0], 40, "logical or of images should not affect initial images")
        self.assertEqual(windowoffset.shape[0], 100, "logical or of images should not affect initial images")
        
        big_and_offset = windowbig & windowoffset

        self.assertEqual(big_and_offset.origin[0], 40, "logical and of images should take overlap region")
        self.assertEqual(big_and_offset.origin[1], 0, "logical and of images should take overlap region")
        self.assertEqual(big_and_offset.shape[0], 60, "logical and of images should take overlap region")
        self.assertEqual(big_and_offset.shape[1], 70, "logical and of images should take overlap region")
        self.assertEqual(windowbig.origin[0], 0, "logical and of images should not affect initial images")
        self.assertEqual(windowbig.shape[0], 100, "logical and of images should not affect initial images")
        self.assertEqual(windowoffset.origin[0], 40, "logical and of images should not affect initial images")
        self.assertEqual(windowoffset.shape[0], 100, "logical and of images should not affect initial images")
        
        windowbig |= windowsmall

        self.assertEqual(windowbig.origin[0], 0, "logical or of images should take largest bounding box")
        self.assertEqual(windowbig.shape[0], 100, "logical or of images should take largest bounding box")
        self.assertEqual(windowsmall.origin[0], 40, "logical or of images should not affect input image")
        self.assertEqual(windowsmall.shape[0], 20, "logical or of images should not affect input image")

        windowbig |= windowoffset

        self.assertEqual(windowbig.origin[0], 0, "logical or of images should take largest bounding box")
        self.assertEqual(windowbig.origin[1], -20, "logical or of images should take largest bounding box")
        self.assertEqual(windowbig.shape[0], 140, "logical or of images should take largest bounding box")
        self.assertEqual(windowbig.shape[1], 130, "logical or of images should take largest bounding box")
        self.assertEqual(windowoffset.origin[0], 40, "logical or of images should not affect input image")
        self.assertEqual(windowoffset.shape[0], 100, "logical or of images should not affect input image")
        
        windowbig = image.AP_Window((0,0), (100,110))

        windowbig &= windowoffset

        self.assertEqual(windowbig.origin[0], 40, "logical and of images should take overlap region")
        self.assertEqual(windowbig.origin[1], 0, "logical and of images should take overlap region")
        self.assertEqual(windowbig.shape[0], 60, "logical and of images should take overlap region")
        self.assertEqual(windowbig.shape[1], 70, "logical and of images should take overlap region")
        self.assertEqual(windowoffset.origin[0], 40, "logical and of images should not affect input image")
        self.assertEqual(windowoffset.shape[0], 100, "logical and of images should not affect input image")

        windowbig &= windowsmall

        self.assertEqual(windowbig, windowsmall, "logical and of images should take overlap region, equality should be internally determined")

    def test_window_buffering(self):

        window = image.AP_Window((0,0), (100,110))

        window_scaled = window * 2
        self.assertEqual(window_scaled.origin[0], -50, "Window scaling should remain centered")
        self.assertEqual(window_scaled.shape[0], 200, "Window scaling should remain centered")
        self.assertEqual(window_scaled.origin[1], -55, "Window scaling should remain centered")
        self.assertEqual(window_scaled.shape[1], 220, "Window scaling should remain centered")
        self.assertEqual(window.origin[0], 0, "Window scaling should not affect initial images")
        self.assertEqual(window.shape[0], 100, "Window scaling should not affect initial images")

        window_buffer = window + 10
        self.assertEqual(window_buffer.origin[0], -10, "Window buffer should remain centered")
        self.assertEqual(window_buffer.shape[0], 120, "Window buffer should remain centered")
        self.assertEqual(window_buffer.origin[1], -10, "Window buffer should remain centered")
        self.assertEqual(window_buffer.shape[1], 130, "Window buffer should remain centered")
        self.assertEqual(window.origin[0], 0, "Window buffering should not affect initial images")
        self.assertEqual(window.shape[0], 100, "Window buffering should not affect initial images")
                
if __name__ == "__main__":
    unittest.main()
        
