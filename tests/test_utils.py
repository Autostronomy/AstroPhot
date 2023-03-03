import unittest
import numpy as np
import torch
from scipy.signal import fftconvolve
import autoprof as ap
from utils import make_basic_sersic, make_basic_gaussian
######################################################################
# Util functions
######################################################################

class TestFFT(unittest.TestCase):

    def test_fft(self):
        
        target = make_basic_sersic()

        convolved = ap.utils.operations.fft_convolve_torch(
            target.data, target.psf,
        )
        scipy_convolve = fftconvolve(
            target.data.detach().numpy(),
            target.psf.detach().numpy(),
            mode = "same",
        )
        self.assertLess(
            torch.std(convolved), torch.std(target.data), "Convolved image should be smoothed"
        )

        self.assertTrue(np.all(np.isclose(convolved.detach().numpy(), scipy_convolve)), "Should reproduce scipy convolve")

    def test_fft_multi(self):

        target = make_basic_sersic()

        convolved = ap.utils.operations.fft_convolve_multi_torch(target.data, [target.psf, target.psf])
        self.assertLess(
            torch.std(convolved), torch.std(target.data), "Convolved image should be smoothed"
        )

class TestOptimize(unittest.TestCase):

    def test_chi2(self):

        # with variance
        # with mask
        mask = torch.zeros(10,dtype = torch.bool)
        mask[2] = 1
        chi2 = ap.utils.optimization.chi_squared(torch.ones(10), torch.zeros(10), mask = mask, variance = 2*torch.ones(10))
        self.assertEqual(
            chi2, 4.5, "Chi squared calculation incorrect"
        )
        chi2_red = ap.utils.optimization.reduced_chi_squared(torch.ones(10), torch.zeros(10), params = 3, mask = mask, variance = 2*torch.ones(10))
        self.assertEqual(
            chi2_red, 0.75, "Chi squared calculation incorrect"
        )

        # no mask
        chi2 = ap.utils.optimization.chi_squared(torch.ones(10), torch.zeros(10), variance = 2*torch.ones(10))
        self.assertEqual(
            chi2, 5, "Chi squared calculation incorrect"
        )
        chi2_red = ap.utils.optimization.reduced_chi_squared(torch.ones(10), torch.zeros(10), params = 3, variance = 2*torch.ones(10))
        self.assertEqual(
            chi2_red, 5/7, "Chi squared calculation incorrect"
        )

        # no variance
        # with mask
        mask = torch.zeros(10,dtype = torch.bool)
        mask[2] = 1
        chi2 = ap.utils.optimization.chi_squared(torch.ones(10), torch.zeros(10), mask = mask)
        self.assertEqual(
            chi2, 9, "Chi squared calculation incorrect"
        )
        chi2_red = ap.utils.optimization.reduced_chi_squared(torch.ones(10), torch.zeros(10), params = 3, mask = mask)
        self.assertEqual(
            chi2_red, 1.5, "Chi squared calculation incorrect"
        )

        # no mask
        chi2 = ap.utils.optimization.chi_squared(torch.ones(10), torch.zeros(10))
        self.assertEqual(
            chi2, 10, "Chi squared calculation incorrect"
        )
        chi2_red = ap.utils.optimization.reduced_chi_squared(torch.ones(10), torch.zeros(10), params = 3)
        self.assertEqual(
            chi2_red, 10/7, "Chi squared calculation incorrect"
        )

class TestPSF(unittest.TestCase):

    def test_make_psf(self):
        
        target = make_basic_gaussian(x = 10, y = 10)
        target += make_basic_gaussian(x = 40, y = 40, rand = 54321)

        psf = ap.utils.initialize.construct_psf([[10,10], [40,40]], target.data.detach().cpu().numpy(), sky_est = 0., size = 5)

        self.assertTrue(np.all(np.isfinite(psf)))

class TestSegtoWindow(unittest.TestCase):

    def test_segtowindow(self):

        segmap = np.zeros((100,100), dtype = int)

        segmap[5:9,20:30] = 1
        segmap[50:90,17:35] = 2
        segmap[26:34,80:85] = 3

        windows = ap.utils.initialize.windows_from_segmentation_map(segmap)

        self.assertEqual(len(windows), 3, "should ignore zero index, but find all three windows")
        
        self.assertEqual(windows[1], [[20,29],[5,8]], "Windows should be identified by index")

        # scale windows

        new_windows = ap.utils.initialize.scale_windows(windows, image_shape = (100,100), expand_scale = 2, expand_border = 3)

        self.assertEqual(new_windows[2], [[5,45],[27,100]], "Windows should scale appropriately")

        filtered_windows = ap.utils.initialize.filter_windows(new_windows, min_size = 10, max_size = 80, min_area = 30, max_area = 1000)
        filtered_windows = ap.utils.initialize.filter_windows(new_windows, min_flux = 10, max_flux = 1000, image = np.ones(segmap.shape))

        self.assertEqual(len(filtered_windows), 2, "windows should have been filtered")

        # check original
        self.assertEqual(windows[3], [[80,84],[26,33]], "Original windows should not have changed")
        
        
if __name__ == "__main__":
    unittest.main()
