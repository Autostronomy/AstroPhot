import unittest
import numpy as np
import torch
import h5py
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
        
class TestConversions(unittest.TestCase):

    def test_conversions_units(self):
        
        #flux to sb
        self.assertEqual(ap.utils.conversions.units.flux_to_sb(1.,1.,0.), 0, "flux incorrectly converted to sb")

        #sb to flux
        self.assertEqual(ap.utils.conversions.units.sb_to_flux(1.,1.,0.), (10**(-1/2.5)), "sb incorrectly converted to flux")
        
        #flux to mag no error
        self.assertEqual(ap.utils.conversions.units.flux_to_mag(1.,0.), 0, "flux incorrectly converted to mag (no error)")

        #flux to mag with error
        self.assertEqual(ap.utils.conversions.units.flux_to_mag(1.,0., fluxe=1.), (0., 2.5/np.log(10)), "flux incorrectly converted to mag (with error)")

        #magperarcsec2 to mag with area A defined
        self.assertEqual(ap.utils.conversions.units.magperarcsec2_to_mag(1., a=None, b=None, A=1.), (1. - 2.5 * np.log10(1.)), "mag/arcsec^2 incorrectly converted to mag (area A given, a and b not defined)")

        #magperarcsec2 to mag with semi major and minor axes defined (a, and b)
        self.assertEqual(ap.utils.conversions.units.magperarcsec2_to_mag(1., a=1., b=1., A=None), (1. - 2.5 * np.log10(np.pi)), "mag/arcsec^2 incorrectly converted to mag (semi major/minor axes defined)")

        #mag to magperarcsec2 with area A defined
        self.assertEqual(ap.utils.conversions.units.mag_to_magperarcsec2(1., a=None, b=None, A=1., R=None), (1. + 2.5 * np.log10(1.)), "mag incorrectly converted to mag/arcsec^2 (area A given)")

        #mag to magperarcsec2 with radius R given (assumes circular)
        self.assertEqual(ap.utils.conversions.units.mag_to_magperarcsec2(1., a=None, b=None, A=None, R=1.), (1. + 2.5 * np.log10(np.pi)), "mag incorrectly converted to mag/arcsec^2 (radius R given)")

        #mag to magperarcsec2 with semi major and minor axes defined (a, and b)
        self.assertEqual(ap.utils.conversions.units.mag_to_magperarcsec2(1., a=1., b=1., A=None, R=None), (1. + 2.5 * np.log10(np.pi)), "mag incorrectly converted to mag/arcsec^2 (area A given)")

        #position angle PA to radians
        self.assertEqual(ap.utils.conversions.units.PA_shift_convention(1., unit='rad'), ((1. - (np.pi / 2)) % np.pi), "PA incorrectly converted to radians")

        #position angle PA to degrees
        self.assertEqual(ap.utils.conversions.units.PA_shift_convention(1., unit='deg'), ((1. - (180 / 2)) % 180), "PA incorrectly converted to degrees")


    def test_conversion_dict_to_hdf5(self):

        #convert string to hdf5
        self.assertEqual(ap.utils.conversions.dict_to_hdf5.to_hdf5_has_None(l='test'), (False), "Failed to properly identify string object while converting to hdf5")

        #convert __iter__ to hdf5
        self.assertEqual(ap.utils.conversions.dict_to_hdf5.to_hdf5_has_None(l='__iter__'), (False), "Attempted to convert '__iter__' to hdf5 key")

        #convert hdf5 file to dict
        h = h5py.File("mytestfile.hdf5", "w")
        dset = h.create_dataset("mydataset", (1,), dtype='i')
        dset[...] = np.array([1.0])
        self.assertEqual(ap.utils.conversions.dict_to_hdf5.hdf5_to_dict(h=h), ({'mydataset': h['mydataset']}), "Failed to convert hdf5 file to dict")

        #convert dict to hdf5
        d = {'mydata1': 'statement', 'mydata2': 'statement2'}
        ap.utils.conversions.dict_to_hdf5.dict_to_hdf5(h=h5py.File('mytestfile2.hdf5','w'),D=d)
        self.assertEqual((list(h5py.File("mytestfile2.hdf5", "r"))), (list(d)), "Failed to convert dict of strings to hdf5")
        
        
if __name__ == "__main__":
    unittest.main()
