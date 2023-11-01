import unittest
import numpy as np
import torch
import h5py
from scipy.signal import fftconvolve
from scipy.special import gamma
from torch.special import gammaln
from scipy.interpolate import RectBivariateSpline
import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian

######################################################################
# Util functions
######################################################################


class TestFFT(unittest.TestCase):
    def test_fft(self):

        target = make_basic_sersic()

        convolved = ap.utils.operations.fft_convolve_torch(
            target.data,
            target.psf.data,
        )
        scipy_convolve = fftconvolve(
            target.data.detach().cpu().numpy(),
            target.psf.data.detach().cpu().numpy(),
            mode="same",
        )
        self.assertLess(
            torch.std(convolved),
            torch.std(target.data),
            "Convolved image should be smoothed",
        )

        self.assertTrue(
            np.all(np.isclose(convolved.detach().cpu().numpy(), scipy_convolve)),
            "Should reproduce scipy convolve",
        )

    def test_fft_multi(self):

        target = make_basic_sersic()

        convolved = ap.utils.operations.fft_convolve_multi_torch(
            target.data, [target.psf.data, target.psf.data]
        )
        self.assertLess(
            torch.std(convolved),
            torch.std(target.data),
            "Convolved image should be smoothed",
        )


class TestOptimize(unittest.TestCase):
    def test_chi2(self):

        # with variance
        # with mask
        mask = torch.zeros(10, dtype=torch.bool, device = ap.AP_config.ap_device)
        mask[2] = 1
        chi2 = ap.utils.optimization.chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            mask=mask, variance=2 * torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)
        )
        self.assertEqual(chi2, 4.5, "Chi squared calculation incorrect")
        chi2_red = ap.utils.optimization.reduced_chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            params=3,
            mask=mask,
            variance=2 * torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
        )
        self.assertEqual(chi2_red.item(), 0.75, "Chi squared calculation incorrect")

        # no mask
        chi2 = ap.utils.optimization.chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            variance=2 * torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
        )
        self.assertEqual(chi2, 5, "Chi squared calculation incorrect")
        chi2_red = ap.utils.optimization.reduced_chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            params=3, variance=2 * torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)
        )
        self.assertEqual(chi2_red.item(), 5 / 7, "Chi squared calculation incorrect")

        # no variance
        # with mask
        mask = torch.zeros(10, dtype=torch.bool, device = ap.AP_config.ap_device)
        mask[2] = 1
        chi2 = ap.utils.optimization.chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            mask=mask
        )
        self.assertEqual(chi2.item(), 9, "Chi squared calculation incorrect")
        chi2_red = ap.utils.optimization.reduced_chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            params=3, mask=mask
        )
        self.assertEqual(chi2_red.item(), 1.5, "Chi squared calculation incorrect")

        # no mask
        chi2 = ap.utils.optimization.chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)
        )
        self.assertEqual(chi2.item(), 10, "Chi squared calculation incorrect")
        chi2_red = ap.utils.optimization.reduced_chi_squared(
            torch.ones(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            torch.zeros(10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device),
            params=3
        )
        self.assertEqual(chi2_red.item(), 10 / 7, "Chi squared calculation incorrect")


class TestPSF(unittest.TestCase):
    def test_make_psf(self):

        target = make_basic_gaussian(x=10, y=10)
        target += make_basic_gaussian(x=40, y=40, rand=54321)

        psf = ap.utils.initialize.construct_psf(
            [[10, 10], [40, 40]],
            target.data.detach().cpu().numpy(),
            sky_est=0.0,
            size=5,
        )

        self.assertTrue(np.all(np.isfinite(psf)))


class TestSegtoWindow(unittest.TestCase):
    def test_segtowindow(self):

        segmap = np.zeros((100, 100), dtype=int)

        segmap[5:9, 20:30] = 1
        segmap[50:90, 17:35] = 2
        segmap[26:34, 80:85] = 3

        centroids = ap.utils.initialize.centroids_from_segmentation_map(
            segmap, image=segmap
        )

        PAs = ap.utils.initialize.PA_from_segmentation_map(
            segmap, image=segmap, centroids = centroids,
        )
        qs = ap.utils.initialize.q_from_segmentation_map(
            segmap, image=segmap, centroids = centroids,
        )

        windows = ap.utils.initialize.windows_from_segmentation_map(segmap)

        self.assertEqual(
            len(windows), 3, "should ignore zero index, but find all three windows"
        )
        self.assertEqual(
            len(centroids), 3, "should ignore zero index, but find all three windows"
        )
        self.assertEqual(
            len(PAs), 3, "should ignore zero index, but find all three windows"
        )
        self.assertEqual(
            len(qs), 3, "should ignore zero index, but find all three windows"
        )

        self.assertEqual(
            windows[1], [[20, 29], [5, 8]], "Windows should be identified by index"
        )

        # scale windows

        new_windows = ap.utils.initialize.scale_windows(
            windows, image_shape=(100, 100), expand_scale=2, expand_border=3
        )

        self.assertEqual(
            new_windows[2], [[5, 45], [27, 100]], "Windows should scale appropriately"
        )

        filtered_windows = ap.utils.initialize.filter_windows(
            new_windows, min_size=10, max_size=80, min_area=30, max_area=1000
        )
        filtered_windows = ap.utils.initialize.filter_windows(
            new_windows, min_flux=10, max_flux=1000, image=np.ones(segmap.shape)
        )

        self.assertEqual(len(filtered_windows), 2, "windows should have been filtered")

        # check original
        self.assertEqual(
            windows[3], [[80, 84], [26, 33]], "Original windows should not have changed"
        )


class TestConversions(unittest.TestCase):
    def test_conversions_units(self):

        # flux to sb
        self.assertEqual(
            ap.utils.conversions.units.flux_to_sb(1.0, 1.0, 0.0),
            0,
            "flux incorrectly converted to sb",
        )

        # sb to flux
        self.assertEqual(
            ap.utils.conversions.units.sb_to_flux(1.0, 1.0, 0.0),
            (10 ** (-1 / 2.5)),
            "sb incorrectly converted to flux",
        )

        # flux to mag no error
        self.assertEqual(
            ap.utils.conversions.units.flux_to_mag(1.0, 0.0),
            0,
            "flux incorrectly converted to mag (no error)",
        )

        # flux to mag with error
        self.assertEqual(
            ap.utils.conversions.units.flux_to_mag(1.0, 0.0, fluxe=1.0),
            (0.0, 2.5 / np.log(10)),
            "flux incorrectly converted to mag (with error)",
        )

        # mag to flux no error:
        self.assertEqual(
            ap.utils.conversions.units.mag_to_flux(1.0, 0.0, mage=None),
            (10 ** (-1 / 2.5)),
            "mag incorrectly converted to flux (no error)",
        )

        # mag to flux with error:
        [
            self.assertAlmostEqual(
                ap.utils.conversions.units.mag_to_flux(1.0, 0.0, mage=1.0)[i],
                (10 ** (-1.0 / 2.5), np.log(10) * (1.0 / 2.5) * 10 ** (-1.0 / 2.5))[i],
                msg="mag incorrectly converted to flux (with error)",
            )
            for i in range(1)
        ]

        # magperarcsec2 to mag with area A defined
        self.assertAlmostEqual(
            ap.utils.conversions.units.magperarcsec2_to_mag(1.0, a=None, b=None, A=1.0),
            (1.0 - 2.5 * np.log10(1.0)),
            msg="mag/arcsec^2 incorrectly converted to mag (area A given, a and b not defined)",
        )

        # magperarcsec2 to mag with semi major and minor axes defined (a, and b)
        self.assertAlmostEqual(
            ap.utils.conversions.units.magperarcsec2_to_mag(1.0, a=1.0, b=1.0, A=None),
            (1.0 - 2.5 * np.log10(np.pi)),
            msg="mag/arcsec^2 incorrectly converted to mag (semi major/minor axes defined)",
        )

        # mag to magperarcsec2 with area A defined
        self.assertAlmostEqual(
            ap.utils.conversions.units.mag_to_magperarcsec2(
                1.0, a=None, b=None, A=1.0, R=None
            ),
            (1.0 + 2.5 * np.log10(1.0)),
            msg="mag incorrectly converted to mag/arcsec^2 (area A given)",
        )

        # mag to magperarcsec2 with radius R given (assumes circular)
        self.assertAlmostEqual(
            ap.utils.conversions.units.mag_to_magperarcsec2(
                1.0, a=None, b=None, A=None, R=1.0
            ),
            (1.0 + 2.5 * np.log10(np.pi)),
            msg="mag incorrectly converted to mag/arcsec^2 (radius R given)",
        )

        # mag to magperarcsec2 with semi major and minor axes defined (a, and b)
        self.assertAlmostEqual(
            ap.utils.conversions.units.mag_to_magperarcsec2(
                1.0, a=1.0, b=1.0, A=None, R=None
            ),
            (1.0 + 2.5 * np.log10(np.pi)),
            msg="mag incorrectly converted to mag/arcsec^2 (area A given)",
        )

        # position angle PA to radians
        self.assertAlmostEqual(
            ap.utils.conversions.units.PA_shift_convention(1.0, unit="rad"),
            ((1.0 - (np.pi / 2)) % np.pi),
            msg="PA incorrectly converted to radians",
        )

        # position angle PA to degrees
        self.assertAlmostEqual(
            ap.utils.conversions.units.PA_shift_convention(1.0, unit="deg"),
            ((1.0 - (180 / 2)) % 180),
            msg="PA incorrectly converted to degrees",
        )

    def test_conversion_dict_to_hdf5(self):

        # convert string to hdf5
        self.assertEqual(
            ap.utils.conversions.dict_to_hdf5.to_hdf5_has_None(l="test"),
            (False),
            "Failed to properly identify string object while converting to hdf5",
        )

        # convert __iter__ to hdf5
        self.assertEqual(
            ap.utils.conversions.dict_to_hdf5.to_hdf5_has_None(l="__iter__"),
            (False),
            "Attempted to convert '__iter__' to hdf5 key",
        )

        # convert hdf5 file to dict
        h = h5py.File("mytestfile.hdf5", "w")
        dset = h.create_dataset("mydataset", (1,), dtype="i")
        dset[...] = np.array([1.0])
        self.assertEqual(
            ap.utils.conversions.dict_to_hdf5.hdf5_to_dict(h=h),
            ({"mydataset": h["mydataset"]}),
            "Failed to convert hdf5 file to dict",
        )

        # convert dict to hdf5
        target = make_basic_sersic().data.detach().cpu().numpy()[0]
        d = {"sersic": target.tolist()}
        ap.utils.conversions.dict_to_hdf5.dict_to_hdf5(
            h=h5py.File("mytestfile2.hdf5", "w"), D=d
        )
        self.assertEqual(
            (list(h5py.File("mytestfile2.hdf5", "r"))),
            (list(d)),
            "Failed to convert dict of strings to hdf5",
        )

    def test_conversion_functions(self):

        sersic_n = ap.utils.conversions.functions.sersic_n_to_b(1.0)
        # sersic I0 to flux - numpy
        self.assertAlmostEqual(
            ap.utils.conversions.functions.sersic_I0_to_flux_np(1.0, 1.0, 1.0, 1.0),
            (2 * np.pi * gamma(2)),
            msg="Error converting sersic central intensity to flux (np)",
        )

        # sersic flux to I0 - numpy
        self.assertAlmostEqual(
            ap.utils.conversions.functions.sersic_flux_to_I0_np(1.0, 1.0, 1.0, 1.0),
            (1.0 / (2 * np.pi * gamma(2))),
            msg="Error converting sersic flux to central intensity (np)",
        )

        # sersic Ie to flux - numpy
        self.assertAlmostEqual(
            ap.utils.conversions.functions.sersic_Ie_to_flux_np(1.0, 1.0, 1.0, 1.0),
            (2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2)),
            msg="Error converting sersic effective intensity to flux (np)",
        )

        # sersic flux to Ie - numpy
        self.assertAlmostEqual(
            ap.utils.conversions.functions.sersic_flux_to_Ie_np(1.0, 1.0, 1.0, 1.0),
            (1 / (2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2))),
            msg="Error converting sersic flux to effective intensity (np)",
        )

        # inverse sersic - numpy
        self.assertAlmostEqual(
            ap.utils.conversions.functions.sersic_inv_np(1.0, 1.0, 1.0, 1.0),
            (1.0 - (1.0 / sersic_n) * np.log(1.0)),
            msg="Error computing inverse sersic function (np)",
        )

        # sersic I0 to flux - torch
        tv = torch.tensor([[1.0]], dtype=torch.float64)
        self.assertEqual(
            torch.round(
                ap.utils.conversions.functions.sersic_I0_to_flux_np(tv, tv, tv, tv),
                decimals=7,
            ),
            torch.round(torch.tensor([[2 * np.pi * gamma(2)]]), decimals=7),
            msg="Error converting sersic central intensity to flux (torch)",
        )

        # sersic flux to I0 - torch
        self.assertEqual(
            torch.round(
                ap.utils.conversions.functions.sersic_flux_to_I0_np(tv, tv, tv, tv),
                decimals=7,
            ),
            torch.round(torch.tensor([[1.0 / (2 * np.pi * gamma(2))]]), decimals=7),
            msg="Error converting sersic flux to central intensity (torch)",
        )

        # sersic Ie to flux - torch
        self.assertEqual(
            torch.round(
                ap.utils.conversions.functions.sersic_Ie_to_flux_np(tv, tv, tv, tv),
                decimals=7,
            ),
            torch.round(
                torch.tensor(
                    [[2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2)]]
                ),
                decimals=7,
            ),
            msg="Error converting sersic effective intensity to flux (torch)",
        )

        # sersic flux to Ie - torch
        self.assertEqual(
            torch.round(
                ap.utils.conversions.functions.sersic_flux_to_Ie_np(tv, tv, tv, tv),
                decimals=7,
            ),
            torch.round(
                torch.tensor(
                    [[1 / (2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2))]]
                ),
                decimals=7,
            ),
            msg="Error converting sersic flux to effective intensity (torch)",
        )

        # inverse sersic - torch
        self.assertEqual(
            torch.round(
                ap.utils.conversions.functions.sersic_inv_np(tv, tv, tv, tv), decimals=7
            ),
            torch.round(
                torch.tensor([[1.0 - (1.0 / sersic_n) * np.log(1.0)]]), decimals=7
            ),
            msg="Error computing inverse sersic function (torch)",
        )
    def test_general_derivative(self):

        res = ap.utils.conversions.functions.general_uncertainty_prop(
            tuple(torch.tensor(a) for a in (1.0, 1.0, 1.0, 0.5)),
            tuple(torch.tensor(a) for a in (0.1, 0.1, 0.1, 0.1)),
            ap.utils.conversions.functions.sersic_Ie_to_flux_torch,
        )

        self.assertAlmostEqual(res.detach().cpu().numpy(), 1.8105, 3, "General uncertianty prop should compute uncertainty")


class TestInterpolate(unittest.TestCase):
    def test_interpolate_functions(self):

        # Lanczos kernel interpolation on the center point of a gaussian (10., 10.)
        model = make_basic_gaussian(x=10.0, y=10.0).data.detach().cpu().numpy()
        lanczos_interp = ap.utils.interpolate.point_Lanczos(
            model, 10.0, 10.0, scale=0.8
        )
        self.assertTrue(
            np.all(np.isfinite(model)), msg="gaussian model returning nonfinite values"
        )
        self.assertLess(
            lanczos_interp, 1.0, msg="Lanczos interpolation greater than total flux"
        )
        self.assertTrue(
            np.isfinite(lanczos_interp),
            msg="Lanczos interpolate returning nonfinite values",
        )


class TestAngleOperations(unittest.TestCase):
    def test_angle_operation_functions(self):

        test_angles = np.array([np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        # angle median
        self.assertAlmostEqual(
            ap.utils.angle_operations.Angle_Median(test_angles),
            -np.pi / 2,
            msg="incorrectly calculating median of list of angles",
        )

        # angle scatter (iqr)
        self.assertAlmostEqual(
            ap.utils.angle_operations.Angle_Scatter(test_angles),
            np.pi,
            msg="incorrectly calculating iqr of list of angles",
        )

    def test_angle_com(self):
        pixelscale = 0.8
        tar = make_basic_sersic(
            N=50,
            M=50,
            pixelscale=pixelscale,
            x=24.5*pixelscale,
            y=24.5*pixelscale,
            PA = 115 * np.pi / 180,
        )

        res = ap.utils.angle_operations.Angle_COM_PA(tar.data.detach().cpu().numpy())

        self.assertAlmostEqual(res + np.pi/2, 115 * np.pi / 180, delta = 0.1)
        
        
if __name__ == "__main__":
    unittest.main()
