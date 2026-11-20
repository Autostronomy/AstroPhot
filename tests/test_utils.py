"""The helper functions of astrophot/utils: unit conversions, Sersic conversions,
interpolation, the segmentation map helpers and the Legacy Survey cutouts.
"""

import platform
import sys
import warnings

import astrophot as ap
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import gamma


def test_target_addition(gaussian_target_factory):
    target = gaussian_target_factory(x=10, y=10)
    target += gaussian_target_factory(x=30, y=30, rand=54321)

    assert np.all(
        np.isfinite(ap.backend.to_numpy(target.data))
    ), "Target image should be finite after creation"


def test_conversions_units():

    # flux to sb
    assert (
        ap.utils.conversions.units.flux_to_sb(1.0, 1.0, 0.0) == 0
    ), "flux incorrectly converted to sb"

    # sb to flux
    assert ap.utils.conversions.units.sb_to_flux(1.0, 1.0, 0.0) == (
        10 ** (-1 / 2.5)
    ), "sb incorrectly converted to flux"

    # flux to mag no error
    assert (
        ap.utils.conversions.units.flux_to_mag(1.0, 0.0) == 0
    ), "flux incorrectly converted to mag (no error)"

    # flux to mag with error
    assert ap.utils.conversions.units.flux_to_mag(1.0, 0.0, fluxe=1.0) == (
        0.0,
        2.5 / np.log(10),
    ), "flux incorrectly converted to mag (with error)"

    # mag to flux no error:
    assert ap.utils.conversions.units.mag_to_flux(1.0, 0.0, mage=None) == (
        10 ** (-1 / 2.5)
    ), "mag incorrectly converted to flux (no error)"

    # mag to flux with error:
    for i in range(1):
        assert np.isclose(
            ap.utils.conversions.units.mag_to_flux(1.0, 0.0, mage=1.0)[i],
            (10 ** (-1.0 / 2.5), np.log(10) * (1.0 / 2.5) * 10 ** (-1.0 / 2.5))[i],
        ), "mag incorrectly converted to flux (with error)"

    # magperarcsec2 to mag with area A defined
    assert np.isclose(
        ap.utils.conversions.units.magperarcsec2_to_mag(1.0, a=None, b=None, A=1.0),
        (1.0 - 2.5 * np.log10(1.0)),
    ), "mag/arcsec^2 incorrectly converted to mag (area A given, a and b not defined)"

    # magperarcsec2 to mag with semi major and minor axes defined (a, and b)
    assert np.isclose(
        ap.utils.conversions.units.magperarcsec2_to_mag(1.0, a=1.0, b=1.0, A=None),
        (1.0 - 2.5 * np.log10(np.pi)),
    ), "mag/arcsec^2 incorrectly converted to mag (semi major/minor axes defined)"

    # mag to magperarcsec2 with area A defined
    assert np.isclose(
        ap.utils.conversions.units.mag_to_magperarcsec2(1.0, a=None, b=None, A=1.0, R=None),
        (1.0 + 2.5 * np.log10(1.0)),
    ), "mag incorrectly converted to mag/arcsec^2 (area A given)"

    # mag to magperarcsec2 with radius R given (assumes circular)
    assert np.isclose(
        ap.utils.conversions.units.mag_to_magperarcsec2(1.0, a=None, b=None, A=None, R=1.0),
        (1.0 + 2.5 * np.log10(np.pi)),
    ), "mag incorrectly converted to mag/arcsec^2 (radius R given)"

    # mag to magperarcsec2 with semi major and minor axes defined (a, and b)
    assert np.isclose(
        ap.utils.conversions.units.mag_to_magperarcsec2(1.0, a=1.0, b=1.0, A=None, R=None),
        (1.0 + 2.5 * np.log10(np.pi)),
    ), "mag incorrectly converted to mag/arcsec^2 (area A given)"


def test_conversion_functions():

    sersic_n = ap.utils.conversions.functions.sersic_n_to_b(1.0)
    # sersic I0 to flux - numpy
    assert np.isclose(
        ap.utils.conversions.functions.sersic_I0_to_flux_np(1.0, 1.0, 1.0, 1.0),
        (2 * np.pi * gamma(2)),
    ), "Error converting sersic central intensity to flux (np)"
    # sersic flux to I0 - numpy
    assert np.isclose(
        ap.utils.conversions.functions.sersic_flux_to_I0_np(1.0, 1.0, 1.0, 1.0),
        (1.0 / (2 * np.pi * gamma(2))),
    ), "Error converting sersic flux to central intensity (np)"

    # sersic Ie to flux - numpy
    assert np.isclose(
        ap.utils.conversions.functions.sersic_Ie_to_flux_np(1.0, 1.0, 1.0, 1.0),
        (2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2)),
    ), "Error converting sersic effective intensity to flux (np)"

    # sersic flux to Ie - numpy
    assert np.isclose(
        ap.utils.conversions.functions.sersic_flux_to_Ie_np(1.0, 1.0, 1.0, 1.0),
        (1 / (2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2))),
    ), "Error converting sersic flux to effective intensity (np)"

    # inverse sersic - numpy
    assert np.isclose(
        ap.utils.conversions.functions.sersic_inv_np(1.0, 1.0, 1.0, 1.0),
        (1.0 - (1.0 / sersic_n) * np.log(1.0)),
    ), "Error computing inverse sersic function (np)"

    # sersic I0 to flux - torch
    tv = ap.backend.as_array([[1.0]], dtype=ap.backend.float64)
    assert ap.backend.allclose(
        ap.utils.conversions.functions.sersic_I0_to_flux_np(tv, tv, tv, tv),
        ap.backend.as_array([[2 * np.pi * gamma(2)]]),
        rtol=1e-7,
    ), "Error converting sersic central intensity to flux (torch)"

    # sersic flux to I0 - torch
    assert ap.backend.allclose(
        ap.utils.conversions.functions.sersic_flux_to_I0_np(tv, tv, tv, tv),
        ap.backend.as_array([[1.0 / (2 * np.pi * gamma(2))]]),
        rtol=1e-7,
    ), "Error converting sersic flux to central intensity (torch)"

    # sersic Ie to flux - torch
    assert ap.backend.allclose(
        ap.utils.conversions.functions.sersic_Ie_to_flux_np(tv, tv, tv, tv),
        ap.backend.as_array([[2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2)]]),
        rtol=1e-7,
    ), "Error converting sersic effective intensity to flux (torch)"

    # sersic flux to Ie - torch
    assert ap.backend.allclose(
        ap.utils.conversions.functions.sersic_flux_to_Ie_np(tv, tv, tv, tv),
        ap.backend.as_array([[1 / (2 * np.pi * gamma(2) * np.exp(sersic_n) * sersic_n ** (-2))]]),
        rtol=1e-7,
    ), "Error converting sersic flux to effective intensity (torch)"

    # inverse sersic - torch
    assert ap.backend.allclose(
        ap.utils.conversions.functions.sersic_inv_np(tv, tv, tv, tv),
        ap.backend.as_array([[1.0 - (1.0 / sersic_n) * np.log(1.0)]]),
        rtol=1e-7,
    ), "Error computing inverse sersic function (torch)"


def test_interp2d(gaussian_target_factory):
    target = gaussian_target_factory()

    I, J = np.meshgrid(
        np.arange(target.data.shape[0]), np.arange(target.data.shape[1]), indexing="ij"
    )
    interp_data = ap.utils.interpolate.interp2d(
        target.data, ap.backend.as_array(I), ap.backend.as_array(J)
    )

    assert ap.backend.allclose(
        interp_data, target.data, rtol=1e-7
    ), "Interpolated data should be close to original data"

    with pytest.raises(ValueError):
        ap.utils.interpolate.interp2d(
            target.data, ap.backend.as_array(I), ap.backend.as_array(J), padding_mode="invalid"
        )


def test_ignore_numpy_warnings_is_temporary():
    """Warnings are only hidden inside the call, the settings of the user are not changed."""

    @ap.utils.decorators.ignore_numpy_warnings
    def noisy():
        warnings.warn("old", DeprecationWarning)
        return np.log(np.zeros(1))

    with warnings.catch_warnings(), np.errstate(divide="raise"):
        warnings.simplefilter("error", DeprecationWarning)
        filters = list(warnings.filters)
        assert noisy() == -np.inf
        assert warnings.filters == filters, "warning filters should be put back"
        assert np.geterr()["divide"] == "raise", "numpy error settings should be put back"
        with pytest.raises(DeprecationWarning):
            warnings.warn("old", DeprecationWarning)


######################################################################
# Segmentation maps (astrophot/utils/initialize/segmentation_map.py)
######################################################################


@pytest.fixture
def two_objects():
    """Image of 30 rows and 40 columns with its segmentation map.

    Segment 1 is bright and elongated along x at (x, y) = (12, 10), segment 2
    is faint and round at (30, 20).
    """
    y, x = np.mgrid[:30, :40]

    def blob(x0, y0, sigma_x, sigma_y, peak):
        return peak * np.exp(-0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2))

    data = blob(12, 10, 4, 1.6, 10) + blob(30, 20, 2, 2, 4)
    segmap = np.zeros(data.shape, dtype=int)
    segmap[5:16, 2:23] = 1  # 21 columns, 11 rows
    segmap[16:25, 26:35] = 2  # 9 columns, 9 rows
    return ap.TargetImage(data=data, pixelscale=1.0), segmap


def test_segmentation_map_shapes(two_objects, tmp_path):
    target, segmap = two_objects
    initialize = ap.utils.initialize

    centroids = initialize.centroids_from_segmentation_map(segmap, target)
    assert set(centroids) == {1, 2}, "the background, id 0, is skipped"
    assert np.allclose(centroids[1], (12, 10), atol=0.01)
    assert np.allclose(centroids[2], (30, 20), atol=0.01)

    PAs = initialize.PA_from_segmentation_map(segmap, target, centroids)
    qs = initialize.q_from_segmentation_map(segmap, target, centroids)
    assert np.isclose(PAs[1], np.pi / 2, atol=0.01), "long axis is along x"
    assert qs[1] < 0.7 and qs[2] == 0.9, "segment 1 is elongated, segment 2 is round"

    # the centroids are found when they are not given, also when ids are skipped
    assert initialize.PA_from_segmentation_map(segmap, target) == PAs
    assert initialize.q_from_segmentation_map(segmap, target) == qs
    assert initialize.q_from_segmentation_map(segmap, target, skip_index=(0, 1)) == {2: qs[2]}

    # a segment without usable pixel values gets a default shape
    broken = target.copy()
    broken.data = np.where(segmap == 2, np.nan, ap.backend.to_numpy(target.data))
    assert initialize.PA_from_segmentation_map(segmap, broken, centroids)[2] == np.pi / 2
    assert initialize.q_from_segmentation_map(segmap, broken, centroids)[2] == 0.7

    # the map may also be a file
    fits.writeto(tmp_path / "segmap.fits", segmap)
    np.save(tmp_path / "segmap.npy", segmap)
    for name in ("segmap.fits", "segmap.npy"):
        assert initialize.centroids_from_segmentation_map(tmp_path / name, target) == centroids
        assert initialize.centroids_from_segmentation_map(str(tmp_path / name), target) == centroids
    with pytest.raises(ValueError):
        initialize.centroids_from_segmentation_map("segmap.txt", target)


def test_segmentation_map_windows(two_objects):
    target, segmap = two_objects
    initialize = ap.utils.initialize

    # windows are [[xmin, ymin], [xmax, ymax]], the upper limits are exclusive
    windows = initialize.windows_from_segmentation_map(segmap)
    assert windows == {1: [[2, 5], [23, 16]], 2: [[26, 16], [35, 25]]}
    assert list(initialize.windows_from_segmentation_map(segmap, skip_index=(0, 1))) == [2]

    assert initialize.scale_windows(windows, expand_border=3) == {
        1: [[-1, 2], [26, 19]],
        2: [[23, 13], [38, 28]],
    }
    # with an image the windows stop at its edges
    assert initialize.scale_windows(windows, image=target, expand_scale=2, expand_border=3) == {
        1: [[0, 0], [36, 24]],
        2: [[18, 8], [40, 30]],
    }

    # segment 1 is 21 by 11 pixels with a total flux near 400, segment 2 is 9 by 9 with near 100
    def kept(**criteria):
        return list(initialize.filter_windows(windows, **criteria))

    assert kept() == [1, 2]
    assert kept(min_size=10) == [1] and kept(max_size=10) == [2]
    assert kept(min_area=100) == [1] and kept(max_area=100) == [2]
    assert kept(min_flux=200, image=target) == [1] and kept(max_flux=200, image=target) == [2]

    # an image of the same sky with pixels of half the size, cut short along y
    fine = ap.TargetImage(data=np.zeros((45, 80)), pixelscale=0.5)
    assert initialize.transfer_windows(windows, target, fine) == {
        1: [[4, 10], [46, 32]],
        2: [[52, 32], [70, 45]],
    }


######################################################################
# Legacy Survey cutouts (astrophot/utils/fitsopen.py)
######################################################################


def survey_cutout():
    """HDUList like a cutout from the Legacy Survey image service, 21 rows and 31 columns.

    As in the real cutouts east is to the right, and the reference pixel is that
    of the survey brick, far outside of the image.
    """
    header = fits.Header()
    header["CRPIX1"], header["CRPIX2"] = 698.5, 1458.5
    header["CRVAL1"], header["CRVAL2"] = 155.0, 15.0
    header["CD1_1"], header["CD1_2"] = 7e-5, 1e-6
    header["CD2_1"], header["CD2_2"] = -1e-6, 7e-5
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    data = np.arange(21 * 31, dtype=np.float64).reshape(21, 31)
    return fits.HDUList([fits.PrimaryHDU(data, header)])


def test_flip_hdu():
    """The image is mirrored along RA, and every pixel keeps its place on the sky."""
    from astrophot.utils.fitsopen import flip_hdu

    original = survey_cutout()
    flipped = flip_hdu(survey_cutout())
    assert np.all(flipped[0].data == original[0].data[:, ::-1])
    assert flipped[0].header["CD1_1"] == -original[0].header["CD1_1"], "east is now to the left"
    i, j = np.meshgrid(np.arange(31), np.arange(21))
    sky = WCS(original[0].header).pixel_to_world_values(i, j)
    sky_flipped = WCS(flipped[0].header).pixel_to_world_values(30 - i, j)
    assert np.allclose(sky, sky_flipped, rtol=0, atol=1e-10)

    with pytest.raises(AssertionError):
        flip_hdu(fits.HDUList([fits.PrimaryHDU(np.zeros((5, 5)))]))


def test_ls_open(monkeypatch, tmp_path):
    """The deepest stacked image of the band is downloaded. The survey service is replaced here."""
    sia = pytest.importorskip("pyvo.dal.sia")
    from astropy.table import Table
    from astrophot.utils import fitsopen

    rows = [  # proctype, prodtype, band, exposure time, url
        ("Stack", "image", "r DECam", 100.0, "shallow_r"),
        ("Stack", "image", "r DECam", 300.0, "deep_r"),
        ("Stack", "image", "g DECam", 900.0, "deep_g"),
        ("Stack", "weight", "r DECam", 900.0, "weight_r"),
        ("InstCal", "image", "r DECam", 900.0, "single_exposure_r"),
        ("Stack", "image", "i DECam", 100.0, "no_wcs_i"),
    ]
    names = ("proctype", "prodtype", "obs_bandpass", "exptime", "access_url")
    searches, downloads = [], []

    class Service:
        def __init__(self, url):
            self.url = url

        def search(self, pos, size, **kwargs):
            searches.append((self.url, pos, size))
            return self

        def to_table(self):
            return Table(rows=rows, names=names, masked=True)

    def download_file(url, **kwargs):
        downloads.append(url)
        cutout = survey_cutout()
        if url.startswith("no_wcs"):
            cutout = fits.HDUList([fits.PrimaryHDU(cutout[0].data)])
        cutout.writeto(tmp_path / f"{url}.fits", overwrite=True)
        return str(tmp_path / f"{url}.fits")

    monkeypatch.setattr(sia, "SIAService", Service)
    monkeypatch.setattr(fitsopen, "download_file", download_file)

    hdu = ap.utils.ls_open(155.0, 60.0, 36.0, band="R", release="LS_DR9")
    assert downloads == ["deep_r"]
    url, pos, size = searches[0]
    assert url.endswith("/sia/ls_dr9") and pos == (155.0, 60.0)
    assert np.allclose(
        size, (0.02, 0.01)
    ), "36 arcsec in degrees, wider in RA at a declination of 60"
    assert hdu[0].header["CD1_1"] < 0, "image should be flipped to have east on the left"
    assert np.all(hdu[0].data == survey_cutout()[0].data[:, ::-1])
    hdu.close()
    if platform.system() != "Windows":  # where a file can not be removed while it is open
        assert not (tmp_path / "deep_r.fits").exists(), "the downloaded file should be removed"

    # an image without a WCS can not be flipped, it is returned as it is
    with ap.utils.ls_open(155.0, 60.0, 36.0, band="i") as hdu:
        assert np.all(hdu[0].data == survey_cutout()[0].data)

    with pytest.raises(ValueError, match="No stacked FITS image"):
        ap.utils.ls_open(155.0, 60.0, 36.0, band="z")

    monkeypatch.setitem(sys.modules, "pyvo.dal", None)  # as if pyvo is not installed
    with pytest.raises(ImportError, match="pip install pyvo"):
        ap.utils.ls_open(155.0, 60.0, 36.0)
