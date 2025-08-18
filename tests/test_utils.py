import numpy as np
from scipy.special import gamma
import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian

######################################################################
# Util functions
######################################################################


def test_make_psf():

    target = make_basic_gaussian(x=10, y=10)
    target += make_basic_gaussian(x=40, y=40, rand=54321)

    assert np.all(
        np.isfinite(ap.backend.to_numpy(target.data))
    ), "Target image should be finite after creation"


def test_conversions_units():

    # flux to sb
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
