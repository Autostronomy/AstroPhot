"""The `astrophot` command (astrophot/cli).

Most tests call the `main` functions in this process, which is quick. The
installed command is started once per sub-command to check that it is wired
up. `astrophot segmap` is also the test of the segmentation map helpers in
astrophot/utils/initialize/segmentation_map.py.
"""

import subprocess
import sys
from pathlib import Path

import astrophot as ap
import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astrophot import cli
from astrophot.cli import cli_segmap, cli_single
from photutils.segmentation import deblend_sources, detect_sources

TUTORIALS = Path(__file__).resolve().parent.parent / "docs" / "source" / "tutorials"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_segmap(data):
    # positional arguments, the keyword names differ between photutils versions
    segmap = detect_sources(data, 0.02, 6)
    return deblend_sources(data, segmap, 5).data


@pytest.fixture(scope="session")
def cutout(tmp_path_factory):
    """Path of a non-square cutout (200 rows, 150 columns) of the tutorial galaxy
    with its WCS. The galaxy is not at the centre. Only read this file."""
    with fits.open(TUTORIALS / "target_image.fits") as hdul:
        data = hdul[0].data[250:450, 270:420]
        wcs = WCS(hdul[0].header)[250:450, 270:420]
    path = tmp_path_factory.mktemp("cutout") / "cutout.fits"
    fits.writeto(path, data, wcs.to_header())
    return path


# The two objects of the `blended_pair` fixture, as (x, y) pixel positions
PAIR_TARGET_XY = (82.3, 24.6)  # far from the diagonal of the image
PAIR_NEIGHBOUR_XY = (96.0, 26.0)


@pytest.fixture(scope="session")
def blended_pair(tmp_path_factory):
    """Paths of an image (60 rows, 110 columns, no WCS) and its segmentation map.

    Segment 1 is a faint Gaussian (sigma 4, flux 100) and segment 2 a brighter
    neighbour (sigma 2, flux 500) close enough to fall inside its fitting window.
    Only read these files.
    """
    ny, nx = 60, 110
    y, x = np.mgrid[:ny, :nx]

    def gaussian(x0, y0, sigma, flux):
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        return flux * np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    rng = np.random.default_rng(42)
    data = gaussian(*PAIR_TARGET_XY, 4.0, 100.0) + gaussian(*PAIR_NEIGHBOUR_XY, 2.0, 500.0)
    data = data + rng.normal(scale=1e-3, size=data.shape)
    segmap = np.zeros((ny, nx), dtype=np.int32)
    segmap[np.hypot(x - PAIR_TARGET_XY[0], y - PAIR_TARGET_XY[1]) < 10] = 1
    segmap[np.hypot(x - PAIR_NEIGHBOUR_XY[0], y - PAIR_NEIGHBOUR_XY[1]) < 8] = 2
    directory = tmp_path_factory.mktemp("blended_pair")
    fits.writeto(directory / "target.fits", data)
    fits.writeto(directory / "segmap.fits", segmap)
    return directory / "target.fits", directory / "segmap.fits"


######################################################################
# Installed commands
######################################################################


def test_installed_single(cutout, tmp_path):
    subprocess.run(
        [
            "astrophot",
            "single",
            str(cutout),
            "--name",
            "test_cli_output",
            "--verbose",
            "1",
        ]
        + ["--model_type", "sersic_galaxy_model", "--zeropoint", "25.0", "--dump"],
        check=True,
        cwd=tmp_path,
    )

    # Verify outputs are created
    assert (tmp_path / "test_cli_output_parameters.yaml").is_file()
    assert (tmp_path / "test_cli_output_model_image.fits").is_file()
    assert (tmp_path / "test_cli_output_residual_image.fits").is_file()
    assert (tmp_path / "test_cli_output_covariance_matrix.npy").is_file()
    assert (tmp_path / "single_astrophot_model.py").is_file()


def test_installed_segmap(blended_pair, tmp_path):
    target, segmap = blended_pair
    subprocess.run(
        ["astrophot", "segmap", str(target), str(segmap), "--name", "test_segmap_cli"]
        + ["--verbose", "1", "--model_type", "gaussian_galaxy_model", "--dump"],
        check=True,
        cwd=tmp_path,
    )

    # Verify outputs are created
    assert (tmp_path / "test_segmap_cli_model_images.fits").is_file()
    assert (tmp_path / "test_segmap_cli_residual_images.fits").is_file()
    assert (tmp_path / "test_segmap_cli_covariance_matrix.npz").is_file()
    assert (tmp_path / "segmap_astrophot_model.py").is_file()

    results = load_yaml(tmp_path / "test_segmap_cli_parameters.yaml")
    assert set(results) == {
        "test_segmap_cli_1",
        "test_segmap_cli_2",
    }, "one fit per segment"


######################################################################
# The astrophot command
######################################################################


def test_cli_dispatch(cutout, capsys):
    # --dump without a target only writes the script, so no fit is run here
    cli.main(["single", "--dump"])
    cli.main(["segmap", "--dump"])
    for script in ("single_astrophot_model.py", "segmap_astrophot_model.py"):
        compile(Path(script).read_text(), script, "exec")

    cli.main([])
    out = capsys.readouterr().out
    assert "single" in out and "segmap" in out, "no arguments should list the commands"

    cli.main(["--version"])
    assert ap.__version__ in capsys.readouterr().out

    with pytest.raises(SystemExit) as error:
        cli.main(["singel", str(cutout)])
    assert "unknown command" in str(error.value)

    # the help of a sub-command names the command as the user typed it
    with pytest.raises(SystemExit):
        cli.main(["segmap", "--help"])
    assert "usage: astrophot segmap" in capsys.readouterr().out


def test_import_is_light():
    """The commands wait for `import astrophot`, so slow packages are only loaded when used."""
    heavy = ("scipy", "matplotlib", "pyvo", "emcee")
    code = f"import sys, astrophot; print([m for m in {heavy} if m in sys.modules])"
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip().splitlines()[-1] == "[]", "these should be imported where used"


######################################################################
# Single model
######################################################################


def test_single_cli_model_params(cutout, tmp_path):
    outdir = tmp_path / "new" / "directory"
    cli_single.main(
        [
            str(cutout),
            "--name",
            "gal",
            "--outdir",
            str(outdir),
            "--verbose",  # also print the results
            "1",
            "--log",
            str(tmp_path / "fit.log"),
            "--zeropoint",
            "22.5",
            "--center",  # two values make a list, negative numbers are values
            "-1.0",
            "3.5",
            "--n",  # not an abbreviation of --name
            "{'value': 2.5, 'dynamic': False}",
            "--q=0.7",
        ]
    )
    results = load_yaml(outdir / "gal_parameters.yaml")
    params = results["gal"]["parameters"]
    assert params["n"] == {"value": 2.5, "uncertainty": None, "dynamic": False}
    assert set(params) == {
        "center",
        "q",
        "PA",
        "n",
        "Re",
        "Ie",
    }, "only the model's parameters"
    assert params["q"]["dynamic"] and params["q"]["uncertainty"] > 0
    # Position in arcsec from the reference coordinate of the WCS. The full
    # tutorial image puts the galaxy at (-1.06, 3.53).
    assert np.allclose(params["center"]["value"], (-1.06, 3.53), atol=0.3)
    assert np.isfinite(results["gal"]["total_magnitude"])
    assert results["sky_model"]["parameters"]["I0"]["dynamic"]

    with fits.open(outdir / "gal_model_image.fits") as hdul:
        assert hdul[0].data.shape == (200, 150)
    assert np.load(outdir / "gal_covariance_matrix.npy").shape == (7, 7)
    assert "Final Chi^2/DoF" in (tmp_path / "fit.log").read_text(), "--log keeps the messages"


def test_single_cli_inputs(cutout, tmp_path):
    """Mask, variance, PSF and target read from typical FITS files."""
    data = fits.getdata(cutout)
    # multi extension file with an empty primary HDU
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data, fits.getheader(cutout), name="SCI")]
    ).writeto(tmp_path / "mef.fits")
    # integer mask, FITS stores it big endian
    mask = np.zeros(data.shape, dtype=np.int16)
    mask[:5] = 1
    fits.writeto(tmp_path / "mask.fits", mask)
    # variance with unusable pixels
    variance = np.full(data.shape, 0.01, dtype=np.float32)
    variance[10, 10] = 0.0
    variance[11, 11] = np.nan
    fits.writeto(tmp_path / "variance.fits", variance)
    fits.writeto(tmp_path / "psf.fits", ap.utils.initialize.gaussian_psf(1.0, 11, 1.0))

    cli_single.main(
        [
            str(tmp_path / "mef.fits"),
            "--target_hdu",
            "SCI",
            "--mask",
            str(tmp_path / "mask.fits"),
            "--variance",
            str(tmp_path / "variance.fits"),
            "--psf",
            str(tmp_path / "psf.fits"),
            "--window",
            "20",
            "400",  # beyond the image, clipped to 150
            "30",
            "170",
            "--sky_locked",
            "--outdir",
            str(tmp_path),
        ]
    )
    results = load_yaml(tmp_path / "astrophot_model_parameters.yaml")
    assert results["astrophot_model"]["message"]
    assert not results["sky_model"]["parameters"]["I0"]["dynamic"], "sky should be locked"
    assert results["sky_model"]["parameters"]["I0"]["value"] is not None
    with fits.open(tmp_path / "astrophot_model_model_image.fits") as hdul:
        assert hdul[0].data.shape == (140, 130), "model image should cover the window"


def test_single_cli_without_wcs(cutout, tmp_path):
    fits.writeto(tmp_path / "plain.fits", fits.getdata(cutout))
    common = [
        "--outdir",
        str(tmp_path),
        "--verbose",
        "0",
        "--no_save_images",
        "--no_save_cov",
    ]

    cli_single.main([str(tmp_path / "plain.fits"), "--variance", "auto", *common])
    assert not (tmp_path / "astrophot_model_model_image.fits").exists()
    assert not (tmp_path / "astrophot_model_covariance_matrix.npy").exists()
    params = load_yaml(tmp_path / "astrophot_model_parameters.yaml")["astrophot_model"]
    center = np.array(params["parameters"]["center"]["value"])
    # without a WCS the coordinates are pixels, from the centre of the first pixel
    assert np.allclose(center, (83.5, 113.0), atol=1.5)

    # an explicit pixelscale also replaces the WCS of the header
    cli_single.main([str(cutout), "--name", "scaled", "--pixelscale", "0.5", *common])
    params = load_yaml(tmp_path / "scaled_parameters.yaml")["scaled"]
    assert np.allclose(params["parameters"]["center"]["value"], 0.5 * center, atol=1.0)


@pytest.mark.parametrize(
    "bad_args",
    [
        [],  # no target
        ["missing.fits"],
        ["--Re", "10", "TARGET"],  # target must come first
        ["TARGET", "--name", "my-galaxy"],
        ["TARGET", "--target_hdu", "3"],
        ["TARGET", "--target_hdu", "SCI"],
        ["TARGET", "--window", "500", "600", "0", "10"],
        ["TARGET", "--mask", "small.fits"],
        ["TARGET", "--variance", "small.fits"],
        ["TARGET", "--psf", "small.fits"],  # even number of pixels
        ["TARGET", "--mask", "missing.fits"],
    ],
)
def test_single_cli_bad_arguments(cutout, tmp_path, bad_args):
    fits.writeto("small.fits", np.ones((10, 10)))
    with pytest.raises(SystemExit) as error:
        cli_single.main([str(cutout) if arg == "TARGET" else arg for arg in bad_args])
    assert error.value.code not in (0, None), "should exit with an error"
    assert not list(tmp_path.glob("*.yaml")), "nothing should be written"


def test_parse_model_params():
    params = cli_single.parse_model_params(
        [
            "--n",
            "2",
            "--center",
            "-1.5",
            "3e1",
            "--q=0.5",
            "--psf_convolve",
            "--PA",
            "[1, 2]",
        ]
        + ["--Re", "{'value': 3.0, 'dynamic': False}", "--label", "text"]
    )
    assert params == {
        "n": 2,
        "center": [-1.5, 30.0],
        "q": 0.5,
        "psf_convolve": True,
        "PA": [1, 2],
        "Re": {"value": 3.0, "dynamic": False},
        "label": "text",
    }
    with pytest.raises(SystemExit):
        cli_single.parse_model_params(["stray", "--n", "2"])


######################################################################
# Segmentation map
######################################################################


def test_segmap_cli_outputs(tmp_path):
    """Non-square image, with a catalogue of initial parameters."""
    with fits.open(TUTORIALS / "group_target_image.fits") as hdul:
        data = np.array(hdul[0].data[20:140], dtype=np.float64)
        wcs = WCS(hdul[0].header)[20:140]
    fits.writeto(tmp_path / "target.fits", data, wcs.to_header())
    segmap = make_segmap(data)
    fits.writeto(tmp_path / "segmap.fits", segmap)
    ids = [int(i) for i in np.unique(segmap) if i != 0]
    assert len(ids) > 3, "test needs a few objects"
    fit_ids, skip_ids, bad_id = ids[:3], ids[3:], ids[1]
    with open(tmp_path / "catalogue.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                fit_ids[0]: {"n": {"value": 1.0, "dynamic": False}},
                bad_id: {"not_a_parameter": 1.0},
                skip_ids[0]: {"n": 1.0},
            },
            f,
        )

    cli_segmap.main(
        [
            str(tmp_path / "target.fits"),
            str(tmp_path / "segmap.fits"),
            "--cat",
            str(tmp_path / "catalogue.yaml"),
            "--filter_ids",
            *[str(i) for i in skip_ids],
            "--window_expand_border",
            "5",
            "--zeropoint",
            "22.5",
            "--outdir",
            str(tmp_path / "out"),
            "--verbose",
            "0",
        ]
    )
    results = load_yaml(tmp_path / "out" / "astrophot_model_parameters.yaml")
    assert set(results) == {f"astrophot_model_{i}" for i in fit_ids}

    # the catalogue overrides the initial parameters
    locked = results[f"astrophot_model_{fit_ids[0]}"]
    assert locked["parameters"]["n"] == {
        "value": 1.0,
        "uncertainty": None,
        "dynamic": False,
    }
    assert np.isfinite(locked["total_magnitude"])
    # an object which fails is reported, the others are still fit
    assert results[f"astrophot_model_{bad_id}"]["message"].startswith("fit failed")
    assert "parameters" in results[f"astrophot_model_{fit_ids[2]}"]

    good_ids = [fit_ids[0], fit_ids[2]]
    cov = np.load(tmp_path / "out" / "astrophot_model_covariance_matrix.npz")
    assert set(cov.files) == {f"astrophot_model_{i}" for i in good_ids}
    for kind in ("model", "residual"):
        with fits.open(tmp_path / "out" / f"astrophot_model_{kind}_images.fits") as hdul:
            assert len(hdul) == 1 + len(good_ids)
            for i in good_ids:
                hdu = hdul[f"astrophot_model_{i}_{kind}"]
                # the window holds the whole segment
                j_seg, i_seg = np.where(segmap == i)
                assert hdu.data.shape[0] >= np.ptp(j_seg) + 1
                assert hdu.data.shape[1] >= np.ptp(i_seg) + 1
                # each extension has the WCS of its window, check against the full image
                corner = WCS(hdu.header).all_pix2world(0, 0, 0)
                low = wcs.all_world2pix(*corner, 0)
                assert np.allclose(low, np.round(low), atol=1e-4), "window on the pixel grid"
                low = np.round(low)
                assert low[0] <= i_seg.min() and low[1] <= j_seg.min()


def test_segmap_cli_masks_interlopers(blended_pair, tmp_path):
    """A bright neighbour inside the fitting window must not affect the fit."""
    target, segmap = blended_pair
    cli_segmap.main(
        [
            str(target),
            str(segmap),
            "--model_type",
            "gaussian_galaxy_model",
            "--filter_ids",
            "2",
            "--window_expand_border",
            "6",
            "--initial_sky",
            "0",
            "--sky_locked",
            "--outdir",
            str(tmp_path),
            "--verbose",
            "0",
        ]
    )
    results = load_yaml(tmp_path / "astrophot_model_parameters.yaml")
    assert list(results) == ["astrophot_model_1"]
    params = results["astrophot_model_1"]["parameters"]
    # no WCS so the coordinates are in pixels
    # No WCS so the coordinates are in pixels. Left unmasked, the neighbour
    # would pull the centre several pixels towards it.
    assert np.allclose(params["center"]["value"], PAIR_TARGET_XY, atol=0.1)
    assert np.isclose(params["sigma"]["value"], 4.0, atol=0.2)
    # the fit ends round (q = 1), where the position angle is not constrained and
    # the hessian is singular. The other parameters still get an uncertainty.
    assert params["sigma"]["uncertainty"] > 0
    with fits.open(tmp_path / "astrophot_model_model_images.fits") as hdul:
        assert hdul[1].header["CRPIX1"] < 0, "window should start right of the image origin"


def test_segmap_cli_bad_arguments():
    fits.writeto("target.fits", np.ones((20, 30)))
    fits.writeto("empty_segmap.fits", np.zeros((20, 30), dtype=np.int32))
    fits.writeto("wrong_shape.fits", np.zeros((30, 20), dtype=np.int32))
    Path("list.yaml").write_text("- 1\n- 2\n")
    for bad_args in (
        ["target.fits"],
        ["target.fits", "missing.fits"],
        ["target.fits", "wrong_shape.fits"],
        ["target.fits", "empty_segmap.fits"],
        ["target.fits", "empty_segmap.fits", "--cat", "list.yaml"],
        ["target.fits", "empty_segmap.fits", "--name", "a/b"],
    ):
        with pytest.raises(SystemExit) as error:
            cli_segmap.main(bad_args)
        assert error.value.code not in (0, None), "should exit with an error"
