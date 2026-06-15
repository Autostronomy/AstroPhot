import os
import shutil
import subprocess
import numpy as np
from astropy.io import fits
from photutils.segmentation import detect_sources, deblend_sources


def test_single_cli():

    # Target image for testing
    target_image_src = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "docs",
        "source",
        "tutorials",
        "target_image.fits",
    )

    # Copy them to current working directory (which is where pytest runs, usually root or tests/)
    target_image = "target_image.fits"

    shutil.copy(target_image_src, target_image)

    try:
        # Run the CLI script
        subprocess.run(
            [
                "astrophot_single",
                target_image,
                "--name",
                "test_cli_output",
                "--verbose",
                "1",
                "--model_type",
                "sersic_galaxy_model",
                "--zeropoint",
                "25.0",
                "--dump",
            ],
            check=True,
        )

        # Verify outputs are created
        assert os.path.exists("test_cli_output_parameters.yaml")
        assert os.path.exists("test_cli_output_model_image.fits")
        assert os.path.exists("test_cli_output_residual_image.fits")
        assert os.path.exists("test_cli_output_covariance_matrix.npy")
        assert os.path.exists("single_astrophot_model.py")

    finally:
        # Cleanup
        for file in [
            target_image,
            "test_cli_output_parameters.yaml",
            "test_cli_output_model_image.fits",
            "test_cli_output_residual_image.fits",
            "test_cli_output_covariance_matrix.npy",
            "single_astrophot_model.py",
        ]:
            if os.path.exists(file):
                os.remove(file)


def test_segmap_cli():

    # Target image for testing
    target_image_src = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "docs",
        "source",
        "tutorials",
        "group_target_image.fits",
    )

    # Copy them to current working directory (which is where pytest runs, usually root or tests/)
    target_image = "group_target_image.fits"

    shutil.copy(target_image_src, target_image)

    hdu = fits.open("group_target_image.fits")
    target_data = np.array(hdu[0].data, dtype=np.float64)
    initsegmap = detect_sources(target_data, threshold=0.02, npixels=6)
    segmap = deblend_sources(target_data, initsegmap, npixels=5).data
    fits.writeto("group_segmap.fits", segmap, overwrite=True)

    try:
        # Run the CLI script
        subprocess.run(
            [
                "astrophot_segmap",
                target_image,
                "group_segmap.fits",
                "--name",
                "test_segmap_cli",
                "--verbose",
                "1",
                "--model_type",
                "sersic_galaxy_model",
                "--window_expand_border",
                "5",
                "--filter_ids",
                "0",
                "8",
                "--zeropoint",
                "25.0",
                "--dump",
            ],
            check=True,
        )

        # Verify outputs are created
        assert os.path.exists("test_segmap_cli_parameters.yaml")
        assert os.path.exists("test_segmap_cli_model_images.fits")
        assert os.path.exists("test_segmap_cli_residual_images.fits")
        assert os.path.exists("test_segmap_cli_covariance_matrix.npz")
        assert os.path.exists("segmap_astrophot_model.py")

    finally:
        # Cleanup
        for file in [
            target_image,
            "test_segmap_cli_parameters.yaml",
            "test_segmap_cli_model_images.fits",
            "test_segmap_cli_residual_images.fits",
            "test_segmap_cli_covariance_matrix.npz",
            "segmap_astrophot_model.py",
            "group_segmap.fits",
        ]:
            if os.path.exists(file):
                os.remove(file)
