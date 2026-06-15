import os
import shutil
import subprocess


def test_cli():

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
