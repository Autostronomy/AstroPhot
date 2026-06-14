import os
import glob
import shutil
import subprocess
import pytest
import astrophot as ap

def test_cli():
    if ap.backend.backend == "jax":
        pytest.skip("Requires torch backend")
    
    # Path to CLI script
    cli_script_src = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "docs", "source", "prebuilt", "single_model_cli.py"
    )
    
    # Target image for testing
    target_image_src = os.path.join(
        os.path.split(os.path.dirname(__file__))[0],
        "docs", "source", "tutorials", "target_image.fits"
    )
    
    # Copy them to current working directory (which is where pytest runs, usually root or tests/)
    cli_script = "single_model_cli.py"
    target_image = "target_image.fits"
    
    shutil.copy(cli_script_src, cli_script)
    shutil.copy(target_image_src, target_image)
    
    try:
        # Run the CLI script
        subprocess.run(
            ["python", cli_script, target_image, "--name", "test_cli_output", "--verbose", "0", "--model_type", "sersic_galaxy_model", "--zeropoint", "25.0"],
            check=True,
        )
        
        # Verify outputs are created
        assert os.path.exists("test_cli_output_parameters.yaml")
        assert os.path.exists("test_cli_output_model_image.fits")
        assert os.path.exists("test_cli_output_residual_image.fits")
        assert os.path.exists("test_cli_output_covariance_matrix.npy")
        
    finally:
        # Cleanup
        for file in [cli_script, target_image, "test_cli_output_parameters.yaml", 
                     "test_cli_output_model_image.fits", "test_cli_output_residual_image.fits", 
                     "test_cli_output_covariance_matrix.npy"]:
            if os.path.exists(file):
                os.remove(file)
