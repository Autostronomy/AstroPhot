============================
Fit Fast! with AstroPhot CLI
============================

Sometimes you just have a FITS file and want to quickly fit a simple model to
it. The AstroPhot Command-Line-Interface (CLI) is designed for this purpose. It
provides a simple interface to fit an isolated model, or a catalogue of objects
all in a single line command.

Fit a single isolated object
----------------------------

This script will fit a single model (plus a sky level) to an image. The simplest
usage is just:

.. code:: bash

    ~$ astrophot_single target_file.fits

Which will fit a sersic model to the whole image, assuming a variance of 1 for
every pixel. This is almost certainly not what you want, but as a starting point
it is a good way check that everything is working. Here is a more realistic example:

.. code:: bash

    ~$ astrophot_single target_file.fits --psf psf.fits --mask mask.fits --variance variance.fits --window 10 100 10 100 --zeropoint 22.5

In this version we pass FITS files for the PSF, mask, and variance, we only fit
a 90x90 pixel window of the image, and we set the zeropoint for magnitude
calculations. See the :doc:`tutorials/GettingStarted` tutorial for more
information on these inputs.

The final output will be a yaml file with the fitted parameters. Depending on
the options you choose, it may also output fits files for the model and residual
images as well as a numpy file with a covariance matrix of uncertainties for the
parameters.

Fit all objects in an image
---------------------------

Get the script here: :download:`segmap_models_fit.py <prebuilt/segmap_models_fit.py>`

This script will fit all objects in an image with a single model type. It will
also fit a sky model (if requested) and a single special model as the "primary
obejct" (if requested).

basic usage is to fill in these blanks at the top of the file. Even just filling
the ``target_file`` and ``segmap_file`` is enough to get started:

.. code:: python

    name = "field_name"  # used for saving files
    target_file = "<required>.fits"  # can be a numpy array instead
    segmap_file = "<required>.fits"  # can be a numpy array instead
    mask_file = None  # "<path to mask>.fits" # can be a numpy array instead
    psf_file = None  # "<path to psf>.fits" # can be a numpy array instead
    variance_file = None  # "<path to variance>.fits" # or numpy array or "auto"
    pixelscale = 0.1  # arcsec/pixel
    zeropoint = 22.5  # mag
    initial_sky = None  # If None, sky will be estimated. Recommended to set manually
    sky_locked = False
    model_type = "sersic galaxy model"  # model type for segmap entries
    segmap_filter = {}  # in pixels or ADU: min_size, min_area, min_flux
    segmap_filter_ids = []  # list of segmap ids to remove from fit
    segmap_override_init_params = {}  # Override some initial parameters for segmap models
    primary_key = None  # segmentation map id, use None to have no primary object
    primary_name = "primary object"  # name for primary object
    primary_model_type = "sersic galaxy model"
    primary_initial_params = None  # {"center": [3, 3], "q": {"value": 0.8, "locked": True}}

then run the script from the command line as a python file:

.. code:: bash

    >>> python segmap_models_fit.py

This will output the fitted parameters and save the model and residual images as
fits files. See the :doc:`tutorials/GroupModels` tutorial for more information.
