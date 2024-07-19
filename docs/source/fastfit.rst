===================
Fit Something Fast!
===================

Here are some scripts which can be used like configuration files. Simply fill in
the blanks at the top of the file and a basic fitting routine will run on your
data. This is useful to get results fast on a fairly standard dataset, or to use
as a starting point when building a more sophisticated analysis routine.

Fit a single isolated object
----------------------------

Get the script here: :download:`single_model_fit.py <prebuilt/single_model_fit.py>`

This script will fit a single object with a single model (plus a sky model if
requested).

basic usage is to fill in these blanks at the top of the file. Even just filling
the ``target_file`` is enough to get started:

.. code:: python

    name = "object_name"  # used for saving files
    target_file = "<required>.fits"  # can be a numpy array instead
    mask_file = None  # "<path to mask>.fits" # can be a numpy array instead
    psf_file = None  # "<path to psf>.fits" # can be a numpy array instead
    variance_file = None  # "<path to variance>.fits" # or numpy array or "auto"
    pixelscale = 0.1  # arcsec/pixel
    zeropoint = 22.5  # mag
    initial_params = None  # e.g. {"center": [3, 3], "q": {"value": 0.8, "locked": True}}
    window = None  # None to fit whole image, otherwise ((xmin,xmax),(ymin,ymax)) pixels
    initial_sky = None  # If None, sky will be estimated
    sky_locked = False
    model_type = "sersic galaxy model"

then run the script from the command line as a python file:

.. code:: bash

    >>> python single_model_fit.py

This will output the fitted parameters and save the model and residual images as
fits files. See the :doc:`tutorials/GettingStarted` tutorial for more
information.

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
