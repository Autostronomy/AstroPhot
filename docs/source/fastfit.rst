============================
Fit Fast! with AstroPhot CLI
============================

Sometimes you just have a FITS file and want to quickly fit a simple model to
it. The AstroPhot Command-Line-Interface (CLI) is designed for this purpose.
Installing AstroPhot with ``pip`` adds the ``astrophot`` command to your
terminal, which has two sub-commands:

``astrophot single``
    Fit one model (plus a sky level) to one object in an image.

``astrophot segmap``
    Fit every object identified in a segmentation map, one after the other.

Both take FITS files as input and write the fitted parameters to a yaml file,
along with FITS images of the model and residuals. Neither needs any Python to
be written. This page first walks through the simplest use of each command and
then covers all of the options. A summary of the options is always available
with ``--help``:

.. code:: bash

    ~$ astrophot --help
    ~$ astrophot single --help
    ~$ astrophot segmap --help

The first lists the sub-commands, the other two list the options of each.
``astrophot --version`` prints the version of AstroPhot that is installed.

If you outgrow the commands, ``--dump`` gives you the script behind them as a
starting point for your own analysis, see :ref:`cli-dump`.

Fit a single object: ``astrophot single``
=========================================

The simplest case
-----------------

All that is required is a FITS file with an image of your object:

.. code:: bash

    ~$ astrophot single target_file.fits

This fits a Sersic model and a flat sky level to the whole image. It will:

#. read the image from the first HDU of ``target_file.fits``, and the WCS from
   its header;
#. choose starting values for all of the parameters automatically;
#. fit the Sersic and sky together with the Levenberg-Marquardt method, printing
   the :math:`\chi^2` at each step;
#. print the fitted model and its total flux;
#. write the results to the current directory.

With no other information AstroPhot assumes a variance of 1 for every pixel and
no PSF. This is almost certainly not what you want for a final result (the
parameter uncertainties in particular are meaningless without a real variance),
but it is a good way to check that everything is working. To try it right now,
any image from the tutorials will do, for example `this one
<https://github.com/Autostronomy/AstroPhot/raw/main/docs/source/tutorials/target_image.fits>`_.

The output files
----------------

Four files are written, each beginning with the model name, which is
``astrophot_model`` unless you change it with ``--name``:

.. list-table:: Files written by ``astrophot single``. "File" is the file name
    for the default ``--name``, "Contents" is what it holds and "Turn off with"
    is the option which stops it being written.
    :header-rows: 1
    :widths: 35 45 20

    * - File
      - Contents
      - Turn off with
    * - ``astrophot_model_parameters.yaml``
      - Fitted parameters and their uncertainties, the total flux, and the sky
      - (always written)
    * - ``astrophot_model_model_image.fits``
      - The fitted model (object plus sky) in the fitting window
      - ``--no_save_images``
    * - ``astrophot_model_residual_image.fits``
      - The image minus the model in the fitting window
      - ``--no_save_images``
    * - ``astrophot_model_covariance_matrix.npy``
      - Covariance matrix of all the fitted parameters (numpy format)
      - ``--no_save_cov``

The yaml file is plain text and looks like this (shortened, with the numbers
rounded):

.. code:: yaml

    astrophot_model:
      message: success
      model_type: sersic galaxy model
      note: Total flux/mag is within the fitting window, not extended to infinity
      parameters:
        Re:
          dynamic: true
          uncertainty: 2.4076
          value: 16.2871
        center:
          dynamic: true
          uncertainty:
          - 0.1429
          - 0.1962
          value:
          - -1.0562
          - 3.5270
        n:
          dynamic: true
          uncertainty: 0.2017
          value: 1.7684
        ...
      total_flux: 4317.03
      total_flux_uncertainty: 1630.70
      total_magnitude: 13.412
      total_magnitude_uncertainty: 0.410
    sky_model:
      model_type: flat sky model
      parameters:
        I0:
          dynamic: true
          uncertainty: 0.0255
          value: 0.0184
        ...

Some things to know when reading it:

- ``message`` is the fitter's report of how the fit ended. Anything other than
  ``success`` deserves a look at the residual image.
- Each parameter has a ``value``, an ``uncertainty`` (one sigma, from the
  covariance matrix) and ``dynamic``, which is ``true`` if the parameter was
  fit and ``false`` if it was held fixed. Fixed parameters have no uncertainty.
- Lengths and positions are in arcsec, angles in radians, and surface
  brightnesses in flux per arcsec\ :sup:`2`. The ``center`` is measured on the
  tangent plane from the reference coordinate of the WCS (``CRVAL``), see
  :doc:`coordinates`. If the image has no WCS the position is measured from the
  centre of the lower left pixel instead, see :ref:`cli-coordinates`.
- ``total_flux`` is the flux of the object model (without the sky) summed
  inside the fitting window. It is not extrapolated to infinity.
  ``total_magnitude`` is only filled in if you give a ``--zeropoint``.
- The covariance matrix has one row per fitted parameter, sky parameters first
  and then the object parameters in the order they are printed at the end of the
  fit (for the default models: ``I0``, ``center`` x and y, ``q``, ``PA``,
  ``n``, ``Re``, ``Ie``). Load it with ``numpy.load``.

A realistic fit
---------------

For real measurements you should provide what you know about the image:

.. code:: bash

    ~$ astrophot single target_file.fits --psf psf.fits --variance variance.fits --mask mask.fits --zeropoint 22.5 --window 10 100 10 100

Here is what each of those does.

``--psf psf.fits``
    An image of the point spread function. The model is convolved with it
    before being compared with the data. The PSF image must have an odd number
    of pixels on each side, so that there is a central pixel, and should be
    normalized to a sum of 1. If the PSF is sampled more finely than the image,
    give the factor with ``--psf_upsample``, for example ``--psf_upsample 2``
    for PSF pixels half the size of the image pixels.

``--variance variance.fits``
    An image of the variance of each pixel, the same shape as the target. Note
    that this is the variance, not the standard deviation (sigma) and not the
    inverse variance (weight). Pixels with a zero, negative or NaN variance are
    masked. If you have no variance image, ``--variance auto`` asks AstroPhot
    to estimate one from the image itself. This is better than nothing, but a
    real variance map is always preferred.

``--mask mask.fits``
    An image marking the pixels to ignore, the same shape as the target. Any
    pixel which is not zero in the mask is ignored in the fit. NaN pixels in
    the target are always ignored.

``--zeropoint 22.5``
    The magnitude zeropoint of the image, so that
    ``magnitude = -2.5 log10(flux) + zeropoint``. With it the total magnitude
    is reported as well as the total flux.

``--window 10 100 10 100``
    Fit only part of the image, given as ``imin imax jmin jmax`` in pixels.
    ``i`` is the horizontal axis of the image as shown in e.g. DS9 (FITS axis
    1) and ``j`` the vertical axis (FITS axis 2). Pixels are counted from 0
    and the upper limits are not included, so this example fits a 90 by 90
    pixel region. A window makes the fit faster and keeps other objects out of
    it. The model and residual images cover only the window.

If any of the images is not in the first HDU of its file, select the HDU with
``--target_hdu``, ``--psf_hdu``, ``--variance_hdu`` or ``--mask_hdu``. These
take either the HDU number (counting from 0) or its name, for example
``--target_hdu 1`` or ``--target_hdu SCI``.

Choosing the model
------------------

The model is set with ``--model_type``. Any AstroPhot model type can be used,
written with underscores in place of the spaces:

.. code:: bash

    ~$ astrophot single target_file.fits --model_type exponential_galaxy_model

Some common choices are listed below, the :doc:`tutorials/ModelZoo` tutorial
shows all of them along with their parameters.

.. list-table:: Some model types for ``--model_type``. "Model type" is the
    value to give, "Describes" is the kind of object it represents and
    "Parameters" are the names of its parameters, which can be used to set
    initial values.
    :header-rows: 1
    :widths: 32 38 30

    * - Model type
      - Describes
      - Parameters
    * - ``sersic_galaxy_model`` (default)
      - Elliptical Sersic profile
      - ``center``, ``q``, ``PA``, ``n``, ``Re``, ``Ie``
    * - ``exponential_galaxy_model``
      - Elliptical exponential disk
      - ``center``, ``q``, ``PA``, ``Re``, ``Ie``
    * - ``gaussian_galaxy_model``
      - Elliptical Gaussian
      - ``center``, ``q``, ``PA``, ``sigma``, ``flux``
    * - ``point_model``
      - A star or other point source (requires ``--psf``)
      - ``center``, ``flux``

Here ``q`` is the axis ratio (b/a, between 0 and 1), ``PA`` the position angle
in radians, and ``center`` a pair of numbers.

The sky is always fit along with the object. By default it is a constant level
(``--sky_model_type flat``), use ``--sky_model_type plane`` for a sky with a
gradient. ``--initial_sky`` sets the starting value of the sky level, in flux
per arcsec\ :sup:`2`, and ``--sky_locked`` holds the sky fixed during the fit.
For an image which is already sky subtracted you would use:

.. code:: bash

    ~$ astrophot single target_file.fits --initial_sky 0 --sky_locked

Initial values and fixed parameters
-----------------------------------

AstroPhot picks starting values for the parameters automatically, which works
well for an isolated object near the middle of the window. If the fit lands in
the wrong place you can give starting values yourself. Any option which is not
one of the command's own options is passed on to the model, so a parameter is
set with its name:

.. code:: bash

    ~$ astrophot single target_file.fits --n 2 --q 0.7 --center 12.1 -3.4

Parameters with more than one value, like ``center``, take their values
separated by spaces. The values are in the units given in the yaml file, so
``center`` is in arcsec from the reference coordinate of the WCS. An easy way to
find the right numbers is to run once with no initial values and look at the
output.

To hold a parameter fixed rather than fit it, give a dictionary with its value
and ``'dynamic': False`` (the quotes around the whole dictionary are needed).
For example, a de Vaucouleurs profile is a Sersic with ``n`` fixed at 4:

.. code:: bash

    ~$ astrophot single target_file.fits --n "{'value': 4, 'dynamic': False}"

The same mechanism sets other options of the model, for example ``--psf_convolve
False`` to skip the PSF convolution or ``--sampling_mode midpoint`` to change
how the model is integrated over each pixel.

.. note::

    The target file must come before any model parameters on the command line,
    since everything after an option like ``--center`` is taken to be one of
    its values. Writing ``--center=12.1`` style options avoids this.

.. _cli-coordinates:

Images without a WCS
--------------------

If the FITS header has a WCS it is used, and all positions are relative to its
reference coordinate. If the header has no WCS, AstroPhot uses a pixel scale of
1 and measures positions from the centre of the lower left pixel, so that
``center`` is simply the pixel position (counting from 0) and ``Re`` is in
pixels.

Giving ``--pixelscale`` sets the pixel size in arcsec and ignores any WCS in the
header. Positions are then measured from the centre of the lower left pixel in
arcsec. ``--pixelscale 1`` is a convenient way to work in pixel units even for
an image which has a WCS.

.. note::

    Only the linear part of a tangent plane (``TAN``) WCS is used. Distortion
    terms such as SIP are ignored by the commands, which is rarely a problem
    over a small cutout. See the :doc:`tutorials/ImageTypes` tutorial for
    images with distortion.

Naming and placing the output
-----------------------------

``--name`` sets the name of the model and the start of every output file name.
It may only contain letters, digits and underscores. ``--outdir`` sets the
directory the files are written to, which is created if it does not exist:

.. code:: bash

    ~$ astrophot single ngc1234_r.fits --name ngc1234_r --outdir results/ngc1234

writes ``results/ngc1234/ngc1234_r_parameters.yaml`` and so on. Existing files
with the same names are overwritten.

``--verbose 0`` runs silently, and ``--verbose 2`` also prints debugging
messages. To keep a record of the run give a file with ``--log``:

.. code:: bash

    ~$ astrophot single ngc1234_r.fits --log ngc1234_r.log

The log has the messages that ``--verbose 1`` prints, or more with a higher
``--verbose``, each with its time. It is also written with ``--verbose 0``, when
nothing is printed. Each run starts with a line giving the version of AstroPhot
and the backend that was used, and is added to the end of the file, so one log
can be shared by many runs.

All options of ``astrophot single``
-----------------------------------

.. list-table:: Options of ``astrophot single``. "Option" is the option and
    the values it takes, "Default" is what is used when the option is not given
    and "Meaning" describes it.
    :header-rows: 1
    :widths: 32 16 52

    * - Option
      - Default
      - Meaning
    * - ``target_file``
      - (required)
      - FITS file with the image to fit. Must come first
    * - ``--model_type TYPE``
      - ``sersic_galaxy_model``
      - Model to fit, with underscores for spaces
    * - ``--<parameter> VALUE``
      - automatic
      - Initial value of any model parameter, e.g. ``--n 2``
    * - ``--window IMIN IMAX JMIN JMAX``
      - whole image
      - Region to fit, in pixels counted from 0, upper limits not included
    * - ``--psf FILE``
      - no PSF
      - FITS file with the PSF image
    * - ``--psf_upsample N``
      - 1
      - How many PSF pixels span one image pixel
    * - ``--variance FILE``
      - variance of 1
      - FITS file with the variance image, or ``auto`` to estimate it
    * - ``--mask FILE``
      - no mask
      - FITS file with the mask, non-zero pixels are ignored
    * - ``--zeropoint ZP``
      - none
      - Magnitude zeropoint, needed for magnitudes to be reported
    * - ``--pixelscale SCALE``
      - from the WCS
      - Pixel size in arcsec, replaces the WCS of the header
    * - ``--sky_model_type TYPE``
      - ``flat``
      - Sky model, ``flat`` or ``plane``
    * - ``--initial_sky VALUE``
      - automatic
      - Initial sky level in flux per arcsec\ :sup:`2`
    * - ``--sky_locked``
      - sky is fit
      - Hold the sky fixed
    * - ``--target_hdu``, ``--psf_hdu``, ``--variance_hdu``, ``--mask_hdu``
      - 0
      - HDU number or name holding each image
    * - ``--name NAME``
      - ``astrophot_model``
      - Name of the model and prefix of the output files
    * - ``--outdir DIR``
      - current directory
      - Directory for the output files
    * - ``--no_save_images``
      - images saved
      - Do not write the model and residual images
    * - ``--no_save_cov``
      - matrix saved
      - Do not write the covariance matrix
    * - ``--verbose LEVEL``
      - 1
      - Amount of printed output: 0 for none, 1 for the progress of the fit, 2
        to add debugging messages
    * - ``--log FILE``
      - none
      - Also write the messages to this file
    * - ``--dump``
      - off
      - Write the script to ``single_astrophot_model.py``

Fit all objects in an image: ``astrophot segmap``
=================================================

This command fits every object in an image with the same type of model. It
works like ``astrophot single``, but instead of one fit it loops over the
objects identified in a segmentation map. Each object is fit on its own, with
its own sky level, in a window around its segment. Pixels belonging to other
segments inside that window are masked.

Note that this is not a joint fit, the objects are not fit simultaneously. If
your objects overlap enough that they need to be modelled together, see the
:doc:`tutorials/GroupModels` tutorial.

The segmentation map
--------------------

A segmentation map is an image of integers, the same shape as the target, in
which every pixel holds the id of the object it belongs to and 0 marks the
background. Many tools make them, such as Source Extractor or `photutils
<https://photutils.readthedocs.io/en/stable/user_guide/segmentation.html>`_.
With photutils a basic one takes a few lines of Python:

.. code:: python

    from astropy.io import fits
    from photutils.segmentation import detect_sources, deblend_sources

    data = fits.getdata("target_file.fits")
    # threshold in the units of the image, minimum number of connected pixels
    segmap = detect_sources(data, 0.02, 6)
    segmap = deblend_sources(data, segmap, 5)
    fits.writeto("segmap_file.fits", segmap.data, overwrite=True)

The simplest case
-----------------

.. code:: bash

    ~$ astrophot segmap target_file.fits segmap_file.fits

This fits a Sersic model and a flat sky to each object in the segmentation
map, starting each one from the position, axis ratio and position angle of its
segment. As for the single object command, this assumes a variance of 1 and no
PSF, so it is a good check that everything works but is not likely to be what
you want in the end.

Three files are written, again starting with the ``--name`` (default
``astrophot_model``):

.. list-table:: Files written by ``astrophot segmap``. "File" is the file name
    for the default ``--name``, "Contents" is what it holds and "Turn off with"
    is the option which stops it being written.
    :header-rows: 1
    :widths: 35 45 20

    * - File
      - Contents
      - Turn off with
    * - ``astrophot_model_parameters.yaml``
      - Fitted parameters, uncertainties and total flux of every object
      - (always written)
    * - ``astrophot_model_model_images.fits`` and
        ``astrophot_model_residual_images.fits``
      - The model and residual of every object, one FITS extension each
      - ``--no_save_images``
    * - ``astrophot_model_covariance_matrix.npz``
      - Covariance matrix of every fit (numpy archive)
      - ``--no_save_cov``

Every object is labelled with the name and its segmentation map id, for example
``astrophot_model_3`` for id 3. That label is its entry in the yaml file and
its key in the covariance archive, and its images are in the extensions named
``astrophot_model_3_model`` and ``astrophot_model_3_residual``. Each image
extension has a header with the WCS of its window, so it can be compared
directly with the original image. Reading the results back in Python looks
like:

.. code:: python

    import yaml
    import numpy as np
    from astropy.io import fits

    with open("astrophot_model_parameters.yaml") as f:
        results = yaml.safe_load(f)
    print(results["astrophot_model_3"]["parameters"]["Re"]["value"])

    covariance = np.load("astrophot_model_covariance_matrix.npz")["astrophot_model_3"]
    model_image = fits.getdata(
        "astrophot_model_model_images.fits", "astrophot_model_3_model"
    )

The entry for each object has its ``parameters``, its ``sky_parameters``,
``total_flux`` (and ``total_magnitude`` if a zeropoint was given) and the
``message`` from the fitter, in the same format as for ``astrophot single``.

If the fit of an object fails outright, for example because its segment has no
flux, the remaining objects are still fit. The entry of the failed object then
holds only a ``message`` beginning with ``fit failed`` and the reason.

A realistic fit
---------------

.. code:: bash

    ~$ astrophot segmap target_file.fits segmap_file.fits --psf psf.fits --variance variance.fits --mask mask.fits --zeropoint 22.5 --window_expand_border 5

The PSF, variance, mask and zeropoint are exactly as described for
``astrophot single`` above, and apply to the whole image. The mask you provide
is combined with the masking of neighbouring segments for each object.

Fitting windows
---------------

Each object is fit inside a window, which starts as the bounding box of its
segment. Segments usually only include the bright part of an object, so it is a
good idea to expand the windows to give the model and the sky some room:

``--window_expand_scale 1.5``
    Scale the width and height of every window by this factor about its centre.

``--window_expand_border 5``
    Then add this many pixels to every side of every window.

Windows are cut off at the edge of the image. Larger windows constrain the sky
and the outskirts of the object better, but are slower to fit.

Choosing which objects to fit
-----------------------------

``--filter_ids 4 8 15``
    Skip the objects with these segmentation map ids. Their pixels are still
    masked in the fits of the other objects. Id 0 is the background and is
    never fit.

``--window_min_size 25``
    Skip objects whose bounding box (before it is expanded) has an area of
    fewer than this many pixels. Useful for dropping tiny detections.

Initial parameters from a catalogue
-----------------------------------

The starting ``center``, ``q`` and ``PA`` of each object come from its segment
and the other parameters are chosen automatically. For more control, pass a
catalogue of initial parameters with ``--cat catalogue.yaml``. This is a yaml
file in which each key is a segmentation map id and each value holds the
parameters to set for that object. Only the objects and parameters that you
want to change need to be listed:

.. code:: yaml

    1:
      center: [10.2, 13.5]
      q:
        value: 0.8
        dynamic: false
    2:
      n: 4
      Re: 3.1

Here object 1 starts at the given centre and has its axis ratio held fixed at
0.8 (``dynamic: false`` means the parameter is not fit), while object 2 starts
with a Sersic index of 4 and an effective radius of 3.1 arcsec. All other
objects use the automatic values. The values are in the same units as the
output, so a ``center`` is in arcsec from the reference coordinate of the WCS,
or from the lower left pixel if there is no WCS or ``--pixelscale`` is given
(see :ref:`cli-coordinates`).

All options of ``astrophot segmap``
-----------------------------------

.. list-table:: Options of ``astrophot segmap``. "Option" is the option and
    the values it takes, "Default" is what is used when the option is not given
    and "Meaning" describes it. Options marked "as single" behave as described
    for ``astrophot single``.
    :header-rows: 1
    :widths: 32 16 52

    * - Option
      - Default
      - Meaning
    * - ``target_file``
      - (required)
      - FITS file with the image to fit
    * - ``segmap_file``
      - (required)
      - FITS file with the segmentation map
    * - ``--segmap_hdu``
      - 0
      - HDU number or name holding the segmentation map
    * - ``--cat FILE``
      - none
      - yaml catalogue of initial parameters for each id
    * - ``--window_expand_scale FACTOR``
      - 1
      - Scale every window by this factor
    * - ``--window_expand_border N``
      - 0
      - Add this many pixels to each side of every window
    * - ``--window_min_size AREA``
      - 1
      - Smallest bounding box area, in pixels, for an object to be fit
    * - ``--filter_ids ID [ID ...]``
      - none
      - Ids to skip, 0 is always skipped
    * - ``--model_type``, ``--psf``, ``--psf_upsample``, ``--variance``,
        ``--mask``, ``--zeropoint``, ``--pixelscale``
      - as single
      - As single
    * - ``--sky_model_type``, ``--initial_sky``, ``--sky_locked``
      - as single
      - As single, applied to the sky of every object
    * - ``--target_hdu``, ``--psf_hdu``, ``--variance_hdu``, ``--mask_hdu``
      - 0
      - As single
    * - ``--name NAME``
      - ``astrophot_model``
      - Prefix of the object labels and of the output files
    * - ``--outdir``, ``--no_save_images``, ``--no_save_cov``
      - as single
      - As single
    * - ``--verbose LEVEL``
      - 1
      - 0 for no output, 1 to name each object as it is fit, 2 to also print
        the progress of every fit
    * - ``--log FILE``
      - none
      - As single
    * - ``--dump``
      - off
      - Write the script to ``segmap_astrophot_model.py``

Unlike ``astrophot single``, model parameters cannot be given on the command
line since they would apply to every object. Use the catalogue instead.

.. _cli-dump:

Going further with ``--dump``
=============================

The commands are deliberately simple. When you need something they do not
offer, such as a second component, a different fitting method or some plots,
``--dump`` writes a copy of the Python script behind the command to the current
directory:

.. code:: bash

    ~$ astrophot single --dump
    ~$ astrophot segmap --dump

This creates ``single_astrophot_model.py`` or ``segmap_astrophot_model.py``.
Each is a single self-contained file which takes exactly the same arguments as
the command, so you can edit it and run it in the same way:

.. code:: bash

    ~$ python single_astrophot_model.py target_file.fits --psf psf.fits

If ``--dump`` is given along with a target file, the script is written and the
fit is run as well. The :doc:`tutorials/GettingStarted` tutorial explains the
AstroPhot objects used in the scripts.

A note on speed
===============

Every run of a command first has to load AstroPhot and PyTorch, which takes a
few seconds before the fit even starts. That is unnoticeable for one fit, but
adds up if you call ``astrophot single`` in a loop over hundreds of images. In
that case it is better to loop inside Python, starting from the ``--dump``
script, so that the loading only happens once. For many objects in the same
image use ``astrophot segmap``.

If you only run on a CPU, installing the CPU-only build of PyTorch before
AstroPhot makes the start-up a little quicker and the installation much
smaller. See the `PyTorch installation page
<https://pytorch.org/get-started/locally/>`_ for the command for your system.
