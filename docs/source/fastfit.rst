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
calculations. See the :doc:`tutorials/GettingStarted` tutorial or use ``--help``
for more information on these inputs.

The final output will be a yaml file with the fitted parameters. Depending on
the options you choose, it may also output fits files for the model and residual
images as well as a numpy file with a covariance matrix of uncertainties for the
parameters.

**Note**: Unfortunately, this script is really slowed down by the fact that
PyTorch needs to be loaded for each run, which can take several seconds. For
faster bulk processing see the section below.

Fit all objects in an image
---------------------------

This script will fit all objects in an image with a single model type. It
functions similarly to the single object fitting script, but instead of fitting
a single model it will loop through and fit everything identified in a
segmentation map. Note that this is not a group model fitting script. So each
segmentation will be fit separately rather than as a joint likelihood. If you
wish to account for blending and fit a group model, see the
:doc:`tutorials/GroupModels` tutorial.

The simplest use case is just:

.. code:: bash

    ~$ astrophot_segmap target_file.fits segmap_file.fits

Just like in the single object case, this is not likely to be what you want, but
it is a good starting point to check that everything is working. A more
realistic example is:

.. code:: bash

    ~$ astrophot_segmap target_file.fits segmap_file.fits --psf psf.fits --mask mask.fits --variance variance.fits --window_expand_border 5 --zeropoint 22.5

In this version we pass FITS files for the PSF, mask, and variance, we expand
the fitting windows by 5 pixels in each direction, and we set the zeropoint for
magnitude calculations. See the :doc:`tutorials/GettingStarted` tutorial or use
``--help`` for more information on these inputs.

If you are looking for a bit more control, you can also pass a catalogue of
initial parameters for each object using ``--cat catalogue.yaml``. This is a
yaml file with keys corresponding to the segmentation ids and values that are
dictionaries of parameters to override the default initial parameters. For
example:

.. code:: yaml

    1:
      center: [10, 10]
      q:
        value: 0.8
        locked: true
    2:
      center: [20, 20]
      q: 0.6

The final output will be a yaml file with the fitted parameters for each object.
Depending on your configuration, the covariance matrix of uncertainties and fits
files for the model and residuals may also be included.
