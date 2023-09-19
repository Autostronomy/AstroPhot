===========
Coordinates
===========

Coordinate systems in astronomy can be complicated, AstroPhot is no
different. Here we explain how coordinate systems are handled to help
you avoid possible pitfalls.

Basics
------

There are three main coordinate systems to think about.

#. ``world`` coordinates are the classic (RA, DEC) that many
   astronomical sources are represented in. These should always be
   used in degree units as far as AstroPhot is concerned.
#. ``plane`` coordinates are the tangent plane on which AstroPhot
   performs it's calculations. Working on a plane makes everything
   linear and does not introduce a noticible effect for small enough
   images. In the tangent plane everything should be represented in
   arcsecond units.
#. ``pixel`` coordinates are specific to each image, they start at
   (0,0) in the center of the [0,0] indexed pixel. These are
   effectively unitless, a step of 1 in pixel coordinates is the same
   as changing an index by 1. Note however that in the pixel
   coordinate system the values are represented by floating point
   numbers and so (1.3,2.8) is a valid pixel coordinate that is just
   partway between pixel centers.

Tranformations exist in AstroPhot for converting ``world`` to/from
``plane`` and for converting ``plane`` to/from ``pixel``. The best way
to interface with these is to use the ``image.header.world_to_plane``
for any AstroPhot image object (you may similarly swap ``world``,
``plane``, and ``pixel``).

One gotcha to keep in mind with regards to ``world_to_plane`` and
``plane_to_world`` is that AstroPhot needs to know the reference
(RA_0, DEC_0) where the tangent plane meets with the celestial
sphere. You can set this by including ``reference_radec = (RA_0,
DEC_0)`` as an argument in an image you create.  If a reference is not
given, then one will be assumed based on avaialble information. Note
that if you are doing multiband analysis you should ensure all the
projection parameters are the same for all images!

Projection Systems
------------------

AstroPhot currently implements three coordinate reference systems:
Gnomonic, Orthographic, and Steriographic. The default projection is
the Gnomonic, which represents the perspective of an observer at the
center of a sphere projected onto a plane. For the exact
implementation by AstroPhot see the `Wolfram MathWorld
<https://mathworld.wolfram.com/GnomonicProjection.html>`_ page.

On small scales the choice of projection doesn't matter. For very
large images the effect may be detectable, though it is likely
insignificant compared to other effects in an image. Just like the
``reference_radec`` you can choose your projection system in an image
you construct by passing ``projection = 'gnomonic'`` as an
argument. Just like with the reference coordinate, for images to talk
to each other they should have the same projection.

If you really want to change the projection after an image has
been created (warning, this may cause serious missalignments between
images), you can force it to update with::

  image.header.projection = 'steriographic'

which would change the projection to steriographic. The image won't
recompute its position in the new projection system, it will just use
new equations going forward. Hence the potential to seriously mess up
your image alignmnt if this is done after some calculations have
already been performed.

Talking to the world
--------------------

If you have images with WCS information (likely Astropy WCS) then you
will want to use this to map images onto the same tangent plane. That
is mostly described in the basics section, however there are some more
features you can take advantage of. When creating an image in
AstroPhot, you need to tell it some basic properties so that the image
knows how to place itself in the tangent plane. Using the WCS object
you should be able to recover the coordinates of the image in (RA,
DEC), for a 10 pixel by 10 pixel image you could accomplish this by
doing something like::

  center = wcs.pixel_to_world(5,5)
  ra, dec = center.ra.deg, center.dec.deg

meaning that you know the world position of the image. To have
AstroPhot place the image at the right location in the tangent plane
you can use the ``center_radec`` argument when constructing the image
which would look something like::

  image = ap.image.Target_Image(
      data = data,
      pixelscale = pixelscale,
      center_radec = (ra, dec),
  )

AstroPhot will set the reference RA, DEC to these center
coordinates. A more explicit alternative is to just say what the
reference coordiante should be. That would look something like::
  
  image = ap.image.Target_Image(
      data = data,
      pixelscale = pixelscale,
      center_radec = (ra, dec),
      reference_radec = (ra,dec),
  )

which will override anything else and set the reference coordinate for
this image. Now ``center_radec`` could be anything and the world
coordinate reference would be ``(ra,dec)``.

You may also have a catalogue of objects that you would like to
project into the image. The easiest way to do this if you already have
an image object is to call the ``world_to_plane`` functions
manually. Say for example that you know the object position as an
Astropy ``SkyCoord`` object, and you want to use this to set the
center position of a sersic model. That would look like::

  model = ap.models.AstroPhot_Model(
      name = "knowloc",
      model_type = "sersic galaxy model",
      target = image,
      parameters = {
          "center": image.header.world_to_plane(obj_pos.ra.deg, obj_pos.dec.deg),
      }
  )

Which will start the object at the correct position in the image given
its world coordinates. As you can see, the ``center`` and in fact all
parameters for AstroPhot models are defined in the tangent plane. This
means that if you have optimized a model and you would like to present
it's position in world coordinates that can be compared with other
sources, you will need to do the opposite operation::

  world_position = image.header.plane_to_world(model["center"].value)

Which should be the coordinates in RA and DEC (degrees), assuming that
you initialized the image with a WCS or by other means ensured that
the world coordinates being used are correct. If you never gave
AstroPhot the information it needs, then it likely assumed a reference
position of (0,0) in the world coordinate system and so probably
doesn't represent your object.

Coordinate reference points
---------------------------

As stated earlier, there are essentially three coordinate systems in
AstroPhot: ``world``, ``plane``, and ``pixel``. To uniquely specify
the transformation from ``world`` to ``plane`` AstroPhot keeps track
of two vectors: ``reference_radec`` and ``reference_planexy``. These
variables are stored in all ``Image_Header`` objects and essentially
pin down the mapping such that one coordinate will get mapped to the
other. All other coordinates follow from the projection system
assumed. It is possible to specify these variables directly when
constructing an image, or implicitly if you give some other relevant
information (eg an Astropy WCS). AstroPhot also keeps track of two
more vectors: ``reference_imageij`` and ``reference_imagexy``. These
variables control where an image is placed in the tangent plane and
represent a fixed point between the pixel coordinates and the tangent
plane coordinates. If your pixel scale matrix includes a rotation then
the rotation will be performed about this position.

All together, these reference positions define how pixels are mapped
in AstroPhot. This level of generality is overkill for analyzing a
single image, so AstroPhot makes reasonable assumptions about these
reference points if you don't specify them all. This makes it easy to
do single image analysis without thinking too much about the
coordinate systems. However, for multi-band or multi-epoch imaging it
is critical to be absolutely clear about these coordinate
transformations so that images can be aligned properly on the sky. As
an intuitive explanation, think of ``reference_radec`` and
``reference_planexy`` as defining the coordinate system that is shared
between images, while ``reference_imageij`` and ``reference_imagexy``
specify where a single image is located. As such, in multi-image
analysis if you wish to use world coordinates, you should explitcitly
pass the same ``reference_radec`` and ``reference_planexy`` to every
image so that the same coordinate system is defined for all of them
(the same tangent plane at the same point on the celestial sphere). If
you aren't going to interact with world coordinates, you can ignore
those reference points entirely and it won't affect your images.

Below is a summary of the reference coordinates and their meaning:

#. ``reference_radec`` world coordinates on the celestial sphere (RA,
   DEC in degrees) where the tangent plane makes contact. This should
   be the same for every image in multi-image analysis.
#. ``reference_planexy`` tangent plane coordinates (arcsec) where it
   makes contact with the celesial sphere. This should typically be
   (0,0) though that is not enforced (it is assumed if not given).
   This reference coordinate should be the same for all images in
   multi-image analysis.
#. ``reference_imageij`` pixel coordinates about which the image is
   defined. For example in an Astropy WCS object the wcs.wcs.crpix
   array gives the pixel coordinate reference point for which the
   world coordinate mapping (wcs.wcs.crval) is defined. One may think
   of the referenced pixel location as being "pinned" to the tangent
   plane.
#. ``reference_imagexy`` tangent plane coordinates (arcsec) about
   which the image is defined. This is the pivot point about which the
   pixelscale matrix operates, therefore if the pixelscale matrix
   defines a rotation then this is the coordinate about which the
   rotation will be performed.
