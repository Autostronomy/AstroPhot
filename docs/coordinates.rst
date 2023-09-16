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
(RA_0, DEC_0) where the tangent plane coordinate (0,0) meets with the
celestial sphere world coordinate. You can set this by including
``reference_radec = (RA_0, DEC_0)`` as an argument in the first image
you create. Subsequent images will not modify the reference. If a
reference is not given, then one will be assumed based on avaialble
information when the first image is created.

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
``reference_radec`` you should choose your projection system in the
first image you construct by passing ``projection = 'gnomonic'`` as an
argument. Just like with the reference coordinate, subsequent image
creation cannot modify the projection.

If you really want to change the projection after the first image has
been created (warning, this may cause serious missalignments between
images), you can force it to update with::

  image.header.projection = 'steriographic'

which would change the projection to steriographic. This change can be
made with any image and it will affect all AstroPhot images
simultaneously. They won't recompute thier position in the new
projection system, they will just use new equations going
forward. Hence the potential to seriously mess up your image alignmnt
if this is done after some calculations have already been performed.

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

If this is the first image being constructed then AstroPhot will set
the reference RA, DEC to these center coordinates and future images
will project onto that tangent plane.

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

  world_position = image.header.plane_to_world(*model["center"].value)

Which should be the coordinates in RA and DEC, assuming that you
initialized the image with a WCS or by other means ensured that the
world coordinates being used are correct. If you never gave AstroPhot
the infomration it needs, then it likely assumed a reference position
of (0,0) in the world coordinate system and so probably doesn't
represent your object.


Why global projection parameters?
---------------------------------

Aren't global variables bad? Well, in this case they are the best way
to minimize potential surprises with regards to coordinate
systems. Transforming from the celestial sphere world coordinates to
tangent plane coordiantes is not a perfect operation, however it is
made infinitely worse if you are unwittingly projecting onto multiple
different tangent plane coordinate systems (say when doing
multi-band/multi-epoch analysis of multiple images). By forcing the
projection parameters to be global variables, all images may be
projected into the same tangent plane using world coordinates with
minimal chance to make mistakes along the way.

I would discourage trying to switch back and forth between two
coordinate systems in the same analysis pipeline. For the sake of your
sanity, and AstroPhot's you are better off splitting multiple image
analysis tasks into separate scripts. If the images are far enough
that they can't be on the same tangent plane, then they are probably
relatively simple to separate anyway.
