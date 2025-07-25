===========
Coordinates
===========

Coordinate systems in astronomy can be complicated, AstroPhot is no
different. Here we explain how coordinate systems are handled to help
you avoid possible pitfalls.

For the most part, AstroPhot follows the FITS standard for coordinates, though
limited to the types of images that AstroPhot can model.

Three Coordinate Systems
------------------------

There are three coordinate systems to think about.

#. ``world`` coordinates are the classic (RA, DEC) that many
   astronomical sources are represented in. These should always be
   used in degree units as far as AstroPhot is concerned.
#. ``plane`` coordinates are the tangent plane on which AstroPhot performs its
   calculations. Working on a plane makes everything linear and does not
   introduce a noticeable projection effect for small enough images. In the
   tangent plane everything should be represented in arcsecond units.
#. ``pixel`` coordinates are specific to each image, they start at (0,0) in the
   center of the [0,0] indexed pixel. These are effectively unitless, a step of
   1 in pixel coordinates is the same as changing an index by 1. AstroPhot
   adopts an indexing scheme standard to FITS files meaning the pixel coordinate
   (5,9) corresponds to the pixel indexed at [5,9]. Normally for numpy arrays
   and PyTorch tensors, the indexing would be flipped as [9,5] so AstroPhot
   applies a transpose on any image it receives in an Image object.  Also, in
   the pixel coordinate system the values are represented by floating point
   numbers, so (1.3,2.8) is a valid pixel coordinate that is just partway
   between pixel centers.

Transformations exist in AstroPhot for converting ``world`` to/from
``plane`` and for converting ``plane`` to/from ``pixel``. The best way
to interface with these is to use the ``image.world_to_plane``
for any AstroPhot image object (you may similarly swap ``world``,
``plane``, and ``pixel``).

One gotcha to keep in mind with regards to ``world_to_plane`` and
``plane_to_world`` is that AstroPhot needs to know the reference (RA, DEC) where
the tangent plane meets with the celestial sphere. AstroPhot now adopts the FITS
standard for this using ``image.crval`` to store the reference world
coordinates. Note that if you are doing simultaneous multi-image analysis you
should ensure that the ``crval`` is same for all images!

Projection Systems
------------------

AstroPhot currently only supports the Gnomonic projection system. This means
that the tangent plane is defined as "contacting" the celestial sphere at a
single point, the reference (crval) coordinates. The tangent plane coordinates
correspond to the world coordinates as viewed from the center of the celestial
sphere. This is the most common projection system used in astronomy and commonly
used in the FITS standard. It is also the one that Astropy usually uses for its
WCS objects.

Coordinate reference points
---------------------------

There are three coordinate systems in AstroPhot: ``world``, ``plane``, and
``pixel``. AstroPhot tracks a reference point in each coordinate system used to
connect each system. Below is a summary of the reference coordinates and their
meaning:

#. ``crval`` world coordinates on the celestial sphere (RA, DEC in degrees)
   where the tangent plane makes contact. crval always contacts the tangent
   plane at (0,0) in the tangent plane coordinates. This should be the same for
   every image in a multi-image analysis.
#. ``crtan`` tangent plane coordinates (arcsec) where the pixel grid makes
   contact with the tangent plane. This is the pivot point about which the
   pixelscale matrix operates, therefore if the pixelscale matrix defines a
   rotation then this is the coordinate about which the rotation will be
   performed. This may be different for each image in a multi-image analysis.
#. ``crpix`` pixel coordinates where the pixel grid makes contact with the
   tangent plane. One may think of the referenced pixel location as being
   "pinned" to the tangent plane. This may be different for each image in a
   multi-image analysis.

Thinking of the celestial sphere, tangent plane, and pixel grid as three
interconnected coordinate systems is crucial for understanding how AstroPhot
operates in a multi-image context. While the transformations may get
complicated, try to remember these contact points:

* ``crval`` is in the world coordinates and contacts the tangent plane at
  (0,0) in the tangent plane coordinates.
* ``crtan`` is in the tangent plane coordinates and contacts the pixel grid at
  ``crpix`` in the pixel coordinates.

What parts go where?
--------------------

Since AstroPhot works in multiple reference frames it can be easy to get lost.
Keep these basics in mind. The world coordinates are where catalogues exist, so
this is the coordinate system you should use when interfacing with external
resources. The tangent plane coordinates are where the models exist. So when
creating a model and considering factors like the position angle, you should
think in the tangent plane coordinates. The pixel coordinates are where the data
exists. So when you create a TargetImage object it is in pixel coordinates, but
so too is a ModelImage object since it is intended to be compared against a
TargetImage. This means that any distortions in the TargetImage (i.e. SIP
distortions) will show up in the ModelImage, but aren't actually part of the
model. This can manifest for example as a round Gaussian model looking
elliptical in its ModelImage because there is a skew in the CD matrix in the
TargetImage it is matching. In general this is a good thing because we care
about how our models look on the sky (tangent plane), not strictly how they look
in the pixel grid.
