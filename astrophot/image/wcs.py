from abc import ABC
import torch
import numpy as np

from .. import AP_config

__all__ = ("WCS",)

deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi
rad_to_arcsec = rad_to_deg * 3600
arcsec_to_rad = deg_to_rad / 3600

class WCS:
    """Class to handle WCS interpretation in AstroPhot.

    AstroPhot performs it's operations on a tangent plane to the
    sphere, this class handles projections between the sphere and the
    tangent plane. It holds class variables for the reference (RA,DEC)
    where the tangent plane contacts the sphere, and the type of
    projection being performed. Note that (RA,DEC) coordinates should always be in degrees while the tangent plane is in arcsec coordinates.

    Attributes:
      reference_radec: The reference (RA,DEC) coordinates in degrees where the tangent plane contacts the sphere.
      projection: The projection system used to convert from (RA,DEC) onto the tangent plane. Should be one of: gnomonic (default), orthographic, steriographic

    """

    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-3
    
    default_reference_radec = (0,0)
    default_projection = "gnomonic"

    def __init__(self, *args, **kwargs):

        self.projection = kwargs.get("projection", self.default_projection)
        self.reference_radec = kwargs.get("reference_radec", self.default_reference_radec)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self.reference_radec = self.reference_radec.to(dtype=dtype, device=device)
        
    def world_to_plane(self, world_RA, world_DEC):
        if self.projection == "gnomonic":
            return self.world_to_plane_gnomonic(
                torch.as_tensor(
                    world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if self.projection == "orthographic":
            return self.world_to_plane_orthographic(
                torch.as_tensor(
                    world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if self.projection == "steriographic":
            return self.world_to_plane_steriographic(
                torch.as_tensor(
                    world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        raise ValueError(
            f"Unrecognized projection: {self.projection}. Should be one of: gnomonic, orthographic, steriographic"
        )

    def plane_to_world(self, plane_x, plane_y):
        if self.projection == "gnomonic":
            return self.plane_to_world_gnomonic(
                torch.as_tensor(
                    plane_x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    plane_y, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if self.projection == "orthographic":
            return self.plane_to_world_orthographic(
                torch.as_tensor(
                    plane_x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    plane_y, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if self.projection == "steriographic":
            return self.plane_to_world_steriographic(
                torch.as_tensor(
                    plane_x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    plane_y, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        raise ValueError(
            f"Unrecognized projection: {self.projection}. Should be one of: gnomonic, orthographic, steriographic"
        )

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, proj):
        assert proj in [
            "gnomonic",
            "orthographic",
            "steriographic",
        ], f"Unrecognized projection: {proj}. Should be one of: gnomonic, orthographic, steriographic"
        self._projection = proj

    @property
    def reference_radec(self):
        return self._reference_radec

    @reference_radec.setter
    def reference_radec(self, radec):
        self._reference_radec = torch.as_tensor(
            radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def _project_world_to_plane(self, world_RA, world_DEC):
        """
        Recurring core calculation in all the projections from world to plane.

        Args:
          world_RA: Right ascension in degrees
          world_DEC: Declination in degrees
        """
        return (
            torch.cos(world_DEC * deg_to_rad)
            * torch.sin((world_RA - self.reference_radec[0]) * deg_to_rad)
            * rad_to_arcsec,
            (
                torch.cos(self.reference_radec[1] * deg_to_rad)
                * torch.sin(world_DEC * deg_to_rad)
                - torch.sin(self.reference_radec[1] * deg_to_rad)
                * torch.cos(world_DEC * deg_to_rad)
                * torch.cos((world_RA - self.reference_radec[0]) * deg_to_rad)
            )
            * rad_to_arcsec,
        )

    def _project_plane_to_world(self, plane_x, plane_y, rho, c):
        """
        Recurring core calculation in all the projections from plane to world.

        Args:
          plane_x: tangent plane x coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.
          plane_y: tangent plane y coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.
          rho: polar radius in tangent plane.
          c: constant term dependent on the projection.
        """
        return (
            (
                self._reference_radec[0] * deg_to_rad
                + torch.arctan2(
                    plane_x * arcsec_to_rad * torch.sin(c),
                    rho * torch.cos(self.reference_radec[1] * deg_to_rad) * torch.cos(c)
                    - plane_y
                    * arcsec_to_rad
                    * torch.sin(self.reference_radec[1] * deg_to_rad)
                    * torch.sin(c),
                )
            )
            * rad_to_deg,
            torch.arcsin(
                torch.cos(c)
                * torch.sin(
                    self.reference_radec[1] * deg_to_rad
                )
                + plane_y
                * arcsec_to_rad
                * torch.sin(c)
                * torch.cos(self.reference_radec[1] * deg_to_rad)
                / rho
            )
            * rad_to_deg,
        )

    def world_to_plane_gnomonic(self, world_RA, world_DEC):
        """Gnomonic projection: (RA,DEC) to tangent plane.

        Performs Gnomonic projection of (RA,DEC) coordinates onto a
        tangent plane. The origin for the tangent plane is at the
        contact point. The tangent plane makes contact at the location
        of the `reference_radec` variable. In a gnomonic projection,
        great circles are mapped to straight lines. The gnomonic
        projection represents the image formed by a spherical lens,
        and is sometimes known as the rectilinear projection.

        Args:
          world_RA: Right ascension in degrees
          world_DEC: Declination in degrees

        See: https://mathworld.wolfram.com/GnomonicProjection.html

        """
        C = torch.sin(self.reference_radec[1] * deg_to_rad) * torch.sin(
            world_DEC * deg_to_rad
        ) + torch.cos(self.reference_radec[1] * deg_to_rad) * torch.cos(
            world_DEC * deg_to_rad
        ) * torch.cos(
            (world_RA - self.reference_radec[0]) * deg_to_rad
        )
        x, y = self._project_world_to_plane(world_RA, world_DEC)
        return x / C, y / C

    def plane_to_world_gnomonic(self, plane_x, plane_y):
        """Inverse Gnomonic projection: tangent plane to (RA,DEC).

        Performs the inverse Gnomonic projection of tangent plane
        coordinates into (RA,DEC) coordinates. The tangent plane makes
        contact at the location of the `reference_radec` variable. In
        a gnomonic projection, great circles are mapped to straight
        lines. The gnomonic projection represents the image formed by
        a spherical lens, and is sometimes known as the rectilinear
        projection.

        Args:
          plane_x: tangent plane x coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.
          plane_y: tangent plane y coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.

        See: https://mathworld.wolfram.com/GnomonicProjection.html

        """
        rho = (torch.sqrt(plane_x ** 2 + plane_y ** 2) + self.softening) * arcsec_to_rad
        c = torch.arctan(rho)

        ra, dec = self._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec

    def world_to_plane_steriographic(self, world_RA, world_DEC):
        """Steriographic projection: (RA,DEC) to tangent plane

        Performs Steriographic projection of (RA,DEC) coordinates onto
        a tangent plane. The origin for the tangent plane is at the
        contact point. The tangent plane makes contact at the location
        of the `reference_radec` variable. The steriographic
        projection preserves circles and angle measures.

        Args:
          world_RA: Right ascension in degrees
          world_DEC: Declination in degrees

        See: https://mathworld.wolfram.com/StereographicProjection.html

        """
        C = (
            1
            + torch.sin(world_DEC * deg_to_rad)
            * torch.sin(self._reference_radec[1] * deg_to_rad)
            + torch.cos(world_DEC * deg_to_rad)
            * torch.cos(self._reference_radec[1] * deg_to_rad)
            * torch.cos((world_RA - self._reference_radec[0]) * deg_to_rad)
        ) / 2
        x, y = self._project_world_to_plane(world_RA, world_DEC)
        return x / C, y / C

    def plane_to_world_steriographic(self, plane_x, plane_y):
        """Inverse Steriographic projection: tangent plane to (RA,DEC).

        Performs the inverse Steriographic projection of tangent plane
        coordinates into (RA,DEC) coordinates. The tangent plane makes
        contact at the location of the `reference_radec` variable. The
        steriographic projection preserves circles and angle measures.

        Args:
          plane_x: tangent plane x coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.
          plane_y: tangent plane y coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.

        See: https://mathworld.wolfram.com/StereographicProjection.html

        """
        rho = (torch.sqrt(plane_x ** 2 + plane_y ** 2) + self.softening) * arcsec_to_rad
        c = 2 * torch.arctan(rho / 2)
        ra, dec = self._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec

    def world_to_plane_orthographic(self, world_RA, world_DEC):
        """Orthographic projection: (RA,DEC) to tangent plane

        Performs Orthographic projection of (RA,DEC) coordinates onto
        a tangent plane. The origin for the tangent plane is at the
        contact point. The tangent plane makes contact at the location
        of the `reference_radec` variable. The point of perspective
        for the orthographic projection is at infinite distance. This
        projection is perhaps better suited to represent the view of
        an exoplanet, however it is included here for completeness.

        Args:
          world_RA: Right ascension in degrees
          world_DEC: Declination in degrees

        See: https://mathworld.wolfram.com/OrthographicProjection.html

        """
        x, y = self._project_world_to_plane(world_RA, world_DEC)
        return x, y

    def plane_to_world_orthographic(self, plane_x, plane_y):
        """Inverse Orthographic projection: tangent plane to (RA,DEC).

        Performs the inverse Orthographic projection of tangent plane
        coordinates into (RA,DEC) coordinates. The tangent plane makes
        contact at the location of the `reference_radec` variable. The
        point of perspective for the orthographic projection is at
        infinite distance. This projection is perhaps better suited to
        represent the view of an exoplanet, however it is included
        here for completeness.

        Args:
          plane_x: tangent plane x coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.
          plane_y: tangent plane y coordinate in arcseconds. The origin of the tangent plane is the contact point with the sphere, represented by `reference_radec`.

        See: https://mathworld.wolfram.com/OrthographicProjection.html

        """
        rho = (torch.sqrt(plane_x ** 2 + plane_y ** 2) + self.softening) * arcsec_to_rad
        c = torch.arcsin(rho)

        ra, dec = self._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec
