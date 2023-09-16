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

    _reference_radec = None
    _projection = None

    def __init__(self, *args, **kwargs):

        if self.projection is None:
            self.projection = "gnomonic"
        if self.reference_radec is None:
            self.reference_radec = kwargs.get("reference_radec", (0, 0))

    @classmethod
    def world_to_plane(cls, world_RA, world_DEC):
        if cls._projection == "gnomonic":
            return cls.world_to_plane_gnomonic(
                torch.as_tensor(
                    world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if cls._projection == "orthographic":
            return cls.world_to_plane_orthographic(
                torch.as_tensor(
                    world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if cls._projection == "steriographic":
            return cls.world_to_plane_steriographic(
                torch.as_tensor(
                    world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        raise ValueError(
            f"Unrecognized projection: {cls._projection}. Should be one of: gnomonic, orthographic, steriographic"
        )

    @classmethod
    def plane_to_world(cls, plane_x, plane_y):
        if cls._projection == "gnomonic":
            return cls.plane_to_world_gnomonic(
                torch.as_tensor(
                    plane_x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    plane_y, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if cls._projection == "orthographic":
            return cls.plane_to_world_orthographic(
                torch.as_tensor(
                    plane_x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    plane_y, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        if cls._projection == "steriographic":
            return cls.plane_to_world_steriographic(
                torch.as_tensor(
                    plane_x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                torch.as_tensor(
                    plane_y, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )
        raise ValueError(
            f"Unrecognized projection: {cls._projection}. Should be one of: gnomonic, orthographic, steriographic"
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
        WCS._projection = proj

    @property
    def reference_radec(self):
        return self._reference_radec

    @reference_radec.setter
    def reference_radec(self, radec):
        WCS._reference_radec = torch.tensor(
            radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    @classmethod
    def _project_world_to_plane(cls, world_RA, world_DEC):
        """
        Recurring core calculation in all the projections from world to plane.

        Args:
          world_RA: Right ascension in degrees
          world_DEC: Declination in degrees
        """
        return (
            torch.cos(world_DEC * deg_to_rad)
            * torch.sin((world_RA - cls._reference_radec[0]) * deg_to_rad)
            * rad_to_arcsec,
            (
                torch.cos(cls._reference_radec[1] * deg_to_rad)
                * torch.sin(world_DEC * deg_to_rad)
                - torch.sin(cls._reference_radec[1] * deg_to_rad)
                * torch.cos(world_DEC * deg_to_rad)
                * torch.cos((world_RA - cls._reference_radec[0]) * deg_to_rad)
            )
            * rad_to_arcsec,
        )

    @classmethod
    def _project_plane_to_world(cls, plane_x, plane_y, rho, c):
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
                cls._reference_radec[0] * deg_to_rad
                + torch.arctan2(
                    plane_x * arcsec_to_rad * torch.sin(c),
                    rho * torch.cos(cls._reference_radec[1] * deg_to_rad) * torch.cos(c)
                    - plane_y
                    * arcsec_to_rad
                    * torch.sin(cls._reference_radec[1] * deg_to_rad)
                    * torch.sin(c),
                )
            )
            * rad_to_deg,
            torch.arcsin(
                torch.cos(c)
                * torch.sin(
                    cls._reference_radec[1] * deg_to_rad
                    + plane_y
                    * arcsec_to_rad
                    * torch.sin(c)
                    * torch.cos(cls._reference_radec[1] * deg_to_rad)
                    / rho
                )
            )
            * rad_to_deg,
        )

    @classmethod
    def world_to_plane_gnomonic(cls, world_RA, world_DEC):
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
        C = torch.sin(cls._reference_radec[1] * deg_to_rad) * torch.sin(
            world_DEC * deg_to_rad
        ) + torch.cos(cls._reference_radec[1] * deg_to_rad) * torch.cos(
            world_DEC * deg_to_rad
        ) * torch.cos(
            (world_RA - cls._reference_radec[0]) * deg_to_rad
        )
        x, y = cls._project_world_to_plane(world_RA, world_DEC)
        return x / C, y / C

    @classmethod
    def plane_to_world_gnomonic(cls, plane_x, plane_y):
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
        rho = torch.sqrt(plane_x ** 2 + plane_y ** 2) * arcsec_to_rad
        c = torch.arctan(rho)

        ra, dec = cls._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec

    @classmethod
    def world_to_plane_steriographic(cls, world_RA, world_DEC):
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
            * torch.sin(cls._reference_radec[1] * deg_to_rad)
            + torch.cos(world_DEC * deg_to_rad)
            * torch.cos(cls._reference_radec[1] * deg_to_rad)
            * torch.cos((world_RA - cls._reference_radec[0]) * deg_to_rad)
        ) / 2
        x, y = cls._project_world_to_plane(world_RA, world_DEC)
        return x / C, y / C

    @classmethod
    def plane_to_world_steriographic(cls, plane_x, plane_y):
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

        rho = torch.sqrt(plane_x ** 2 + plane_y ** 2) * arcsec_to_rad
        c = 2 * torch.arctan(rho / 2)
        ra, dec = cls._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec

    @classmethod
    def world_to_plane_orthographic(cls, world_RA, world_DEC):
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
        x, y = cls._project_world_to_plane(world_RA, world_DEC)
        return x, y

    @classmethod
    def plane_to_world_orthographic(cls, plane_x, plane_y):
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
        rho = torch.sqrt(plane_x ** 2 + plane_y ** 2) * arcsec_to_rad
        c = torch.arcsin(rho)

        ra, dec = cls._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    WCS.reference_radec = torch.tensor((90.0, 0.0))

    fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
    for proj, ax in zip(["gnomonic", "orthographic", "steriographic"], axarr):
        WCS.projection = proj
        wcs = WCS()
        long_RA, long_DEC = torch.meshgrid(
            torch.linspace(90 - 1 / 3600, 90 + 1 / 3600, 24, dtype=torch.float64),
            torch.linspace(-1 / 3600, 1 / 3600, 1000, dtype=torch.float64),
            indexing="xy",
        )

        long_x, long_y = wcs.world_to_plane(long_RA, long_DEC)

        ax.plot(long_x.numpy(), long_y.numpy())
        ax.set_title(proj)
    plt.tight_layout()
    plt.savefig(f"small_field_longitude.jpg")
    plt.close()

    fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
    for proj, ax in zip(["gnomonic", "orthographic", "steriographic"], axarr):
        WCS.projection = proj
        wcs = WCS()
        long_RA, long_DEC = torch.meshgrid(
            torch.linspace(10, 350, 24), torch.linspace(-80, 80, 1000), indexing="xy"
        )

        long_x, long_y = wcs.world_to_plane(long_RA, long_DEC)

        ax.plot(long_x.numpy(), long_y.numpy())
        ax.set_title(proj)
    plt.tight_layout()
    plt.savefig(f"large_field_longitude.jpg")
    plt.close()
