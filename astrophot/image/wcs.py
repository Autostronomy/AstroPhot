import torch
import numpy as np

from .. import AP_config
from ..utils.conversions.units import deg_to_arcsec
from ..errors import SpecificationConflict, InvalidWCS

__all__ = ("WPCS", "PPCS", "WCS")

deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi
rad_to_arcsec = rad_to_deg * 3600
arcsec_to_rad = deg_to_rad / 3600

class WPCS:
    """World to Plane Coordinate System in AstroPhot.

    AstroPhot performs its operations on a tangent plane to the
    celestial sphere, this class handles projections between the sphere and the
    tangent plane. It holds variables for the reference (RA,DEC) where
    the tangent plane contacts the sphere, and the type of projection
    being performed. Note that (RA,DEC) coordinates should always be
    in degrees while the tangent plane is in arcsecs.

    Attributes:
      reference_radec: The reference (RA,DEC) coordinates in degrees where the tangent plane contacts the sphere.
      reference_planexy: The reference tangent plane coordinates in arcsec where the tangent plane contacts the sphere.
      projection: The projection system used to convert from (RA,DEC) onto the tangent plane. Should be one of: gnomonic (default), orthographic, steriographic

    """

    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0). This is in units of arcsec.
    softening = 1e-3
    
    default_reference_radec = (0,0)
    default_reference_planexy = (0,0)
    default_projection = "gnomonic"

    def __init__(self, **kwargs):
        self.projection = kwargs.get("projection", self.default_projection)
        self.reference_radec = kwargs.get("reference_radec", self.default_reference_radec)
        self.reference_planexy = kwargs.get("reference_planexy", self.default_reference_planexy)
        
        
    def world_to_plane(self, world_RA, world_DEC=None):
        """Take a coordinate on the world coordinate system, also called the
        celesial sphere, (RA, DEC in degrees) and transform it to the
        cooresponding tangent plane coordinate
        (arcsec). Transformation is done based on the chosen
        projection (default gnomonic) and reference positions. See the
        :doc:`coordinates` documentation for more details on how the
        transformation is performed.

        """
        
        if world_DEC is None:
            return torch.stack(self.world_to_plane(*world_RA))

        world_RA = torch.as_tensor(
            world_RA, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        world_DEC = torch.as_tensor(
            world_DEC, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        
        if self.projection == "gnomonic":
            coords = self._world_to_plane_gnomonic(
                world_RA,
                world_DEC,
            )
        elif self.projection == "orthographic":
            coords = self._world_to_plane_orthographic(
                world_RA,
                world_DEC,
            )
        elif self.projection == "steriographic":
            coords = self._world_to_plane_steriographic(
                world_RA,
                world_DEC,
            )
        return coords[0] + self.reference_planexy[0], coords[1] + self.reference_planexy[1]
    

    def plane_to_world(self, plane_x, plane_y=None):
        """Take a coordinate on the tangent plane (arcsec), and transform it
        to the cooresponding world coordinate (RA, DEC in
        degrees). Transformation is done based on the chosen
        projection (default gnomonic) and reference positions. See the
        :doc:`coordinates` documentation for more details on how the
        transformation is performed.

        """
        
        if plane_y is None:
            return torch.stack(self.plane_to_world(*plane_x))
        plane_x = torch.as_tensor(
            plane_x - self.reference_planexy[0],
            dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        plane_y = torch.as_tensor(
            plane_y - self.reference_planexy[1],
            dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        if self.projection == "gnomonic":
            return self._plane_to_world_gnomonic(
                plane_x,
                plane_y,
            )
        if self.projection == "orthographic":
            return self._plane_to_world_orthographic(
                plane_x,
                plane_y,
            )
        if self.projection == "steriographic":
            return self._plane_to_world_steriographic(
                plane_x,
                plane_y,
            )

    @property
    def projection(self):
        """
        The mathematical projection formula which described how world coordinates are mapped to the tangent plane.
        """
        return self._projection

    @projection.setter
    def projection(self, proj):
        if proj not in (
                "gnomonic",
                "orthographic",
                "steriographic",
        ):
            raise InvalidWCS(f"Unrecognized projection: {proj}. Should be one of: gnomonic, orthographic, steriographic")
        self._projection = proj

    @property
    def reference_radec(self):
        """
        RA DEC (world) coordinates where the tangent plane meets the celestial sphere. These should be in degrees.
        """
        return self._reference_radec

    @reference_radec.setter
    def reference_radec(self, radec):
        self._reference_radec = torch.as_tensor(
            radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
    @property
    def reference_planexy(self):
        """
        x y tangent plane coordinates where the tangent plane meets the celestial sphere. These should be in arcsec.
        """
        return self._reference_planexy

    @reference_planexy.setter
    def reference_planexy(self, planexy):
        self._reference_planexy = torch.as_tensor(
            planexy, dtype=AP_config.ap_dtype, device=AP_config.ap_device
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
          plane_x: tangent plane x coordinate in arcseconds.
          plane_y: tangent plane y coordinate in arcseconds.
          rho: polar radius on tangent plane.
          c: coordinate term dependent on the projection.
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

    def _world_to_plane_gnomonic(self, world_RA, world_DEC):
        """Gnomonic projection: (RA,DEC) to tangent plane.

        Performs Gnomonic projection of (RA,DEC) coordinates onto a
        tangent plane. The tangent plane makes contact at the location
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

    def _plane_to_world_gnomonic(self, plane_x, plane_y):
        """Inverse Gnomonic projection: tangent plane to (RA,DEC).

        Performs the inverse Gnomonic projection of tangent plane
        coordinates into (RA,DEC) coordinates. The tangent plane makes
        contact at the location of the `reference_radec` variable. In
        a gnomonic projection, great circles are mapped to straight
        lines. The gnomonic projection represents the image formed by
        a spherical lens, and is sometimes known as the rectilinear
        projection.

        Args:
          plane_x: tangent plane x coordinate in arcseconds.
          plane_y: tangent plane y coordinate in arcseconds.

        See: https://mathworld.wolfram.com/GnomonicProjection.html

        """
        rho = (torch.sqrt(plane_x ** 2 + plane_y ** 2) + self.softening) * arcsec_to_rad
        c = torch.arctan(rho)

        ra, dec = self._project_plane_to_world(plane_x, plane_y, rho, c)
        return ra, dec

    def _world_to_plane_steriographic(self, world_RA, world_DEC):
        """Steriographic projection: (RA,DEC) to tangent plane

        Performs Steriographic projection of (RA,DEC) coordinates onto
        a tangent plane. The tangent plane makes contact at the
        location of the `reference_radec` variable. The steriographic
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

    def _plane_to_world_steriographic(self, plane_x, plane_y):
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

    def _world_to_plane_orthographic(self, world_RA, world_DEC):
        """Orthographic projection: (RA,DEC) to tangent plane

        Performs Orthographic projection of (RA,DEC) coordinates onto
        a tangent plane. The tangent plane makes contact at the
        location of the `reference_radec` variable. The point of
        perspective for the orthographic projection is at infinite
        distance. This projection is perhaps better suited to
        represent the view of an exoplanet, however it is included
        here for completeness.

        Args:
          world_RA: Right ascension in degrees
          world_DEC: Declination in degrees

        See: https://mathworld.wolfram.com/OrthographicProjection.html

        """
        x, y = self._project_world_to_plane(world_RA, world_DEC)
        return x, y

    def _plane_to_world_orthographic(self, plane_x, plane_y):
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

    def get_state(self):
        """Returns a dictionary with the information needed to recreate the
        WPCS object.

        """
        return {
            "projection": self.projection,
            "reference_radec": self.reference_radec.detach().cpu().tolist(),
            "reference_planexy": self.reference_planexy.detach().cpu().tolist(),
        }
    def set_state(self, state):
        """Takes a state dictionary and re-creates the state of the WPCS
        object.

        """
        self.projection = state.get("projection", self.default_projection)
        self.reference_radec = state.get("reference_radec", self.default_reference_radec)
        self.reference_planexy = state.get("reference_planexy", self.default_reference_planexy)

    def get_fits_state(self):
        """
        Similar to get_state, except specifically tailored to be stored in a FITS format.
        """
        return {
            "PROJ": self.projection,
            "REFRADEC": str(self.reference_radec.detach().cpu().tolist()),
            "REFPLNXY": str(self.reference_planexy.detach().cpu().tolist()),
        }
    
    def set_fits_state(self, state):
        """
        Reads and applies the state from the get_fits_state function.
        """
        self.projection = state["PROJ"]
        self.reference_radec = eval(state["REFRADEC"])
        self.reference_planexy = eval(state["REFPLNXY"])
        
    def copy(self, **kwargs):
        """Create a copy of the WPCS object with the same projection
        paramaters.

        """
        copy_kwargs = {
            "projection": self.projection,
            "reference_radec": self.reference_radec,
            "reference_planexy": self.reference_planexy,
        }
        copy_kwargs.update(kwargs)
        return self.__class__(
            **copy_kwargs,
        )
    
    def to(self, dtype=None, device=None):
        """
        Convert all stored tensors to a new device and data type
        """
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self._reference_radec = self._reference_radec.to(dtype=dtype, device=device)
        self._reference_planexy = self._reference_planexy.to(dtype=dtype, device=device)

    def __str__(self):
        return f"WPCS reference_radec: {self.reference_radec.detach().cpu().tolist()}, reference_planexy: {self.reference_planexy.detach().cpu().tolist()}"
    def __repr__(self):
        return f"WPCS reference_radec: {self.reference_radec.detach().cpu().tolist()}, reference_planexy: {self.reference_planexy.detach().cpu().tolist()}, projection: {self.projection}"

class PPCS:
    """
    plane to pixel coordinate system


    Args:
      pixelscale : float or None, optional
          The physical scale of the pixels in the image, this is
          represented as a matrix which projects pixel units into sky
          units: :math:`pixelscale @ pixel_vec = sky_vec`. The pixel
          scale matrix can be thought of in four components:
          :math:`\vec{s} @ F @ R @ S` where :math:`\vec{s}` is the side
          length of the pixels, :math:`F` is a diagonal matrix of {1,-1}
          which flips the axes orientation, :math:`R` is a rotation
          matrix, and :math:`S` is a shear matrix which turns
          rectangular pixels into parallelograms. Default is None.      
      reference_imageij : Sequence or None, optional
          The pixel coordinate at which the image is fixed to the
          tangent plane. By default this is (-0.5, -0.5) or the bottom
          corner of the [0,0] indexed pixel.
      reference_imagexy : Sequence or None, optional
          The tangent plane coordinate at which the image is fixed,
          corresponding to the reference_imageij coordinate. These two
          reference points ar pinned together, any rotations would occur
          about this point. By default this is (0., 0.).
    
    """
    
    default_reference_imageij = (-0.5,-0.5)
    default_reference_imagexy = (0,0)
    default_pixelscale = 1

    def __init__(self, *, wcs=None, pixelscale=None, **kwargs):

        self.reference_imageij = kwargs.get("reference_imageij", self.default_reference_imageij)        
        self.reference_imagexy = kwargs.get("reference_imagexy", self.default_reference_imagexy)

        # Collect the pixelscale of the pixel grid
        if wcs is not None and pixelscale is None:
            self.pixelscale = deg_to_arcsec * wcs.pixel_scale_matrix
        elif pixelscale is not None:
            if wcs is not None and isinstance(pixelscale, float):
                AP_config.ap_logger.warning(
                    "Overriding WCS pixelscale with manual input! To remove this message, either let WCS define pixelscale, or input full pixelscale matrix"
                )
            self.pixelscale = pixelscale
        else:
            AP_config.ap_logger.warning(
                "Assuming pixelscale of 1! To remove this message please provide the pixelscale explicitly"
            )
            self.pixelscale = self.default_pixelscale
        
    @property
    def pixelscale(self):
        """Matrix defining the shape of pixels in the tangent plane, these
        can be any parallelogram defined by the matrix.

        """
        return self._pixelscale

    @pixelscale.setter
    def pixelscale(self, pix):
        if pix is None:
            self._pixelscale = None
            return

        self._pixelscale = (
            torch.as_tensor(pix, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            .clone()
            .detach()
        )
        if self._pixelscale.numel() == 1:
            self._pixelscale = torch.tensor(
                [[self._pixelscale.item(), 0.0], [0.0, self._pixelscale.item()]],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        self._pixel_area = torch.linalg.det(self.pixelscale).abs()
        self._pixel_length = self._pixel_area.sqrt()
        self._pixelscale_inv = torch.linalg.inv(self.pixelscale)

    @property
    def pixel_area(self):
        """The area inside a pixel in arcsec^2

        """
        return self._pixel_area

    @property
    def pixel_length(self):
        """The approximate length of a pixel, which is just
        sqrt(pixel_area). For square pixels this is the actual pixel
        length, for rectangular pixels it is a kind of average.

        The pixel_length is typically not used for exact calculations
        and instead sets a size scale within an image.

        """
        return self._pixel_length

    @property
    def reference_imageij(self):
        """pixel coordinates where the pixel grid is fixed to the tangent
        plane. These should be in pixel units where (0,0) is the
        center of the [0,0] indexed pixel. However, it is still in xy
        format, meaning that the first index gives translations in the
        x-axis (horizontal-axis) of the image.

        """
        return self._reference_imageij

    @reference_imageij.setter
    def reference_imageij(self, imageij):
        self._reference_imageij = torch.as_tensor(
            imageij, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
    @property
    def reference_imagexy(self):
        """plane coordinates where the image grid is fixed to the tangent
        plane. These should be in arcsec.

        """
        return self._reference_imagexy

    @reference_imagexy.setter
    def reference_imagexy(self, imagexy):
        self._reference_imagexy = torch.as_tensor(
            imagexy, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def pixel_to_plane(self, pixel_i, pixel_j=None):
        """Take in a coordinate on the regular pixel grid, where 0,0 is the
        center of the [0,0] indexed pixel. This coordinate is
        transformed into the tangent plane coordinate system (arcsec)
        based on the pixel scale and reference positions. If the pixel
        scale matrix is :math:`P`, the reference pixel is
        :math:`\vec{r}_{pix}`, the reference tangent plane point is
        :math:`\vec{r}_{tan}`, and the coordinate to transform is
        :math:`\vec{c}_{pix}` then the coordinate in the tangent plane
        is:

        .. math::
            \vec{c}_{tan} = [P(\vec{c}_{pix} - \vec{r}_{pix})] + \vec{r}_{tan}

        """
        if pixel_j is None:
            return torch.stack(self.pixel_to_plane(*pixel_i))
        coords = torch.mm(self.pixelscale, torch.stack((pixel_i.reshape(-1), pixel_j.reshape(-1))) - self.reference_imageij.view(2,1)) + self.reference_imagexy.view(2,1)
        return coords[0].reshape(pixel_i.shape), coords[1].reshape(pixel_j.shape)

    def plane_to_pixel(self, plane_x, plane_y=None):
        """Take a coordinate on the tangent plane (arcsec) and transform it to
        the cooresponding pixel grid coordinate (pixel units where
        (0,0) is the [0,0] indexed pixel). Transformation is done
        based on the pixel scale and reference positions. If the pixel
        scale matrix is :math:`P`, the reference pixel is
        :math:`\vec{r}_{pix}`, the reference tangent plane point is
        :math:`\vec{r}_{tan}`, and the coordinate to transform is
        :math:`\vec{c}_{tan}` then the coordinate in the pixel grid
        is:

        .. math::
            \vec{c}_{pix} = [P^{-1}(\vec{c}_{tan} - \vec{r}_{tan})] + \vec{r}_{pix}

        """
        if plane_y is None:
            return torch.stack(self.plane_to_pixel(*plane_x))
        coords = torch.mm(self._pixelscale_inv, torch.stack((plane_x.reshape(-1), plane_y.reshape(-1))) - self.reference_imagexy.view(2,1)) + self.reference_imageij.view(2,1)
        return coords[0].reshape(plane_x.shape), coords[1].reshape(plane_y.shape)

    def pixel_to_plane_delta(self, pixel_delta_i, pixel_delta_j=None):
        """Take a translation in pixel space and determine the cooresponding
        translation in the tangent plane (arcsec). Essentially this performs
        the pixel scale matrix multiplication without any reference
        coordinates applied.

        """
        if pixel_delta_j is None:
            return torch.stack(self.pixel_to_plane_delta(*pixel_delta_i))
        coords = torch.mm(self.pixelscale, torch.stack((pixel_delta_i.reshape(-1), pixel_delta_j.reshape(-1))))
        return coords[0].reshape(pixel_delta_i.shape), coords[1].reshape(pixel_delta_j.shape)

    def plane_to_pixel_delta(self, plane_delta_x, plane_delta_y=None):
        """Take a translation in tangent plane space (arcsec) and determine
        the cooresponding translation in pixel space. Essentially this
        performs the pixel scale matrix multiplication without any
        reference coordinates applied.

        """
        if plane_delta_y is None:
            return torch.stack(self.plane_to_pixel_delta(*plane_delta_x))
        coords = torch.mm(self._pixelscale_inv, torch.stack((plane_delta_x.reshape(-1), plane_delta_y.reshape(-1))))
        return coords[0].reshape(plane_delta_x.shape), coords[1].reshape(plane_delta_y.shape)
        
    def copy(self, **kwargs):
        """Create a copy of the PPCS object with the same projection
        paramaters.

        """
        copy_kwargs = {
            "pixelscale": self.pixelscale,
            "reference_imageij": self.reference_imageij,
            "reference_imagexy": self.reference_imagexy,
        }
        copy_kwargs.update(kwargs)
        return self.__class__(
            **copy_kwargs,
        )
    
    def get_state(self):
        return {
            "pixelscale": self.pixelscale.detach().cpu().tolist(),
            "reference_imageij": self.reference_imageij.detach().cpu().tolist(),
            "reference_imagexy": self.reference_imagexy.detach().cpu().tolist(),
        }
    def set_state(self, state):
        self.pixelscale = state.get("pixelscale", self.default_pixelscale)
        self.reference_imageij = state.get("reference_imageij", self.default_reference_imageij)
        self.reference_imagexy = state.get("reference_imagexy", self.default_reference_imagexy)
    
    def get_fits_state(self):
        """
        Similar to get_state, except specifically tailored to be stored in a FITS format.
        """
        return {
            "PXLSCALE": str(self.pixelscale.detach().cpu().tolist()),
            "REFIMGIJ": str(self.reference_imageij.detach().cpu().tolist()),
            "REFIMGXY": str(self.reference_imagexy.detach().cpu().tolist()),
        }
    
    def set_fits_state(self, state):
        """
        Reads and applies the state from the get_fits_state function.
        """
        self.pixelscale = eval(state["PXLSCALE"])
        self.reference_imageij = eval(state["REFIMGIJ"])
        self.reference_imagexy = eval(state["REFIMGXY"])
        
    def to(self, dtype=None, device=None):
        """
        Convert all stored tensors to a new device and data type
        """
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self._pixelscale = self._pixelscale.to(dtype=dtype, device=device)
        self._reference_imageij = self._reference_imageij.to(dtype=dtype, device=device)
        self._reference_imagexy = self._reference_imagexy.to(dtype=dtype, device=device)

    def __str__(self):
        return f"PPCS reference_imageij: {self.reference_imageij.detach().cpu().tolist()}, reference_imagexy: {self.reference_imagexy.detach().cpu().tolist()}"
    def __repr__(self):
        return f"PPCS reference_imageij: {self.reference_imageij.detach().cpu().tolist()}, reference_imagexy: {self.reference_imagexy.detach().cpu().tolist()}, pixelscale: {self.pixelscale.detach().cpu().tolist()}"

        
class WCS(WPCS, PPCS):
    """
    Full world coordinate system defines mappings from world to tangent plane to pixel grid and all other variations.
    """

    def __init__(self, *args, wcs=None, **kwargs):
        if kwargs.get("state", None) is not None:
            self.set_state(kwargs["state"])
            return
        
        if wcs is not None:
            if wcs.wcs.ctype[0] != "RA---TAN":
                AP_config.ap_logger.warning("Astropy WCS not tangent plane coordinate system! May not be compatible with AstroPhot.")
            if wcs.wcs.ctype[1] != "DEC--TAN":
                AP_config.ap_logger.warning("Astropy WCS not tangent plane coordinate system! May not be compatible with AstroPhot.")
                
        if wcs is not None:
            kwargs["reference_radec"] = kwargs.get("reference_radec", wcs.wcs.crval)
            kwargs["reference_imageij"] = wcs.wcs.crpix
            WPCS.__init__(self, *args, wcs=wcs, **kwargs)
            sky_coord = wcs.pixel_to_world(*wcs.wcs.crpix)
            kwargs["reference_imagexy"] = self.world_to_plane(torch.tensor((sky_coord.ra.deg, sky_coord.dec.deg), dtype=AP_config.ap_dtype, device=AP_config.ap_device))
        else:
            WPCS.__init__(self, *args, **kwargs)
            
        PPCS.__init__(self, *args, wcs=wcs, **kwargs)

    def world_to_pixel(self, world_RA, world_DEC=None):
        """A wrapper which applies :meth:`world_to_plane` then
        :meth:`plane_to_pixel`, see those methods for further
        information.

        """
        if world_DEC is None:
            return torch.stack(self.world_to_pixel(*world_RA))
        return self.plane_to_pixel(*self.world_to_plane(world_RA, world_DEC))
    
    def pixel_to_world(self, pixel_i, pixel_j=None):
        """A wrapper which applies :meth:`pixel_to_plane` then
        :meth:`plane_to_world`, see those methods for further
        information.

        """
        if pixel_j is None:
            return torch.stack(self.pixel_to_world(*pixel_i))
        return self.plane_to_world(*self.pixel_to_plane(pixel_i, pixel_j))

    def copy(self, **kwargs):
        copy_kwargs = {
            "pixelscale": self.pixelscale,
            "reference_imageij": self.reference_imageij,
            "reference_imagexy": self.reference_imagexy,
            "projection": self.projection,
            "reference_radec": self.reference_radec,
            "reference_planexy": self.reference_planexy,
        }
        copy_kwargs.update(kwargs)
        return self.__class__(
            **copy_kwargs,     
        )

    def to(self, dtype=None, device=None):
        WPCS.to(self, dtype, device)
        PPCS.to(self, dtype, device)

    def get_state(self):
        state = WPCS.get_state(self)
        state.update(PPCS.get_state(self))
        return state

    def set_state(self, state):
        WPCS.set_state(self, state)
        PPCS.set_state(self, state)
        
    def get_fits_state(self):
        """
        Similar to get_state, except specifically tailored to be stored in a FITS format.
        """
        state = WPCS.get_fits_state(self)
        state.update(PPCS.get_fits_state(self))
        return state
    
    def set_fits_state(self, state):
        """
        Reads and applies the state from the get_fits_state function.
        """
        WPCS.set_fits_state(self, state)
        PPCS.set_fits_state(self, state)
        
    def __str__(self):
        return f"WCS:\n{WPCS.__str__(self)}\n{PPCS.__str__(self)}"
    def __repr__(self):
        return f"WCS:\n{WPCS.__repr__(self)}\n{PPCS.__repr__(self)}"
