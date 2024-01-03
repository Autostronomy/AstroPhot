import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS

from .. import AP_config
from ..utils.conversions.coordinates import Rotate_Cartesian
from .wcs import WCS
from ..errors import ConflicingWCS, SpecificationConflict

__all__ = ["Window", "Window_List"]


class Window(WCS):
    """class to define a window on the sky in coordinate space. These
    windows can undergo arithmetic and preserve logical behavior. Image
    objects can also be indexed using windows and will return an
    appropriate subsection of their data.

    There are several ways to tell a Window object where to
    place itself. The simplest method is to pass an
    Astropy WCS object such as::

      H = ap.image.Window(wcs = wcs)

    this will automatically place your image at the correct RA, DEC
    and assign the correct pixel scale, etc. WARNING, it will default to
    setting the reference RA DEC at the reference RA DEC of the wcs
    object; if you have multiple images you should force them all to
    have the same reference world coordinate by passing
    ``reference_radec = (ra, dec)``. See the :doc:`coordinates`
    documentation for more details. There are several other ways to
    initialize a window. If you provide ``origin_radec`` then
    it will place the image origin at the requested RA DEC
    coordinates. If you provide ``center_radec`` then it will place
    the image center at the requested RA DEC coordiantes. Note that in
    these cases the fixed point between the pixel grid and image plane
    is different (pixel origin and center respectively); so if you
    have rotated pixels in your pixel scale matrix then everything
    will be rotated about different points (pixel origin and center
    respectively). If you provide ``origin`` or ``center`` then those
    are coordiantes in the tangent plane (arcsec) and they will
    correspondingly become fixed points. For arbitrary control over
    the pixel positioning, use ``reference_imageij`` and
    ``reference_imagexy`` to fix the pixel and tangent plane
    coordinates respectively to each other, any rotation or shear will
    happen about that fixed point.

    Args:
      origin : Sequence or None, optional
          The origin of the image in the tangent plane coordinate system
          (arcsec), as a 1D array of length 2. Default is None.
      origin_radec : Sequence or None, optional
          The origin of the image in the world coordinate system (RA,
          DEC in degrees), as a 1D array of length 2. Default is None.
      center : Sequence or None, optional
          The center of the image in the tangent plane coordinate system
          (arcsec), as a 1D array of length 2. Default is None.
      center_radec : Sequence or None, optional
          The center of the image in the world coordinate system (RA,
          DEC in degrees), as a 1D array of length 2. Default is None.
      wcs: An astropy.wcs.WCS object which gives information about the
          origin and orientation of the window.
      reference_radec: world coordinates on the celestial sphere (RA,
          DEC in degrees) where the tangent plane makes contact. This should
          be the same for every image in multi-image analysis.
      reference_planexy: tangent plane coordinates (arcsec) where it
          makes contact with the celesial sphere. This should typically be
          (0,0) though that is not stricktly enforced (it is assumed if not
          given). This reference coordinate should be the same for all
          images in multi-image analysis.
      reference_imageij: pixel coordinates about which the image is
          defined. For example in an Astropy WCS object the wcs.wcs.crpix
          array gives the pixel coordinate reference point for which the
          world coordinate mapping (wcs.wcs.crval) is defined. One may think
          of the referenced pixel location as being "pinned" to the tangent
          plane. This may be different for each image in multi-image
          analysis..
      reference_imagexy: tangent plane coordinates (arcsec) about
          which the image is defined. This is the pivot point about which the
          pixelscale matrix operates, therefore if the pixelscale matrix
          defines a rotation then this is the coordinate about which the
          rotation will be performed. This may be different for each image in
          multi-image analysis.

    """

    def __init__(
        self,
        *,
        pixel_shape=None,
        origin=None,
        origin_radec=None,
        center=None,
        center_radec=None,
        state=None,
        fits_state=None,
        wcs=None,
        **kwargs,
    ):
        # If loading from a previous state, simply update values and end init
        if state is not None:
            self.set_state(state)
            return
        if fits_state is not None:
            self.set_fits_state(fits_state)
            return

        # Collect the shape of the window
        if pixel_shape is not None:
            self.pixel_shape = pixel_shape
        else:
            self.pixel_shape = wcs.pixel_shape

        # Determine relative positioning of tangent plane and pixel grid. Also world coordinates and tangent plane
        if not sum(C is not None for C in [wcs, origin_radec, center_radec, origin, center]) <= 1:
            raise SpecificationConflict(
                "Please provide only one reference position for the window, otherwise the placement is ambiguous"
            )

        # Image coordinates provided by WCS
        if wcs is not None:
            super().__init__(wcs=wcs, **kwargs)
        # Image reference position from RA and DEC of image origin
        elif origin_radec is not None:
            # Origin given, it is reference point
            origin_radec = torch.as_tensor(
                origin_radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            kwargs["reference_radec"] = kwargs.get("reference_radec", origin_radec)
            super().__init__(**kwargs)
            self.reference_imageij = (-0.5, -0.5)
            self.reference_imagexy = self.world_to_plane(origin_radec)
        # Image reference position from RA and DEC of image center
        elif center_radec is not None:
            pix_center = self.pixel_shape.to(dtype=AP_config.ap_dtype) / 2 - 0.5
            center_radec = torch.as_tensor(
                center_radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            kwargs["reference_radec"] = kwargs.get("reference_radec", center_radec)
            super().__init__(**kwargs)
            center = self.world_to_plane(center_radec)
            self.reference_imageij = pix_center
            self.reference_imagexy = center
        # Image reference position from tangent plane position of image origin
        elif origin is not None:
            kwargs.update(
                {
                    "reference_imageij": (-0.5, -0.5),
                    "reference_imagexy": origin,
                }
            )
            super().__init__(**kwargs)
        # Image reference position from tangent plane position of image center
        elif center is not None:
            pix_center = self.pixel_shape.to(dtype=AP_config.ap_dtype) / 2 - 0.5
            kwargs.update(
                {
                    "reference_imageij": pix_center,
                    "reference_imagexy": center,
                }
            )
            super().__init__(**kwargs)
        # Image origin assumed to be at tangent plane origin
        else:
            super().__init__(**kwargs)

    @property
    def shape(self):
        dtype, device = self.pixelscale.dtype, self.pixelscale.device
        S1 = self.pixel_shape.to(dtype=dtype, device=device)
        S1[1] = 0.0
        S2 = self.pixel_shape.to(dtype=dtype, device=device)
        S2[0] = 0.0
        return torch.stack(
            (
                torch.linalg.norm(self.pixelscale @ S1),
                torch.linalg.norm(self.pixelscale @ S2),
            )
        )

    @shape.setter
    def shape(self, shape):
        if shape is None:
            self._pixel_shape = None
            return
        shape = torch.as_tensor(shape, dtype=self.pixelscale.dtype, device=self.pixelscale.device)
        self.pixel_shape = shape / torch.sqrt(torch.sum(self.pixelscale**2, dim=0))

    @property
    def pixel_shape(self):
        return self._pixel_shape

    @pixel_shape.setter
    def pixel_shape(self, shape):
        if shape is None:
            self._pixel_shape = None
            return
        self._pixel_shape = torch.as_tensor(shape, device=AP_config.ap_device)
        self._pixel_shape = torch.round(self.pixel_shape).to(
            dtype=torch.int32, device=AP_config.ap_device
        )

    @property
    def size(self):
        """The number of pixels in the window"""
        return torch.prod(self.pixel_shape)

    @property
    def end(self):
        return self.pixel_to_plane_delta(
            self.pixel_shape.to(dtype=self.pixelscale.dtype, device=self.pixelscale.device)
        )

    @property
    def origin(self):
        return self.pixel_to_plane(-0.5 * torch.ones_like(self.reference_imageij))

    @property
    def center(self):
        return self.origin + self.end / 2

    def copy(self, **kwargs):
        copy_kwargs = {"pixel_shape": torch.clone(self.pixel_shape)}
        copy_kwargs.update(kwargs)
        return super().copy(**copy_kwargs)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)
        self.pixel_shape = self.pixel_shape.to(dtype=dtype, device=device)

    def rescale_pixel(self, scale, **kwargs):
        return self.copy(
            pixelscale=self.pixelscale * scale,
            pixel_shape=self.pixel_shape // scale,
            reference_imageij=(self.reference_imageij + 0.5) / scale - 0.5,
            **kwargs,
        )

    @staticmethod
    @torch.no_grad()
    def _get_indices(ref_window, obj_window):
        other_origin_pix = torch.round(ref_window.plane_to_pixel(obj_window.origin) + 0.5).int()
        new_origin_pix = torch.maximum(torch.zeros_like(other_origin_pix), other_origin_pix)

        other_pixel_end = torch.round(
            ref_window.plane_to_pixel(obj_window.origin + obj_window.end) + 0.5
        ).int()
        new_pixel_end = torch.minimum(ref_window.pixel_shape, other_pixel_end)
        return slice(new_origin_pix[1], new_pixel_end[1]), slice(
            new_origin_pix[0], new_pixel_end[0]
        )

    def get_self_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        if isinstance(obj, Window):
            return self._get_indices(self, obj)
        return self._get_indices(self, obj.window)

    def get_other_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        if isinstance(obj, Window):
            return self._get_indices(obj, self)
        return self._get_indices(obj.window, self)

    def overlap_frac(self, other):
        overlap = self & other
        overlap_area = torch.prod(overlap.shape)
        full_area = torch.prod(self.shape) + torch.prod(other.shape) - overlap_area
        return overlap_area / full_area

    def shift(self, shift):
        """
        Shift the location of the window by a specified amount in tangent plane coordinates
        """
        self.reference_imagexy = self.reference_imagexy + shift
        return self

    def pixel_shift(self, shift):
        """
        Shift the location of the window by a specified amount in pixel grid coordinates
        """

        self.reference_imageij = self.reference_imageij - shift
        return self

    def get_astropywcs(self, **kwargs):
        wargs = {
            "NAXIS": 2,
            "NAXIS1": self.pixel_shape[0].item(),
            "NAXIS2": self.pixel_shape[1].item(),
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": self.pixel_to_world(self.reference_imageij)[0].item(),
            "CRVAL2": self.pixel_to_world(self.reference_imageij)[1].item(),
            "CRPIX1": self.reference_imageij[0].item(),
            "CRPIX2": self.reference_imageij[1].item(),
            "CD1_1": self.pixelscale[0][0].item(),
            "CD1_2": self.pixelscale[0][1].item(),
            "CD2_1": self.pixelscale[1][0].item(),
            "CD2_2": self.pixelscale[1][1].item(),
        }
        wargs.update(kwargs)
        return AstropyWCS(wargs)

    def get_state(self):
        state = super().get_state()
        state["pixel_shape"] = self.pixel_shape.detach().cpu().tolist()
        return state

    def set_state(self, state):
        super().set_state(state)
        self.pixel_shape = torch.tensor(
            state["pixel_shape"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def get_fits_state(self):
        state = super().get_fits_state()
        state["PXL_SHPE"] = str(self.pixel_shape.detach().cpu().tolist())
        return state

    def set_fits_state(self, state):
        super().set_fits_state(state)
        self.pixel_shape = torch.tensor(
            eval(state["PXL_SHPE"]), dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def crop_pixel(self, pixels):
        """
        [crop all sides] or
        [crop x, crop y] or
        [crop x low, crop y low, crop x high, crop y high]
        """
        if len(pixels) == 1:
            self.pixel_shape = self.pixel_shape - 2 * pixels[0]
            self.reference_imageij = self.reference_imageij - pixels[0]
        elif len(pixels) == 2:
            pix_shift = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.pixel_shape = self.pixel_shape - 2 * pix_shift
            self.reference_imageij = self.reference_imageij - pix_shift
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            self.pixel_shape = self.pixel_shape - pixels[:2] - pixels[2:]
            self.reference_imageij = self.reference_imageij - pixels[:2]
        else:
            raise ValueError(f"Unrecognized pixel crop format: {pixels}")
        return self

    def crop_to_pixel(self, pixels):
        """
        format: [[xmin, xmax],[ymin,ymax]]
        """
        pixels = torch.tensor(pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        self.reference_imageij = self.reference_imageij - pixels[:, 0]
        self.pixel_shape = pixels[:, 1] - pixels[:, 0]
        return self

    def pad_pixel(self, pixels):
        """
        [pad all sides] or
        [pad x, pad y] or
        [pad x low, pad y low, pad x high, pad y high]
        """
        if len(pixels) == 1:
            self.pixel_shape = self.pixel_shape + 2 * pixels[0]
            self.reference_imageij = self.reference_imageij + pixels[0]
        elif len(pixels) == 2:
            pix_shift = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.pixel_shape = self.pixel_shape + 2 * pix_shift
            self.reference_imageij = self.reference_imageij + pix_shift
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            self.pixel_shape = self.pixel_shape + pixels[:2] + pixels[2:]
            self.reference_imageij = self.reference_imageij + pixels[:2]
        else:
            raise ValueError(f"Unrecognized pixel crop format: {pixels}")
        return self

    @torch.no_grad()
    def get_coordinate_meshgrid(self):
        """Returns a meshgrid with tangent plane coordinates for the center
        of every pixel.

        """
        pix = self.pixel_shape.to(dtype=AP_config.ap_dtype)
        xsteps = torch.arange(pix[0], dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        ysteps = torch.arange(pix[1], dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        meshx, meshy = torch.meshgrid(
            xsteps,
            ysteps,
            indexing="xy",
        )
        Coords = self.pixel_to_plane(meshx, meshy)
        return torch.stack(Coords)

    @torch.no_grad()
    def get_coordinate_corner_meshgrid(self):
        """Returns a meshgrid with tangent plane coordinates for the corners
        of every pixel.

        """
        pix = self.pixel_shape.to(dtype=AP_config.ap_dtype)
        xsteps = (
            torch.arange(pix[0] + 1, dtype=AP_config.ap_dtype, device=AP_config.ap_device) - 0.5
        )
        ysteps = (
            torch.arange(pix[1] + 1, dtype=AP_config.ap_dtype, device=AP_config.ap_device) - 0.5
        )
        meshx, meshy = torch.meshgrid(
            xsteps,
            ysteps,
            indexing="xy",
        )
        Coords = self.pixel_to_plane(meshx, meshy)
        return torch.stack(Coords)

    @torch.no_grad()
    def get_coordinate_simps_meshgrid(self):
        """Returns a meshgrid with tangent plane coordinates for performing
        simpsons method pixel integration (all corners, centers, and
        middle of each edge). This is approximately 4 times more
        points than the standard :meth:`get_coordinate_meshgrid`.

        """
        pix = self.pixel_shape.to(dtype=AP_config.ap_dtype)
        xsteps = (
            0.5
            * torch.arange(
                2 * (pix[0]) + 1,
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            - 0.5
        )
        ysteps = (
            0.5
            * torch.arange(
                2 * (pix[1]) + 1,
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            - 0.5
        )
        meshx, meshy = torch.meshgrid(
            xsteps,
            ysteps,
            indexing="xy",
        )
        Coords = self.pixel_to_plane(meshx, meshy)
        return torch.stack(Coords)

    # Window Comparison operators
    @torch.no_grad()
    def __eq__(self, other):
        return (
            torch.all(self.pixel_shape == other.pixel_shape)
            and torch.all(self.pixelscale == other.pixelscale)
            and (self.projection == other.projection)
            and (
                torch.all(
                    self.pixel_to_plane(torch.zeros_like(self.reference_imageij))
                    == other.pixel_to_plane(torch.zeros_like(other.reference_imageij))
                )
            )
        )  # fixme more checks?

    @torch.no_grad()
    def __ne__(self, other):
        return not self == other

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.minimum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.maximum(
            self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end
        )
        return self.copy(
            origin=self.pixel_to_plane(new_origin_pix),
            pixel_shape=new_pixel_end - new_origin_pix,
        )

    @torch.no_grad()
    def __ior__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.minimum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.maximum(
            self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end
        )

        self.reference_imageij = self.reference_imageij - (new_origin_pix + 0.5)
        self.pixel_shape = new_pixel_end - new_origin_pix
        return self

    @torch.no_grad()
    def __and__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.maximum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.minimum(
            self.pixel_shape.to(dtype=AP_config.ap_dtype) - 0.5, other_pixel_end
        )
        return self.copy(
            origin=self.pixel_to_plane(new_origin_pix),
            pixel_shape=new_pixel_end - new_origin_pix,
        )

    @torch.no_grad()
    def __iand__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.maximum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.minimum(
            self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end
        )

        self.reference_imageij = self.reference_imageij - (new_origin_pix + 0.5)
        self.pixel_shape = new_pixel_end - new_origin_pix
        return self

    def __str__(self):
        return f"window origin: {self.origin.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}, center: {self.center.detach().cpu().tolist()}, pixelscale: {self.pixelscale.detach().cpu().tolist()}"

    def __repr__(self):
        return (
            f"window pixel_shape: {self.pixel_shape.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}\n"
            + super().__repr__()
        )


class Window_List(Window):
    def __init__(self, window_list=None, state=None):
        if state is not None:
            self.set_state(state)
        else:
            if window_list is None:
                window_list = []
            self.window_list = list(window_list)

        self.check_wcs()

    def check_wcs(self):
        """Ensure the WCS systems being used by all the windows in this list
        are consistent with each other. They should all project world
        coordinates onto the same tangent plane.

        """
        windows = tuple(
            W.reference_radec for W in filter(lambda w: w is not None, self.window_list)
        )
        if len(windows) == 0:
            return
        ref = torch.stack(windows)
        if not torch.allclose(ref, ref[0]):
            raise ConflicingWCS(
                "Reference (world) coordinate mismatch! All windows in Window_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means."
            )

        ref = torch.stack(
            tuple(W.reference_planexy for W in filter(lambda w: w is not None, self.window_list))
        )
        if not torch.allclose(ref, ref[0]):
            raise ConflicingWCS(
                "Reference (tangent plane) coordinate mismatch! All windows in Window_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means."
            )

        if len(set(W.projection for W in filter(lambda w: w is not None, self.window_list))) > 1:
            raise ConflicingWCS(
                "Projection mismatch! All windows in Window_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means."
            )

    @property
    @torch.no_grad()
    def origin(self):
        return tuple(w.origin for w in self)

    @property
    @torch.no_grad()
    def shape(self):
        return tuple(w.shape for w in self)

    @property
    @torch.no_grad()
    def center(self):
        return tuple(w.center for w in self)

    def shift_origin(self, shift):
        raise NotImplementedError("shift origin not implemented for window list")

    def copy(self):
        return self.__class__(list(w.copy() for w in self))

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        for window in self:
            window.to(dtype, device)

    def get_state(self):
        return list(window.get_state() for window in self)

    def set_state(self, state):
        self.window_list = list(Window(state=st) for st in state)

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        new_windows = list((sw | ow) for sw, ow in zip(self, other))
        return self.__class__(window_list=new_windows)

    @torch.no_grad()
    def __ior__(self, other):
        for sw, ow in zip(self, other):
            sw |= ow
        return self

    @torch.no_grad()
    def __and__(self, other):
        new_windows = list((sw & ow) for sw, ow in zip(self, other))
        return self.__class__(window_list=new_windows)

    @torch.no_grad()
    def __iand__(self, other):
        for sw, ow in zip(self, other):
            sw &= ow
        return self

    # Window Comparison operators
    @torch.no_grad()
    def __eq__(self, other):
        results = list((sw == ow).view(-1) for sw, ow in zip(self, other))
        return torch.all(torch.cat(results))

    @torch.no_grad()
    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return len(self.window_list)

    def __iter__(self):
        return (win for win in self.window_list)

    def __str__(self):
        return "Window List: \n" + ("\n".join(list(str(window) for window in self)) + "\n")

    def __repr__(self):
        return "Window List: \n" + ("\n".join(list(repr(window) for window in self)) + "\n")
