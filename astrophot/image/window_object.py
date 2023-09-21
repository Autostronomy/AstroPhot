import numpy as np
import torch

from .. import AP_config
from ..utils.conversions.coordinates import Rotate_Cartesian
from .wcs import WCS

__all__ = ["Window", "Window_List"]

class Window(WCS):
    """class to define a window on the sky in coordinate space. These
    windows can undergo arithmetic an preserve logical behavior. Image
    objects can also be indexed using windows and will return an
    appropriate subsection of their data.

    There are several ways to tell a Window object where to
    place itself. The simplest method is to pass an
    Astropy WCS object such as::

      H = ap.image.Window(
          data_shape = data.shape,
          wcs = wcs,
      )

    this will automatically place your image at the correct RA, DEC
    and assign the correct pixel scale. WARNING, it will default to
    setting the reference RA DEC at the reference RA DEC of the wcs
    object; if you have multiple images you should force them all to
    have the same reference world coordiante by passing
    ``reference_radec = (ra, dec)``. See the :doc:`coordinates`
    documentation for more details. There are several other ways to
    initialize the image header. If you provide ``origin_radec`` then
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
      wcs: An astropy.wcs.wcs.WCS object which gives information about the origin and orientation of the window.
      reference_radec: AstroPhot works on a tangent plane where everything is nice and linear, but the sky is a sphere. The worst case would be an image where one of the poles lies inside the image and there is a singularity in the coordinate system. In RA and DEC the approximation works best at `RA = 0`, `DEC = 0` (or really anywhere on the equator). This "good spot" for the coordinates is just an artifact of where we choose to put our coordinates, hence the pole singularity is also artificial. The `reference_radec` is an (RA, DEC) coordinate which can be used to re-orient the polar coordinates so that the "good spot" is anywhere you want. This all happens internally and you should not have to worry about it.
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
        wcs=None,
        **kwargs,
    ):
        # If loading from a previous state, simply update values and end init
        if state is not None:
            self.set_state(state)
            return
        
        # Collect the shape of the window
        assert pixel_shape is not None, "Window must know pixel shape of region (how many pixels on each side)"
        self.pixel_shape = pixel_shape
            
        # Determine relative positioning of tangent plane and pixel grid. Also world coordinates and tangent plane
        assert sum(C is not None for C in [wcs, origin_radec, center_radec, origin, center]) <= 1, "Please provide only one reference position for the window, otherwise the placement is ambiguous"
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
            self.reference_imageij = (-0.5,-0.5)
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
            kwargs.update({
                "reference_imageij": (-0.5,-0.5),
                "reference_imagexy": origin,
            })
            super().__init__(**kwargs)
        # Image reference position from tangent plane position of image center
        elif center is not None:
            pix_center = self.pixel_shape.to(dtype=AP_config.ap_dtype) / 2 - 0.5
            kwargs.update({
                "reference_imageij": pix_center,
                "reference_imagexy": center,                
            })
            super().__init__(**kwargs)
        # Image origin assumed to be at tangent plane origin
        else:
            super().__init__(**kwargs)
        
    @property
    def shape(self):
        S1 = self.pixel_shape.to(dtype=AP_config.ap_dtype)
        S1[1] = 0.
        S2 = self.pixel_shape.to(dtype=AP_config.ap_dtype)
        S2[0] = 0.
        return torch.stack((
            torch.linalg.norm(self.pixelscale @ S1),
            torch.linalg.norm(self.pixelscale @ S2),
        ))
    @shape.setter
    def shape(self, shape):
        if shape is None:
            self._pixel_shape = None
            return
        shape = torch.as_tensor(
            shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.pixel_shape = shape / torch.sqrt(torch.sum(self.pixelscale**2, dim = 0))
        
    @property
    def pixel_shape(self):
        return self._pixel_shape
    @pixel_shape.setter
    def pixel_shape(self, shape):
        if shape is None:
            self._pixel_shape = None
            return
        self._pixel_shape = torch.as_tensor(
            shape, device=AP_config.ap_device
        )
        self._pixel_shape = torch.round(self.pixel_shape).to(dtype=torch.int32, device=AP_config.ap_device)
        
    @property
    def end(self):
        return self.pixel_to_plane_delta(self.pixel_shape.to(dtype=AP_config.ap_dtype))

    @property
    def origin(self):
        return self.pixel_to_plane(-0.5*torch.ones_like(self.reference_imageij))
    
    @property
    def center(self):
        return self.origin + self.end / 2
    
    def copy(self, **kwargs):
        copy_kwargs = {
            "pixel_shape": torch.clone(self.pixel_shape)
        }
        copy_kwargs.update(kwargs)
        return super().copy(
            **copy_kwargs
        )

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)
        self.pixel_shape = self.pixel_shape.to(dtype=dtype, device=device)

    # def get_shape(self, pixelscale):
    #     return (torch.round(torch.linalg.solve(pixelscale, self.end).abs())).int()

    # def get_shape_flip(self, pixelscale):
    #     return torch.flip(self.get_shape(pixelscale), (0,))

    def rescale(self, scale, **kwargs):
        return self.copy(
            pixelscale = self.pixelscale / scale,
            pixel_shape = self.pixel_shape * scale,
            reference_imageij=(self.reference_imageij + 0.5)*scale - 0.5,
            **kwargs,
        )

    @torch.no_grad()
    def _get_indices(self, obj_window):
        other_origin_pix = torch.round(self.plane_to_pixel(obj_window.origin) + 0.5).int()
        new_origin_pix = torch.maximum(torch.zeros_like(other_origin_pix), other_origin_pix)

        other_pixel_end = torch.round(self.plane_to_pixel(obj_window.origin + obj_window.end) + 0.5).int()
        new_pixel_end = torch.minimum(self.pixel_shape, other_pixel_end)
        return slice(new_origin_pix[1], new_pixel_end[1]), slice(new_origin_pix[0], new_pixel_end[0])

        # low = torch.round(self.plane_to_pixel_delta(self.reference_imagexy - obj_window.reference_imagexy) + self.reference_imageij - obj_window.reference_imageij).int()
        # high = torch.round(self.plane_to_pixel_delta(self.reference_imagexy + self.end - obj_window.reference_imagexy) + self.reference_imageij - obj_window.reference_imageij).int()
        # max_pix = torch.round(self.plane_to_pixel_delta(obj_window.end)).int()
        # print(low, high, max_pix)
        # limits = torch.clamp(torch.stack((low, high)), min=torch.zeros_like(high), max=high)
        # print(limits)
        # return slice(limits[0][1], limits[1][1]), slice(limits[0][0], limits[1][0])
        
    # @torch.no_grad()
    # def _get_indices(self, obj_window, obj_pixelscale):
    #     """
    #     Return an index slicing tuple for obj corresponding to this window
    #     """
    #     unclipped_start = torch.round(
    #         torch.linalg.solve(obj_pixelscale, (self.origin - obj_window.origin))
    #     ).int()
    #     unclipped_end = torch.round(
    #         torch.linalg.solve(
    #             obj_pixelscale, (self.origin + self.end - obj_window.origin)
    #         )
    #     ).int()
    #     clipping_end = torch.round(
    #         torch.linalg.solve(obj_pixelscale, obj_window.end)
    #     ).int()
    #     return (
    #         slice(
    #             torch.max(
    #                 torch.tensor(0, dtype=torch.int, device=AP_config.ap_device),
    #                 unclipped_start[1],
    #             ),
    #             torch.min(clipping_end[1], unclipped_end[1]),
    #         ),
    #         slice(
    #             torch.max(
    #                 torch.tensor(0, dtype=torch.int, device=AP_config.ap_device),
    #                 unclipped_start[0],
    #             ),
    #             torch.min(clipping_end[0], unclipped_end[0]),
    #         ),
    #     )

    def get_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        return self._get_indices(obj.window)

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

    def get_state(self):
        state = super().get_state()
        state["pixel_shape"] = tuple(self.pixel_shape.detach().cpu().tolist())
        return state

    def set_state(self, state):
        super().set_state(state)
        self.pixel_shape = torch.tensor(
            state["pixel_shape"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def crop_pixel(self, pixels):
        """
        [crop all sides] or
        [crop x, crop y] or
        [crop x low, crop y low, crop x high, crop y high]
        """
        if len(pixels) == 1:
            self.pixel_shape = self.pixel_shape - 2*pixels[0]
            self.reference_imageij = self.reference_imageij - pixels[0]
        elif len(pixels) == 2:
            pix_shift = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.pixel_shape = self.pixel_shape - 2*pix_shift
            self.reference_imageij = self.reference_imageij - pix_shift
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.pixel_shape = self.pixel_shape - pixels[::2] - pixels[1::2]
            self.reference_imageij = self.reference_imageij - pixels[::2]
        else:
            raise ValueError(f"Unrecognized pixel crop format: {pixels}")
        return self

    def crop_to_pixel(self, pixels):
        """
        format: [[xmin, xmax],[ymin,ymax]]
        """
        pixels = torch.tensor(pixels,dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        self.reference_imageij = self.reference_imageij - pixels[:,0]
        self.pixel_shape = pixels[:,1] - pixels[:,0]
        return self

    def pad_pixel(self, pixels):
        """
        [pad all sides] or
        [pad x, pad y] or
        [pad x low, pad y low, pad x high, pad y high]
        """
        if len(pixels) == 1:
            self.pixel_shape = self.pixel_shape + 2*pixels[0]
            self.reference_imageij = self.reference_imageij + pixels[0]
        elif len(pixels) == 2:
            pix_shift = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.pixel_shape = self.pixel_shape + 2*pix_shift
            self.reference_imageij = self.reference_imageij + pix_shift
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.pixel_shape = self.pixel_shape + pixels[::2] + pixels[1::2]
            self.reference_imageij = self.reference_imageij + pixels[::2]
        else:
            raise ValueError(f"Unrecognized pixel crop format: {pixels}")
        return self

    @torch.no_grad()
    def get_coordinate_meshgrid(self):
        """Returns a meshgrid with tangent plane coordinates for the center
        of every pixel.

        """
        pix = self.pixel_shape.to(dtype=AP_config.ap_dtype)
        xsteps = torch.arange(
            pix[0], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        ysteps = torch.arange(
            pix[1], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
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
            torch.arange(
                pix[0] + 1, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            - 0.5
        )
        ysteps = (
            torch.arange(
                pix[1] + 1, dtype=AP_config.ap_dtype, device=AP_config.ap_device
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

    # # Window adjustment operators
    # @torch.no_grad()
    # def __add__(self, other):
    #     """Add to the size of the window. This operation preserves the window
    #     center and changes the size (shape) of the window by
    #     increasing the border.

    #     """
    #     if isinstance(other, (float, int, torch.dtype)):
    #         new_shape = self.shape + 2 * other
    #         return self.copy(
    #             center=self.center,
    #             shape=new_shape,
    #             pixel_shape=None,
    #         )
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         new_shape = self.shape + 2 * torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         return self.copy(
    #             center=self.center,
    #             shape=new_shape,
    #             pixel_shape=None,
    #         )
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __iadd__(self, other):
    #     if isinstance(other, (float, int, torch.dtype)):
    #         keep_center = self.center.clone()
    #         self.shape += 2 * other
    #         self.center = keep_center
    #         return self
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         keep_center = self.center.clone()
    #         self.shape += 2 * torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         self.center = keep_center
    #         return self
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == (
    #         2 * len(self.origin)
    #     ):
    #         self.origin -= self.cartesian_to_plane(
    #             torch.as_tensor(
    #                 other[::2], dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #             )
    #         )
    #         self.shape -= torch.as_tensor(
    #             torch.sum(other.view(-1, 2), axis=0),
    #             dtype=AP_config.ap_dtype,
    #             device=AP_config.ap_device,
    #         )
    #         return self
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __sub__(self, other):
    #     """Reduce the size of the window. This operation preserves the window
    #     center and changes the size (shape) of the window by reducing
    #     the border.

    #     """
    #     if isinstance(other, (float, int, torch.dtype)):
    #         new_shape = self.shape - 2 * other
    #         return self.__class__(
    #             center=self.center,
    #             shape=new_shape,
    #             pixelshape=self.pixelshape,
    #             projection=self.projection,
    #             reference_radec=self.reference_radec,
    #         )
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         new_shape = self.shape - 2 * torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         return self.__class__(
    #             center=self.center,
    #             shape=new_shape,
    #             pixelshape=self.pixelshape,
    #             projection=self.projection,
    #             reference_radec=self.reference_radec,
    #         )
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __isub__(self, other):
    #     if isinstance(other, (float, int, torch.dtype)) or (
    #         isinstance(other, torch.Tensor) and other.numel() == 1
    #     ):
    #         keep_center = self.center.clone()
    #         self.shape -= 2 * other
    #         self.center = keep_center
    #         return self
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         keep_center = self.center.clone()
    #         self.shape -= 2 * torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         self.center = keep_center
    #         return self
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == (
    #         2 * len(self.origin)
    #     ):
    #         self.origin += torch.as_tensor(
    #             other[::2], dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         self.shape -= torch.as_tensor(
    #             torch.sum(other.view(-1, 2), axis=0),
    #             dtype=AP_config.ap_dtype,
    #             device=AP_config.ap_device,
    #         )
    #         return self
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __mul__(self, other):
    #     """Add to the size of the window. This operation preserves the window
    #     center and changes the size (shape) of the window by
    #     multiplying the border.

    #     """
    #     if isinstance(other, (float, int, torch.dtype)):
    #         new_shape = self.shape * other
    #         return self.__class__(
    #             center=self.center,
    #             shape=new_shape,
    #             pixelshape=self.pixelshape,
    #             projection=self.projection,
    #             reference_radec=self.reference_radec,
    #         )
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         new_shape = self.shape * torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         return self.__class__(
    #             center=self.center,
    #             shape=new_shape,
    #             pixelshape=self.pixelshape,
    #             projection=self.projection,
    #             reference_radec=self.reference_radec,
    #         )
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __imul__(self, other):
    #     if isinstance(other, (float, int, torch.dtype)):
    #         keep_center = self.center.clone()
    #         self.shape *= other
    #         self.center = keep_center
    #         return self
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         keep_center = self.center.clone()
    #         self.shape *= torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         self.center = keep_center
    #         return self
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __truediv__(self, other):
    #     """Reduce the size of the window. This operation preserves the window
    #     center and changes the size (shape) of the window by
    #     dividing the border.

    #     """
    #     if isinstance(other, (float, int, torch.dtype)):
    #         new_shape = self.shape / other
    #         return self.__class__(
    #             center=self.center,
    #             shape=new_shape,
    #             pixelshape=self.pixelshape,
    #             projection=self.projection,
    #             reference_radec=self.reference_radec,
    #         )
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         new_shape = self.shape / torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         return self.__class__(
    #             center=self.center,
    #             shape=new_shape,
    #             pixelshape=self.pixelshape,
    #             projection=self.projection,
    #             reference_radec=self.reference_radec,
    #         )
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # @torch.no_grad()
    # def __itruediv__(self, other):
    #     if isinstance(other, (float, int, torch.dtype)):
    #         keep_center = self.center.clone()
    #         self.shape /= other
    #         self.center = keep_center
    #         return self
    #     elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
    #         self.origin
    #     ):
    #         keep_center = self.center.clone()
    #         self.shape /= torch.as_tensor(
    #             other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
    #         )
    #         self.center = keep_center
    #         return self
    #     raise ValueError(f"Window object cannot be added with {type(other)}")

    # Window Comparison operators
    @torch.no_grad()
    def __eq__(self, other):
        return torch.all(
            self.pixel_shape == other.pixel_shape
        ) and torch.all(
            self.pixelscale == other.pixelscale
        ) and (
            self.projection == other.projection
        ) and (
            torch.all(self.pixel_to_plane(torch.zeros_like(self.reference_imageij)) == other.pixel_to_plane(torch.zeros_like(other.reference_imageij)))
        ) # fixme more checks?
    

    @torch.no_grad()
    def __ne__(self, other):
        return not self == other

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.minimum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.maximum(self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end)
        return self.copy(
            origin=self.pixel_to_plane(new_origin_pix),
            pixel_shape=new_pixel_end - new_origin_pix,
        )

    @torch.no_grad()
    def __ior__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.minimum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.maximum(self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end)

        self.reference_imageij = self.reference_imageij - (new_origin_pix + 0.5)
        self.pixel_shape = new_pixel_end - new_origin_pix
        return self

    @torch.no_grad()
    def __and__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.maximum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.minimum(self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end)
        return self.copy(
            origin=self.pixel_to_plane(new_origin_pix),
            pixel_shape=new_pixel_end - new_origin_pix,
        )

    @torch.no_grad()
    def __iand__(self, other):
        other_origin_pix = self.plane_to_pixel(other.origin)
        new_origin_pix = torch.maximum(-0.5 * torch.ones_like(other_origin_pix), other_origin_pix)

        other_pixel_end = self.plane_to_pixel(other.origin + other.end)
        new_pixel_end = torch.minimum(self.pixel_shape.to(dtype=AP_config.ap_dtype), other_pixel_end)

        self.reference_imageij = self.reference_imageij - (new_origin_pix + 0.5)
        self.pixel_shape = new_pixel_end - new_origin_pix
        return self

    def __str__(self):
        return f"window origin: {self.origin.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}, center: {self.center.detach().cpu().tolist()}, pixelscale: {self.pixelscale.detach().tolist()}"

    def __repr__(self):
        return f"window pixel_shape: {self.pixel_shape.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}\n" + super().__repr__()


class Window_List(Window):
    def __init__(self, window_list=None, state=None):
        if state is not None:
            self.set_state(state)
        else:
            assert (
                window_list is not None
            ), "window_list must be a list of Window objects"
            self.window_list = list(window_list)

        self.check_wcs()
        
    def check_wcs(self):
        """Ensure the WCS system being used by all the windows in this list
        are consistent with each other. They should all project world
        coordinates onto the same tangent plane.

        """
        ref = torch.stack(tuple(W.reference_radec for W in self.window_list))
        if not torch.allclose(ref, ref[0]):
            AP_config.error("Reference (world) coordinate missmatch! All windows in Window_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means.")

        ref = torch.stack(tuple(W.reference_planexy for W in self.window_list))
        if not torch.allclose(ref, ref[0]):
            AP_config.error("Reference (tangent plane) coordinate missmatch! All windows in Window_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means.")

        if len(set(W.projection for W in self.window_list)) > 1:
            AP_config.error("Projection missmatch! All windows in Window_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means.")
            
            
    @property
    @torch.no_grad()
    def origin(self):
        return tuple(w.origin for w in self)

    @property
    @torch.no_grad()
    def shape(self):
        return tuple(w.shape for w in self)

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

    # @torch.no_grad()
    # def __gt__(self, other):
    #     results = list((sw > ow).view(-1) for sw, ow in zip(self, other))
    #     return torch.all(torch.cat(results))

    # @torch.no_grad()
    # def __ge__(self, other):
    #     results = list((sw >= ow).view(-1) for sw, ow in zip(self, other))
    #     return torch.all(torch.cat(results))

    # @torch.no_grad()
    # def __lt__(self, other):
    #     results = list((sw < ow).view(-1) for sw, ow in zip(self, other))
    #     return torch.all(torch.cat(results))

    # @torch.no_grad()
    # def __le__(self, other):
    #     results = list((sw <= ow).view(-1) for sw, ow in zip(self, other))
    #     return torch.all(torch.cat(results))

    # # Window adjustment operators
    # @torch.no_grad()
    # def __add__(self, other):
    #     try:
    #         new_windows = list(sw + ow for sw, ow in zip(self, other))
    #     except TypeError:
    #         new_windows = list(sw + other for sw in self)
    #     return self.__class__(window_list=new_windows)

    # @torch.no_grad()
    # def __sub__(self, other):
    #     try:
    #         new_windows = list(sw - ow for sw, ow in zip(self, other))
    #     except TypeError:
    #         new_windows = list(sw - other for sw in self)
    #     return self.__class__(window_list=new_windows)

    # @torch.no_grad()
    # def __mul__(self, other):
    #     try:
    #         new_windows = list(sw * ow for sw, ow in zip(self, other))
    #     except TypeError:
    #         new_windows = list(sw * other for sw in self)
    #     return self.__class__(window_list=new_windows)

    # @torch.no_grad()
    # def __truediv__(self, other):
    #     try:
    #         new_windows = list(sw / ow for sw, ow in zip(self, other))
    #     except TypeError:
    #         new_windows = list(sw / other for sw in self)
    #     return self.__class__(window_list=new_windows)

    # @torch.no_grad()
    # def __iadd__(self, other):
    #     try:
    #         for sw, ow in zip(self, other):
    #             sw += ow
    #     except TypeError:
    #         for sw in self:
    #             sw += other
    #     return self

    # @torch.no_grad()
    # def __isub__(self, other):
    #     try:
    #         for sw, ow in zip(self, other):
    #             sw -= ow
    #     except TypeError:
    #         for sw in self:
    #             sw -= other
    #     return self

    # @torch.no_grad()
    # def __imul__(self, other):
    #     try:
    #         for sw, ow in zip(self, other):
    #             sw *= ow
    #     except TypeError:
    #         for sw in self:
    #             sw *= other
    #     return self

    # @torch.no_grad()
    # def __itruediv__(self, other):
    #     try:
    #         for sw, ow in zip(self, other):
    #             sw /= ow
    #     except TypeError:
    #         for sw in self:
    #             sw /= other
    #     return self

    def __len__(self):
        return len(self.window_list)

    def __iter__(self):
        return (win for win in self.window_list)

    def __str__(self):
        return "Window List: \n" + (
            "\n".join(list(str(window) for window in self)) + "\n"
        )

    def __repr__(self):
        return "Window List: \n" + (
            "\n".join(list(repr(window) for window in self)) + "\n"
        )
