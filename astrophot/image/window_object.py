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

    Args:
      origin: the position of the bottom left corner of the window. Note that for an image, the origin corresponds to the pixel location -0.5, -0.5 since the pixel coordinates have zero at the center of the pixel while the on sky bounding box should include the size of the pixels as well and so will be half a pixel width shifted. [arcsec]
      origin_radec: The position of outer corner of the (0,0) pixel given as Right Ascention and Declination (RA, DEC) in degrees. The most straightforward usage of `origin_radec` is to call `C = wcs.pixel_to_world(-0.5,-0.5)` which will give a SkyCoord object, then call `origin_radec = (C.ra.deg, C.dec.deg)` as an argument to the target image. **Note** see also `reference_radec`.
      shape: the length of the sides of the window in physical units [arcsec]
      projection: A det = 1 matrix which describes the projection of coordinates on the sky. If nothing is given then an identity matrix is assumed. If a wcs object is given then the projection is the pixel scale matrix normalized to determinant of 1. Essentially this matrix described the transformation from a simple cartesian grid onto the sky which may be flipped, rotated or streched. [unitless 2x2 matrix]
      center: Instead of providing the origin, one can provide the center position. This will just be used to update the origin. [arcsec]
      center_radec: Similar to `origin_radec` one may provide the RA and DEC of the center of the image in degrees. **Note** see also `reference_radec` below.
      state: A dictionary containing the origin, shape, and orientation information. [dict]
      wcs: An astropy.wcs.wcs.WCS object which gives information about the origin and orientation of the window.
      reference_radec: AstroPhot works on a tangent plane where everything is nice and linear, but the sky is a sphere. The worst case would be an image where one of the poles lies inside the image and there is a singularity in the coordinate system. In RA and DEC the approximation works best at `RA = 0`, `DEC = 0` (or really anywhere on the equator). This "good spot" for the coordinates is just an artifact of where we choose to put our coordinates, hence the pole singularity is also artificial. The `reference_radec` is an (RA, DEC) coordinate which can be used to re-orient the polar coordinates so that the "good spot" is anywhere you want. This all happens internally and you should not have to worry about it.   
    """

    def __init__(
        self,
        origin=None,
        origin_radec=None,
        shape=None,
        projection=None,
        center=None,
        center_radec=None,
        state=None,
        wcs=None,
        **kwargs,
    ):
        # If loading from a previous state, simply update values and end init
        if state is not None:
            self.update_state(state)
            return

        if projection is not None: # Projection matrix provided
            projection = torch.as_tensor(
                projection, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            # normalize determinant to area of 1
            self.projection = projection / torch.linalg.det(projection).abs().sqrt()
        if wcs is not None: # Projection matrix from WCS
            proj = wcs.pixel_scale_matrix
            proj /= np.sqrt(np.abs(np.linalg.det(proj)))
            self.projection = torch.tensor(
                proj, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        else: # Projection matrix assumed to be identity
            self.projection = torch.eye(
                2, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

        self.shape = torch.as_tensor(
            shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        
        # Determine origin
        if origin_radec is not None:
            origin_radec = torch.as_tensor(
                origin_radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            super().__init__(reference_radec = origin_radec)
            self.origin = self.world_to_plane(
                origin_radec
            )
        elif center_radec is not None:
            center_radec = torch.as_tensor(
                center_radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            super().__init__(reference_radec = center_radec)
            self.center = self.world_to_plane(
                center_radec
            )
        elif origin is not None:
            self.origin = torch.as_tensor(
                origin, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        elif center is not None:
            self.center = torch.as_tensor(
                center, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        elif wcs is not None:
            wcs_origin = wcs.pixel_to_world(-0.5, -0.5)
            wcs_origin = torch.as_tensor(
                [wcs_origin.ra.deg, wcs_origin.dec.deg],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            super().__init__(reference_radec = wcs_origin)
            self.origin = self.world_to_plane(
                wcs_origin
            )
        else:
            raise ValueError(
                "One of center or origin must be provided to create window"
            )

    @property
    def end(self):
        return self.cartesian_to_plane(self.shape)

    @property
    def center(self):
        return self.origin + self.end / 2

    @center.setter
    def center(self, c):
        self.origin = (
            torch.as_tensor(c, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            - self.end / 2
        )

    def plane_to_cartesian(self, world_coordinate):
        """Projects a tangent plane coordinate which may be rotated, flipped, or
        sheered into a regular square cartesian grid for the purpose
        of comparisons and where arithmetic is more straightforward.

        """
        return torch.linalg.solve(self.projection, world_coordinate)

    def cartesian_to_plane(self, cartesian_coordinate):
        return self.projection @ cartesian_coordinate

    def copy(self):
        return self.__class__(
            origin=torch.clone(self.origin),
            shape=torch.clone(self.shape),
            projection=torch.clone(self.projection),
        )

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self.shape = self.shape.to(dtype=dtype, device=device)
        self.origin = self.origin.to(dtype=dtype, device=device)
        self.projection = self.projection.to(dtype=dtype, device=device)

    def get_shape(self, pixelscale):
        return (torch.round(torch.linalg.solve(pixelscale, self.end).abs())).int()

    def get_shape_flip(self, pixelscale):
        return torch.flip(self.get_shape(pixelscale), (0,))

    @torch.no_grad()
    def _get_indices(self, obj_window, obj_pixelscale):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        unclipped_start = torch.round(
            torch.linalg.solve(obj_pixelscale, (self.origin - obj_window.origin))
        ).int()
        unclipped_end = torch.round(
            torch.linalg.solve(
                obj_pixelscale, (self.origin + self.end - obj_window.origin)
            )
        ).int()
        clipping_end = torch.round(
            torch.linalg.solve(obj_pixelscale, obj_window.end)
        ).int()
        return (
            slice(
                torch.max(
                    torch.tensor(0, dtype=torch.int, device=AP_config.ap_device),
                    unclipped_start[1],
                ),
                torch.min(clipping_end[1], unclipped_end[1]),
            ),
            slice(
                torch.max(
                    torch.tensor(0, dtype=torch.int, device=AP_config.ap_device),
                    unclipped_start[0],
                ),
                torch.min(clipping_end[0], unclipped_end[0]),
            ),
        )

    def get_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        return self._get_indices(obj.window, obj.pixelscale)

    def overlap_frac(self, other):
        overlap = self & other
        overlap_area = torch.prod(overlap.shape)
        full_area = torch.prod(self.shape) + torch.prod(other.shape) - overlap_area
        return overlap_area / full_area

    def shift_origin(self, shift):
        """
        Shift the origin of the window by a specified amount in tangent plane coordinates
        """
        self.origin += shift
        return self

    def get_state(self):
        state = {
            "origin": tuple(self.origin.detach().cpu().tolist()),
            "shape": tuple(self.shape.detach().cpu().tolist()),
            "projection": tuple(tuple(p) for p in self.projection.detach().tolist()),
        }
        return state

    def update_state(self, state):
        self.origin = torch.tensor(
            state["origin"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.shape = torch.tensor(
            state["shape"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.projection = torch.tensor(
            state["projection"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    # Window adjustment operators
    @torch.no_grad()
    def __add__(self, other):
        """Add to the size of the window. This operation preserves the window
        center and changes the size (shape) of the window by
        increasing the border.

        """
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape + 2 * other
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            new_shape = self.shape + 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __iadd__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            keep_center = self.center.clone()
            self.shape += 2 * other
            self.center = keep_center
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            keep_center = self.center.clone()
            self.shape += 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.center = keep_center
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == (
            2 * len(self.origin)
        ):
            self.origin -= self.cartesian_to_plane(
                torch.as_tensor(
                    other[::2], dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            )
            self.shape -= torch.as_tensor(
                torch.sum(other.view(-1, 2), axis=0),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __sub__(self, other):
        """Reduce the size of the window. This operation preserves the window
        center and changes the size (shape) of the window by reducing
        the border.

        """
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape - 2 * other
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            new_shape = self.shape - 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __isub__(self, other):
        if isinstance(other, (float, int, torch.dtype)) or (
            isinstance(other, torch.Tensor) and other.numel() == 1
        ):
            keep_center = self.center.clone()
            self.shape -= 2 * other
            self.center = keep_center
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            keep_center = self.center.clone()
            self.shape -= 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.center = keep_center
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == (
            2 * len(self.origin)
        ):
            self.origin += torch.as_tensor(
                other[::2], dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.shape -= torch.as_tensor(
                torch.sum(other.view(-1, 2), axis=0),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __mul__(self, other):
        """Add to the size of the window. This operation preserves the window
        center and changes the size (shape) of the window by
        multiplying the border.

        """
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape * other
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            new_shape = self.shape * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __imul__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            keep_center = self.center.clone()
            self.shape *= other
            self.center = keep_center
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            keep_center = self.center.clone()
            self.shape *= torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.center = keep_center
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __truediv__(self, other):
        """Reduce the size of the window. This operation preserves the window
        center and changes the size (shape) of the window by
        dividing the border.

        """
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape / other
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            new_shape = self.shape / torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self.__class__(
                center=self.center, shape=new_shape, projection=self.projection
            )
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __itruediv__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            keep_center = self.center.clone()
            self.shape /= other
            self.center = keep_center
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(
            self.origin
        ):
            keep_center = self.center.clone()
            self.shape /= torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.center = keep_center
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")

    # Window Comparison operators
    @torch.no_grad()
    def __eq__(self, other):
        return torch.all(self.origin == other.origin) and torch.all(
            self.shape == other.shape
        )

    @torch.no_grad()
    def __ne__(self, other):
        return not self == other

    @torch.no_grad()
    def __gt__(self, other):
        return torch.all(self.origin < other.origin) and torch.all(
            (self.origin + self.shape) > (other.origin + other.shape)
        )

    @torch.no_grad()
    def __ge__(self, other):
        return torch.all(self.origin <= other.origin) and torch.all(
            (self.origin + self.shape) >= (other.origin + other.shape)
        )

    @torch.no_grad()
    def __lt__(self, other):
        return torch.all(self.origin > other.origin) and torch.all(
            (self.origin + self.shape) < (other.origin + other.shape)
        )

    @torch.no_grad()
    def __le__(self, other):
        return torch.all(self.origin >= other.origin) and torch.all(
            (self.origin + self.shape) <= (other.origin + other.shape)
        )

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        cart_self_origin = self.plane_to_cartesian(self.origin)
        cart_other_origin = self.plane_to_cartesian(other.origin)

        new_origin = torch.minimum(cart_self_origin, cart_other_origin)
        new_end = torch.maximum(
            cart_self_origin + self.shape, cart_other_origin + other.shape
        )
        return self.__class__(
            origin=self.cartesian_to_plane(new_origin),
            shape=new_end - new_origin,
            projection=self.projection,
        )

    @torch.no_grad()
    def __ior__(self, other):
        cart_self_origin = self.plane_to_cartesian(self.origin)
        cart_other_origin = self.plane_to_cartesian(other.origin)

        new_origin = torch.minimum(cart_self_origin, cart_other_origin)
        new_end = torch.maximum(
            cart_self_origin + self.shape, cart_other_origin + other.shape
        )
        self.origin = self.cartesian_to_plane(new_origin)
        self.shape = new_end - new_origin
        return self

    @torch.no_grad()
    def __and__(self, other):
        cart_self_origin = self.plane_to_cartesian(self.origin)
        cart_other_origin = self.plane_to_cartesian(other.origin)

        new_origin = torch.maximum(cart_self_origin, cart_other_origin)
        new_end = torch.minimum(
            cart_self_origin + self.shape, cart_other_origin + other.shape
        )
        return self.__class__(
            self.cartesian_to_plane(new_origin),
            new_end - new_origin,
            projection=self.projection,
        )

    @torch.no_grad()
    def __iand__(self, other):
        cart_self_origin = self.plane_to_cartesian(self.origin)
        cart_other_origin = self.plane_to_cartesian(other.origin)

        new_origin = torch.maximum(cart_self_origin, cart_other_origin)
        new_end = torch.minimum(
            cart_self_origin + self.shape, cart_other_origin + other.shape
        )
        self.origin = self.cartesian_to_plane(new_origin)
        self.shape = new_end - new_origin
        return self

    def __str__(self):
        return f"window origin: {self.origin.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}, center: {self.center.detach().cpu().tolist()}, projection: {self.projection.detach().tolist()}"
    def __repr__(self):
        return f"window origin: {self.origin.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}, center: {self.center.detach().cpu().tolist()}, projection: {self.projection.detach().tolist()}"


class Window_List(Window):
    def __init__(self, window_list=None, state=None):
        if state is not None:
            self.update_state(state)
        else:
            assert (
                window_list is not None
            ), "window_list must be a list of Window objects"
            self.window_list = list(window_list)

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

    def update_state(self, state):
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

    @torch.no_grad()
    def __gt__(self, other):
        results = list((sw > ow).view(-1) for sw, ow in zip(self, other))
        return torch.all(torch.cat(results))

    @torch.no_grad()
    def __ge__(self, other):
        results = list((sw >= ow).view(-1) for sw, ow in zip(self, other))
        return torch.all(torch.cat(results))

    @torch.no_grad()
    def __lt__(self, other):
        results = list((sw < ow).view(-1) for sw, ow in zip(self, other))
        return torch.all(torch.cat(results))

    @torch.no_grad()
    def __le__(self, other):
        results = list((sw <= ow).view(-1) for sw, ow in zip(self, other))
        return torch.all(torch.cat(results))

    # Window adjustment operators
    @torch.no_grad()
    def __add__(self, other):
        try:
            new_windows = list(sw + ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw + other for sw in self)
        return self.__class__(window_list=new_windows)

    @torch.no_grad()
    def __sub__(self, other):
        try:
            new_windows = list(sw - ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw - other for sw in self)
        return self.__class__(window_list=new_windows)

    @torch.no_grad()
    def __mul__(self, other):
        try:
            new_windows = list(sw * ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw * other for sw in self)
        return self.__class__(window_list=new_windows)

    @torch.no_grad()
    def __truediv__(self, other):
        try:
            new_windows = list(sw / ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw / other for sw in self)
        return self.__class__(window_list=new_windows)

    @torch.no_grad()
    def __iadd__(self, other):
        try:
            for sw, ow in zip(self, other):
                sw += ow
        except TypeError:
            for sw in self:
                sw += other
        return self

    @torch.no_grad()
    def __isub__(self, other):
        try:
            for sw, ow in zip(self, other):
                sw -= ow
        except TypeError:
            for sw in self:
                sw -= other
        return self

    @torch.no_grad()
    def __imul__(self, other):
        try:
            for sw, ow in zip(self, other):
                sw *= ow
        except TypeError:
            for sw in self:
                sw *= other
        return self

    @torch.no_grad()
    def __itruediv__(self, other):
        try:
            for sw, ow in zip(self, other):
                sw /= ow
        except TypeError:
            for sw in self:
                sw /= other
        return self

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
