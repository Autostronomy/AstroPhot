from functools import partial

import numpy as np
import torch

from .. import AP_config

__all__ = ["Window", "Window_List"]

class Window(object):
    """class to define a window on the sky in coordinate space. These
    windows can undergo arithmetic an preserve logical behavior. Image
    objects can also be indexed using windows and will return an
    appropriate subsection of their data.

    """
    subclasses = {}

    def __init_subclass__(cls):
        if hasattr(cls, "subname"):
            Window.subclasses[cls.subname] = cls
        
    def __new__(cls, *args, **kwargs):
        if isinstance(kwargs.get("origin", None), torch.Tensor) and kwargs["origin"].dim() == 2:
            return super().__new__(Window.subclasses["batch"])
        if kwargs.get("window_list", None) is not None:
            return super().__new__(Window.subclasses["list"])
        return super().__new__(cls)

    def __init__(self, origin=None, shape=None, center=None, state=None):
        # If loading from a previous state, simply update values and end init
        if state is not None:
            self.update_state(state)
            return
        if center is None and shape is not None and origin is not None:
            self.shape = torch.as_tensor(
                shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.origin = torch.as_tensor(
                origin, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        elif origin is None and center is not None and shape is not None:
            self.shape = torch.as_tensor(
                shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.origin = (
                torch.as_tensor(
                    center, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                - self.shape / 2
            )
        else:
            raise ValueError(
                "One of center or origin must be provided to create window"
            )

    @property
    def center(self):
        return self.origin + self.shape / 2

    @property
    def plt_extent(self):
        return tuple(
            pe.detach().cpu().item()
            for pe in (
                self.origin[0],
                self.origin[0] + self.shape[0],
                self.origin[1],
                self.origin[1] + self.shape[1],
            )
        )

    def copy(self):
        return Window(origin=torch.clone(self.origin), shape=torch.clone(self.shape))

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self.shape = self.shape.to(dtype=dtype, device=device)
        self.origin = self.origin.to(dtype=dtype, device=device)

    def get_shape(self, pixelscale):
        return (torch.round(self.shape / pixelscale)).int()

    def get_shape_flip(self, pixelscale):
        return (torch.round(torch.flip(self.shape, (0,)) / pixelscale)).int()

    @staticmethod
    def _indices(origin_in, shape_in, origin_out, shape_out, pixelscale):
        return torch.max(
            torch.tensor(0, dtype=torch.int, device=AP_config.ap_device),
            (
                torch.round((origin_in[1] - origin_out[1]) / pixelscale)
            ).int(),
        ), torch.min(
            (torch.round(shape_out[1] / pixelscale)).int(),
            (
                torch.round((origin_in[1] + shape_in[1] - origin_out[1]) / pixelscale)
            ).int(),
        ), torch.max(
            torch.tensor(0, dtype=torch.int, device=AP_config.ap_device),
            (
                torch.round((origin_in[0] - origin_out[0]) / pixelscale)
            ).int(),
        ), torch.min(
            (torch.round(shape_out[0] / pixelscale)).int(),
            (
                torch.round((origin_in[0] + shape_in[0] - origin_out[0]) / pixelscale)
            ).int(),
        )
        
    def _get_indices(self, obj_window, obj_pixelscale):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        if isinstance(obj_window, self.__class__.subclasses["batch"]):
            indices = tuple(self._indices(self.origin, self.shape, OO, obj_window.shape, obj_pixelscale) for OO in obj_window.origin)
            return tuple((
                slice(subindices[0], subindices[1]),
                slice(subindices[2], subindices[3]),
            ) for subindices in indices)
        indices = self._indices(self.origin, self.shape, obj_window.origin, obj_window.shape, obj_pixelscale)
        return (
            slice(indices[0], indices[1]),
            slice(indices[2], indices[3]),
        )

    def get_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        return self._get_indices(obj.window, obj.pixelscale)

    def get_coordinate_meshgrid_np(self, pixelscale, x=0.0, y=0.0):
        return np.meshgrid(
            np.linspace(
                (self.origin[0] + pixelscale / 2 - x).detach().cpu().item(),
                (self.origin[0] + self.shape[0] - pixelscale / 2 - x)
                .detach()
                .cpu()
                .item(),
                int(round((self.shape[0].detach().cpu().item() / pixelscale))),
            ),
            np.linspace(
                (self.origin[1] + pixelscale / 2 - y).detach().cpu().item(),
                (self.origin[1] + self.shape[1] - pixelscale / 2 - y)
                .detach()
                .cpu()
                .item(),
                int(round((self.shape[1].detach().cpu().item() / pixelscale))),
            ),
        )

    def get_coordinate_meshgrid_torch(self, pixelscale, c):
        if c is None or c.dim() == 1:
            return torch.meshgrid(
                torch.linspace(
                    (self.origin[0] + pixelscale / 2).detach(),
                    (self.origin[0] + self.shape[0] - pixelscale / 2).detach(),
                    torch.round((self.shape[0] / pixelscale).detach()).int(),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                - c[0],
                torch.linspace(
                    (self.origin[1] + pixelscale / 2).detach(),
                    (self.origin[1] + self.shape[1] - pixelscale / 2).detach(),
                    torch.round((self.shape[1] / pixelscale).detach()).int(),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                - c[1],
                indexing="xy",
            )
        elif c.dim() == 2:
            X = torch.stack(
                tuple(torch.linspace(
                    (O[0] + pixelscale / 2).detach(),
                    (O[0] + self.shape[0] - pixelscale / 2).detach(),
                    torch.round((self.shape[0] / pixelscale).detach()).int(),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                - C[0]
                for O, C in zip(self.origin, c))
            )
            Y = torch.stack(
                tuple(torch.linspace(
                    (O[1] + pixelscale / 2).detach(),
                    (O[1] + self.shape[1] - pixelscale / 2).detach(),
                    torch.round((self.shape[1] / pixelscale).detach()).int(),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                - C[1]
                for O, C in zip(self.origin, c))
            )
            return torch.vmap(partial(torch.meshgrid, indexing = "xy"))(X, Y)
        else:
            raise ValueError(f"Center 'c' should have 1 or 2 dimensions")

    def overlap_frac(self, other):
        overlap = self & other
        overlap_area = torch.prod(overlap.shape)
        full_area = torch.prod(self.shape) + torch.prod(other.shape) - overlap_area
        return overlap_area / full_area

    def shift_origin(self, shift):
        self.origin += shift
        return self

    def get_state(self):
        state = {
            "origin": tuple(self.origin.detach().cpu().tolist()),
            "shape": tuple(self.shape.detach().cpu().tolist()),
        }
        return state

    def update_state(self, state):
        self.origin = torch.tensor(
            state["origin"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.shape = torch.tensor(
            state["shape"], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    # Window adjustment operators
    @torch.no_grad()
    def __add__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_origin = self.origin - other
            new_shape = self.shape + 2 * other
            return self.__class__(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            new_origin = self.origin - torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            new_shape = self.shape + 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self.__class__(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __iadd__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            self.origin -= other
            self.shape += 2 * other
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            self.origin -= torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.shape += 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == (2 * self.origin.shape[-1]):
            self.origin -= torch.as_tensor(
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
    def __sub__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_origin = self.origin + other
            new_shape = self.shape - 2 * other
            return self.__class__(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            new_origin = self.origin + torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            new_shape = self.shape - 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self.__class__(new_origin, new_shape)
        raise ValueError(f"Window object cannot be subtracted with {type(other)}")

    @torch.no_grad()
    def __isub__(self, other):
        if isinstance(other, (float, int, torch.dtype)) or (
            isinstance(other, torch.Tensor) and other.numel() == 1
        ):
            self.origin += other
            self.shape -= 2 * other
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            self.origin += torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.shape -= 2 * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == (2 * self.origin.shape[-1]):
            self.origin += torch.as_tensor(
                other[::2], dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.shape -= torch.as_tensor(
                torch.sum(other.view(-1, 2), axis=0),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            return self
        raise ValueError(f"Window object cannot be subtracted with {type(other)}")

    @torch.no_grad()
    def __mul__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape * other
            new_origin = self.center - new_shape / 2
            return self.__class__(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            new_shape = self.shape * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            new_origin = self.center - new_shape / 2
            return self.__class__(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __imul__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape * other
            self.origin = self.center - new_shape / 2
            self.shape = new_shape
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            new_shape = self.shape * torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.origin = self.center - new_shape / 2
            self.shape = new_shape
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __truediv__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape / other
            new_origin = self.center - new_shape / 2
            return self.__class__(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            new_shape = self.shape / torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            new_origin = self.center - new_shape / 2
            return self.__class__(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")

    @torch.no_grad()
    def __itruediv__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape / other
            self.origin = self.center - new_shape / 2
            self.shape = new_shape
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == self.origin.shape[-1]:
            new_shape = self.shape / torch.as_tensor(
                other, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.origin = self.center - new_shape / 2
            self.shape = new_shape
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
        new_origin = torch.minimum(self.origin.clone(), other.origin)
        new_end = torch.maximum(
            self.origin.clone() + self.shape.clone(), other.origin + other.shape
        )
        return Window(new_origin, new_end - new_origin)

    @torch.no_grad()
    def __ior__(self, other):
        new_origin = torch.minimum(self.origin.clone(), other.origin)
        new_end = torch.maximum(
            self.origin.clone() + self.shape.clone(), other.origin + other.shape
        )
        self.origin = new_origin
        self.shape = new_end - new_origin
        return self

    @torch.no_grad()
    def __and__(self, other):
        if isinstance(other, self.__class__.subclasses["batch"]):
            new_origin = torch.maximum(self.origin.clone(), other.origin)
            new_end = torch.minimum(
                self.origin[0].clone() + self.shape.clone(), other.origin[0] + other.shape
            )
            return self.__class__.subclasses["batch"](new_origin, new_end - new_origin[0])            
        new_origin = torch.maximum(self.origin.clone(), other.origin)
        new_end = torch.minimum(
            self.origin.clone() + self.shape.clone(), other.origin + other.shape
        )
        return Window(new_origin, new_end - new_origin)

    @torch.no_grad()
    def __iand__(self, other):
        new_origin = torch.maximum(self.origin.clone(), other.origin)
        new_end = torch.minimum(
            self.origin.clone() + self.shape.clone(), other.origin + other.shape
        )
        self.origin = new_origin
        self.shape = new_end - new_origin
        return self

    def __str__(self):
        return f"window origin: {self.origin.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}, center: {self.center.detach().cpu().tolist()}"


class Window_List(Window):
    subname = "list"
    
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
        # fixme, this should return a tensor of origins, or a tuple of origin tensors
        origins = torch.cat(list(w.origin.view(-1, 2) for w in self.window_list))
        return torch.min(origins, dim=0)[0]

    @property
    @torch.no_grad()
    def shape(self):
        # fixme, this should return a tensor of shapes, or a tuple of shape tensors
        ends = torch.cat(
            list((w.origin + w.shape).view(-1, 2) for w in self.window_list)
        )
        return torch.max(ends, dim=0)[0] - self.origin

    def shift_origin(self, shift):
        for window, sub_shift in zip(self, shift):
            window.shift_origin(sub_shift)
        return self

    def copy(self):
        return Window_List(list(w.copy() for w in self.window_list))

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        for window in self.window_list:
            window.to(dtype, device)

    def get_state(self):
        return list(window.get_state() for window in self)

    def update_state(self, state):
        self.window_list = list(Window(state=st) for st in state)

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        new_windows = list((sw | ow) for sw, ow in zip(self, other))
        return Window_List(new_windows)

    @torch.no_grad()
    def __ior__(self, other):
        for sw, ow in zip(self, other):
            sw |= ow
        return self

    @torch.no_grad()
    def __and__(self, other):
        new_windows = list((sw & ow) for sw, ow in zip(self, other))
        return Window_List(new_windows)

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
        return Window_List(new_windows)

    @torch.no_grad()
    def __sub__(self, other):
        try:
            new_windows = list(sw - ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw - other for sw in self)
        return Window_List(new_windows)

    @torch.no_grad()
    def __mul__(self, other):
        try:
            new_windows = list(sw * ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw * other for sw in self)
        return Window_List(new_windows)

    @torch.no_grad()
    def __div__(self, other):
        try:
            new_windows = list(sw / ow for sw, ow in zip(self, other))
        except TypeError:
            new_windows = list(sw / other for sw in self)
        return Window_List(new_windows)

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
    def __idiv__(self, other):
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
        return "\n".join(list(str(window) for window in self.window_list)) + "\n"
