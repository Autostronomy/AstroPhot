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
    def __init__(self, origin = None, shape = None, center = None):
        if center is None:
            self.shape = torch.as_tensor(shape, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.origin = torch.as_tensor(origin, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        elif origin is None:
            self.shape = torch.as_tensor(shape, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.origin = torch.as_tensor(center, dtype = AP_config.ap_dtype, device = AP_config.ap_device) - self.shape/2
        else:
            raise Exception("One of center or origin must be provided to create window")
        assert torch.all(self.shape > 0)
        
    @property
    def center(self):
        return self.origin + self.shape/2
    @property
    def plt_extent(self):
        return tuple(pe.detach().cpu().item() for pe in (self.origin[0], self.origin[0] + self.shape[0], self.origin[1], self.origin[1] + self.shape[1]))
    def make_copy(self):
        return Window(origin = torch.clone(self.origin), shape = torch.clone(self.shape))

    def to(self, dtype = None, device = None):
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        self.shape = self.shape.to(dtype = dtype, device = device)
        self.origin = self.origin.to(dtype = dtype, device = device)
    
    def get_shape(self, pixelscale):
        return (torch.round(self.shape / pixelscale)).int()
    def get_shape_flip(self, pixelscale):
        return (torch.round(torch.flip(self.shape, (0,)) / pixelscale)).int()

    def _get_indices(self, obj_window, obj_pixelscale):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        # alignment = ((self.origin + self.shape - obj.origin) / obj.pixelscale)
        # if not np.allclose(alignment/np.round(alignment), 1.):
        #     print(alignment, self.origin, self.shape, obj.origin, obj.pixelscale)
        #     raise ValueError("Cannot determine indices for misaligned windows")
        return (
            slice(torch.max(torch.tensor(0,dtype = torch.int, device = AP_config.ap_device),(torch.round((self.origin[1] - obj_window.origin[1])/obj_pixelscale)).int()),
                  torch.min((torch.round(obj_window.shape[1]/obj_pixelscale)).int(), (torch.round((self.origin[1] + self.shape[1] - obj_window.origin[1])/obj_pixelscale)).int())),
            slice(torch.max(torch.tensor(0,dtype = torch.int, device = AP_config.ap_device),(torch.round((self.origin[0] - obj_window.origin[0])/obj_pixelscale)).int()),
                  torch.min((torch.round(obj_window.shape[0]/obj_pixelscale)).int(), (torch.round((self.origin[0] + self.shape[0] - obj_window.origin[0])/obj_pixelscale)).int())),
        )    
        
    def get_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        return self._get_indices(obj.window, obj.pixelscale)

    def get_coordinate_meshgrid_np(self, pixelscale, x = 0., y = 0.):
        return np.meshgrid(
            np.linspace((self.origin[0] + pixelscale/2 - x).detach().cpu().item(), (self.origin[0] + self.shape[0] - pixelscale/2 - x).detach().cpu().item(), int(round((self.shape[0].detach().cpu().item()/pixelscale)))),
            np.linspace((self.origin[1] + pixelscale/2 - y).detach().cpu().item(), (self.origin[1] + self.shape[1] - pixelscale/2 - y).detach().cpu().item(), int(round((self.shape[1].detach().cpu().item()/pixelscale)))),
        )
    def get_coordinate_meshgrid_torch(self, pixelscale, x = 0., y = 0.):
        return torch.meshgrid(
            torch.linspace((self.origin[0] + pixelscale/2).item(), (self.origin[0] + self.shape[0] - pixelscale/2).item(), torch.round((self.shape[0]/pixelscale)).int().item(), dtype = AP_config.ap_dtype, device = AP_config.ap_device) - x,
            torch.linspace((self.origin[1] + pixelscale/2).item(), (self.origin[1] + self.shape[1] - pixelscale/2).item(), torch.round((self.shape[1]/pixelscale)).int().item(), dtype = AP_config.ap_dtype, device = AP_config.ap_device) - y,
            indexing = 'xy',
        )
        
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
            "origin": tuple(float(o) for o in self.origin.detach().cpu().numpy()),
            "shape": tuple(float(s) for s in self.shape.detach().cpu().numpy()),
        }
        return state

    # Window adjustment operators
    def __add__(self, other):
        if isinstance(other, (float,int, torch.dtype)):
            new_origin = self.origin - other
            new_shape = self.shape + 2*other
            return Window(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            new_origin = self.origin - torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            new_shape = self.shape + 2*torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            return Window(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __iadd__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            self.origin -= other
            self.shape += 2*other
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            self.origin -= torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.shape += 2*torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __sub__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_origin = self.origin - other
            new_shape = self.shape + 2*other
            return Window(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            new_origin = self.origin - torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            new_shape = self.shape + 2*torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            return Window(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __isub__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            self.origin += other
            self.shape -= 2*other
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            self.origin += torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.shape -= 2*torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __mul__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape * other
            new_origin = self.center - new_shape / 2
            return Window(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            new_shape = self.shape * torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            new_origin = self.center - new_shape / 2
            return Window(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __imul__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            self.shape *= other
            self.origin = self.center - new_window_shape / 2
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            self.shape *= torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.origin = self.center - new_window_shape / 2
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __div__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            new_shape = self.shape / other
            new_origin = self.center - new_shape / 2
            return Window(new_origin, new_shape)
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            new_shape = self.shape / torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            new_origin = self.center - new_shape / 2
            return Window(new_origin, new_shape)
        raise ValueError(f"Window object cannot be added with {type(other)}")
    def __idiv__(self, other):
        if isinstance(other, (float, int, torch.dtype)):
            self.shape /= other
            self.origin = self.center - new_window_shape / 2
            return self
        elif isinstance(other, (tuple, torch.Tensor)) and len(other) == len(self.origin):
            self.shape /= torch.as_tensor(other, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.origin = self.center - new_window_shape / 2
            return self
        raise ValueError(f"Window object cannot be added with {type(other)}")

    # Window Comparison operators
    def __eq__(self, other):
        return torch.all(self.origin == other.origin) and torch.all(self.shape == other.shape)
    def __ne__(self, other):
        return not self == other
    def __gt__(self, other):
        return torch.all(self.origin < other.origin) and torch.all((self.origin + self.shape) > (other.origin + other.shape))
    def __ge__(self, other):
        return torch.all(self.origin <= other.origin) and torch.all((self.origin + self.shape) >= (other.origin + other.shape))
    def __lt__(self, other):
        return torch.all(self.origin > other.origin) and torch.all((self.origin + self.shape) < (other.origin + other.shape))
    def __le__(self, other):
        return torch.all(self.origin >= other.origin) and torch.all((self.origin + self.shape) <= (other.origin + other.shape))

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        new_origin = torch.minimum(self.origin.clone(), other.origin)
        new_end = torch.maximum(self.origin.clone() + self.shape.clone(), other.origin + other.shape)
        return Window(new_origin, new_end - new_origin)
    @torch.no_grad()
    def __ior__(self, other):
        new_origin = torch.minimum(self.origin.clone(), other.origin)
        new_end = torch.maximum(self.origin.clone() + self.shape.clone(), other.origin + other.shape)
        self.origin = new_origin
        self.shape = new_end - new_origin
        return self
    @torch.no_grad()
    def __and__(self, other):
        new_origin = torch.maximum(self.origin.clone(), other.origin)
        new_end = torch.minimum(self.origin.clone() + self.shape.clone(), other.origin + other.shape)
        return Window(new_origin, new_end - new_origin)
    @torch.no_grad()
    def __iand__(self, other):
        new_origin = torch.maximum(self.origin.clone(), other.origin)
        new_end = torch.minimum(self.origin.clone() + self.shape.clone(), other.origin + other.shape)
        self.origin = new_origin
        self.shape = new_end - new_origin
        return self
        
    def __str__(self):
        return f"window origin: {list(self.origin.detach().cpu().numpy())}, shape: {list(self.shape.detach().cpu().numpy())}, center: {list(self.center.detach().cpu().numpy())}"

class Window_List(Window):
    def __init__(self, window_list):
        self.window_list = list(window_list)
        
    @property
    @torch.no_grad()
    def origin(self):
        origins = torch.cat(list(w.origin.view(-1,2) for w in self.window_list))
        return torch.min(origins, dim = 0)[0]
    @property
    @torch.no_grad()
    def shape(self):
        ends = torch.cat(list((w.origin + w.shape).view(-1,2) for w in self.window_list))
        return torch.max(ends, dim = 0)[0] - self.origin
    @property
    @torch.no_grad()
    def center(self):
        return self.origin + self.shape/2
    
    def make_copy(self):
        return Window_List(list(w.make_copy() for w in self.window_list))
    
    def to(self, dtype = None, device = None):
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        for window in self.window_list:
            window.to(dtype, device)
            
    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        new_windows = list(sw | ow for sw, ow in zip(self, other))
        return Window_List(new_windows)
    @torch.no_grad()
    def __ior__(self, other):
        for sw, ow in zip(self, other):
            sw |= ow
        return self
    @torch.no_grad()
    def __and__(self, other):
        new_windows = list(sw & ow for sw, ow in zip(self, other))
        return Window_List(new_windows)
    @torch.no_grad()
    def __iand__(self, other):
        for sw, ow in zip(self, other):
            sw &= ow
        return self
    
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index >= len(self.window_list):
            raise StopIteration
        img = self.window_list[self.index]
        self.index += 1
        return img
