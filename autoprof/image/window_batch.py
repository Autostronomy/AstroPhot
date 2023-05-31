from functools import partial

import numpy as np
import torch

from .window_object import Window
from .. import AP_config

__all__ = ["Window_Batch"]

class Window_Batch(Window):
    """class to define a batch of windows on the sky in coordinate space. These
    windows can undergo arithmetic an preserve logical behavior. Image
    objects can also be indexed using windows and will return an
    appropriate subsection of their data.

    """
    subname = "batch"
    
    @property
    def plt_extent(self):
        return tuple(
            pe.detach().cpu().item()
            for pe in (
                torch.min(self.origin[0]),
                torch.max(self.origin[0]) + self.shape[0],
                torch.min(self.origin[1]),
                torch.max(self.origin[1]) + self.shape[1],
            )
        )

    def squish(self):
        """
        Flattens the batch image into a single image which encloses all of the objects
        """
        new_origin = torch.min(self.origin, dim=0)[0]
        new_end = torch.max(self.origin + self.shape, dim=0)[0]
        return Window(origin = new_origin, shape = new_end - new_origin)

    def _get_indices(self, obj_window, obj_pixelscale):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        if isinstance(obj_window, Window_Batch):
            indices = tuple(self._indices(SO, self.shape, OO, obj_window.shape, obj_pixelscale) for SO, OO in zip(self.origin, obj_window.origin))
            return tuple((
                slice(subindices[0], subindices[1]),
                slice(subindices[2], subindices[3]),
            ) for subindices in indices)
        elif isinstance(obj_window, Window):
            indices = tuple(self._indices(SO, self.shape, obj_window.origin, obj_window.shape, obj_pixelscale) for SO in self.origin)
            return tuple((
                slice(subindices[0], subindices[1]),
                slice(subindices[2], subindices[3]),
            ) for subindices in indices)
        else:
            raise ValueError(f"Window_Batch cannot get indices with {type(obj_window)}")

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

    def overlap_frac(self, other):#fixme
        overlap = self & other
        overlap_area = torch.prod(overlap.shape)
        full_area = torch.prod(self.shape) + torch.prod(other.shape) - overlap_area
        return overlap_area / full_area

    # Window interaction operators
    @torch.no_grad()
    def __or__(self, other):
        if isinstance(other, Window_Batch):
            new_origin = torch.minimum(self.origin.clone(), other.origin)
            new_end = torch.maximum(
                self.origin[0].clone() + self.shape.clone(), other.origin[0] + other.shape
            )
            return self.__class__(new_origin, new_end - new_origin[0])
        else:
            raise ValueError(f"Window_Batch cannot perform | with {type(other)}")

    @torch.no_grad()
    def __ior__(self, other):
        if isinstance(other, Window_Batch):
            new_origin = torch.minimum(self.origin.clone(), other.origin)
            new_end = torch.maximum(
                self.origin[0].clone() + self.shape.clone(), other[0].origin + other.shape
            )
            self.origin = new_origin
            self.shape = new_end - new_origin[0]
            return self
        else:
            raise ValueError(f"Window_Batch cannot perform |= with {type(other)}")
            
    @torch.no_grad()
    def __and__(self, other):
        if isinstance(other, Window_Batch):
            new_origin = torch.maximum(self.origin.clone(), other.origin)
            new_end = torch.minimum(
                self.origin[0].clone() + self.shape.clone(), other.origin[0] + other.shape
            )
            return self.__class__(new_origin, new_end - new_origin[0])
        else:
            raise ValueError(f"Window_Batch cannot perform & with {type(other)}")

    @torch.no_grad()
    def __iand__(self, other):
        if isinstance(other, Window_Batch):
            new_origin = torch.maximum(self.origin.clone(), other.origin)
            new_end = torch.minimum(
                self.origin[0].clone() + self.shape.clone(), other.origin[0] + other.shape
            )
            self.origin = new_origin
            self.shape = new_end - new_origin[0]
            return self
        else:
            raise ValueError(f"Window_Batch cannot perform &= with {type(other)}")
            
    def __str__(self):
        return f"window origin: {self.origin.detach().cpu().tolist()}, shape: {self.shape.detach().cpu().tolist()}, center: {self.center.detach().cpu().tolist()}"

    def __len__(self):
        return self.origin.shape[0]
