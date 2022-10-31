from .image_object import BaseImage
import torch
import numpy as np
from torch.nn.functional import avg_pool2d

class Target_Image(BaseImage):
    """Image object which represents the data to be fit by a model. It can
    include a variance image, mask, and PSF as anciliary data which
    describes the target image.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_variance(kwargs.get("variance", None))
        self.set_mask(kwargs.get("mask", None))
        self.set_psf(kwargs.get("psf", None))

    @property
    def variance(self):
        if self._variance is None:
            return torch.ones(self.data.shape, dtype = torch.float32)
        return self._variance
    @variance.setter
    def variance(self, variance):
        self.set_variance(variance)
    @property
    def mask(self):
        if self._mask is None:
            return np.zeros(self.data.shape, dtype = np.bool)
        return self._mask
    @mask.setter
    def mask(self, mask):
        self.set_mask(mask)
    @property
    def masked(self):
        return self._mask is not None
    @property
    def psf(self):
        return self._psf
    @psf.setter
    def psf(self, psf):
        self.set_psf(psf)
    @property
    def psf_border(self):
        return tuple(self.pixelscale * (1 + np.flip(np.array(self.psf.shape))) / 2)
    @property
    def psf_border_int(self):
        return tuple(int(pb) for pb in ((1 + np.flip(np.array(self.psf.shape))) / 2))

    def set_variance(self, variance):
        if variance is None:
            self._variance = None
            return
        assert variance.shape == self.data.shape, "variance must have same shape as data"
        self._variance = variance if isinstance(variance, torch.Tensor) else torch.tensor(variance, dtype = torch.float64)
        
    def set_psf(self, psf):
        if psf is None:
            self._psf = None
            return
        assert np.all(list((s % 2) == 1 for s in psf.shape)), "psf must have odd shape"
        self._psf = psf if isinstance(psf, torch.Tensor) else torch.tensor(psf, dtype = torch.float64)

    def set_mask(self, mask):
        if mask is None:
            self._mask = None
            return
        assert mask.shape == self.data.shape, "mask must have same shape as data"
        self._mask = mask
        
    def or_mask(self, mask):
        self._mask = np.logical_or(self.mask, mask)
    def and_mask(self, mask):
        self._mask = np.logical_and(self.mask, mask)
        
    def blank_copy(self):
        return self.__class__(
            data = torch.zeros(self.data.shape, dtype = torch.float32),
            zeropoint = self.zeropoint,
            mask = None if self._mask is None else self._mask,
            psf = None if self._psf is None else self.psf,
            note = self.note,
            window = self.window,
        )
    
    def get_window(self, window):
        indices = window.get_indices(self)
        return self.__class__(
            data = self.data[indices],
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            variance = None if self._variance is None else self._variance[indices],
            mask = None if self._mask is None else self._mask[indices],
            psf = None if self._psf is None else self.psf,
            note = self.note,
            origin = (max(self.origin[0], window.origin[0]),
                      max(self.origin[1], window.origin[1]))
        )
    
    def reduce(self, scale):
        assert isinstance(scale, int)
        assert scale > 1

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale
        return self.__class__(
            data = self.data.detach().numpy()[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)),
            pixelscale = self.pixelscale * scale,
            zeropoint = self.zeropoint,
            variance = None if self._variance is None else self.variance.detach().numpy()[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)),
            mask = None if self._mask is None else self.mask.detach().numpy()[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).max(axis=(1, 3)),
            psf = None if self._psf is None else self.psf.detach().numpy()[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)),
            note = self.note,
            origin = self.origin,
        )
            
