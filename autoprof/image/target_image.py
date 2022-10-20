from .image_object import BaseImage
import torch
import numpy as np

class Target_Image(BaseImage):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_variance(kwargs.get("variance", None))
        self.set_psf(kwargs.get("psf", None))

    @property
    def variance(self):
        if self._variance is None:
            return torch.ones(self.data.shape)
        return self._variance

    @property
    def psf(self):
        return self._psf
    @property
    def psf_border(self):
        return self.pixelscale * (1 + np.array(self.psf.shape)) / 2
    @property
    def psf_border_int(self):
        return (1 + np.array(self.psf.shape, dtype = int)) / 2

    def set_variance(self, variance):
        if variance is None:
            self._variance = None
            return
        assert variance.shape == self.data.shape, "variance must have same shape as data"
        self._variance = variance if isinstance(variance, torch.Tensor) else torch.tensor(variance)
        
    def set_psf(self, psf):
        if psf is None:
            self._psf = None
            return
        assert np.all(list(s % 2 == 1 for s in psf.shape)), "psf must have odd shape"
        self._psf = psf if isinstance(psf, torch.Tensor) else torch.tensor(psf)

    def get_window(self, window):
        indices = window.get_indices(self)
        return self.__class__(
            data = self.data[indices],
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            variance = None if self._variance is None else self.variance[indices],
            psf = None if self.psf is None else self.psf[window.get_indices(self)],
            note = self.note,
            origin = (max(self.origin[0], window.origin[0]),
                      max(self.origin[1], window.origin[1]))
        )

