from .image_object import BaseImage, Image_List
import torch
import numpy as np
from torch.nn.functional import avg_pool2d
from astropy.io import fits
from .. import AP_config
__all__ = ["Target_Image", "Target_Image_List"]

class Target_Image(BaseImage):
    """Image object which represents the data to be fit by a model. It can
    include a variance image, mask, and PSF as anciliary data which
    describes the target image.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.has_variance:
            self.set_variance(kwargs.get("variance", None))
        if not self.has_mask:
            self.set_mask(kwargs.get("mask", None))
        if not self.has_psf:
            self.set_psf(kwargs.get("psf", None))
        self.psf_upscale = torch.as_tensor(kwargs.get("psf_upscale", 1), dtype = torch.int32, device = AP_config.ap_device) 

    @property
    def variance(self):
        if self.has_variance:
            return self._variance
        return torch.ones_like(self.data)
    @variance.setter
    def variance(self, variance):
        self.set_variance(variance)
    @property
    def has_variance(self):
        try:
            return self._variance is not None
        except AttributeError:
            return False
    
    @property
    def mask(self):
        if self.has_mask:
            return self._mask
        return torch.zeros_like(self.data, dtype = torch.bool)
    @mask.setter
    def mask(self, mask):
        self.set_mask(mask)
    @property
    def has_mask(self):
        try:
            return self._mask is not None
        except AttributeError:
            return False
    
    @property
    def psf(self):
        if self.has_psf:
            return self._psf
        raise AttributeError("This image does not have a PSF")
    @psf.setter
    def psf(self, psf):
        self.set_psf(psf)
    @property
    def psf_border_int(self):
        return torch.ceil((1 + torch.flip(torch.tensor(self.psf.shape, dtype = AP_config.ap_dtype, device = AP_config.ap_device), (0,))/self.psf_upscale) / 2).int()
    @property
    def psf_border(self):
        return self.pixelscale * self.psf_border_int
    @property
    def has_psf(self):
        try:
            return self._psf is not None
        except AttributeError:
            return False

    def set_variance(self, variance):
        if variance is None:
            self._variance = None
            return
        assert variance.shape == self.data.shape, "variance must have same shape as data"
        self._variance = variance.to(dtype = AP_config.ap_dtype, device = AP_config.ap_device) if isinstance(variance, torch.Tensor) else torch.as_tensor(variance, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        
    def set_psf(self, psf):
        if psf is None:
            self._psf = None
            return
        assert torch.all((torch.tensor(psf.shape) % 2) == 1), "psf must have odd shape"
        self._psf = psf.to(dtype = AP_config.ap_dtype, device = AP_config.ap_device) if isinstance(psf, torch.Tensor) else torch.as_tensor(psf, dtype = AP_config.ap_dtype, device = AP_config.ap_device)

    def set_mask(self, mask):
        if mask is None:
            self._mask = None
            return
        assert mask.shape == self.data.shape, "mask must have same shape as data"
        self._mask = mask.to(dtype = torch.bool, device = AP_config.ap_device) if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, dtype = torch.bool, device = AP_config.ap_device)
        
    def to(self, dtype = None, device = None):
        super().to(dtype = dtype, device = device)
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        
        if self.has_variance:
            self._variance = self._variance.to(dtype = dtype, device = device)
        if self.has_psf:
            self._psf = self._psf.to(dtype = dtype, device = device)
        if self.has_mask:
            self._mask = self.mask.to(dtype = torch.bool, device = device)
        return self
            
    def or_mask(self, mask):
        self._mask = torch.logical_or(self.mask, mask)
    def and_mask(self, mask):
        self._mask = torch.logical_and(self.mask, mask)
        
    def copy(self):
        return self.__class__(
            data = torch.clone(self.data),
            zeropoint = self.zeropoint,
            mask = self._mask,
            psf = self._psf,
            psf_upscale = self.psf_upscale,
            variance = self._variance,
            note = self.note,
            window = self.window,
        )
    def blank_copy(self):
        return self.__class__(
            data = torch.zeros_like(self.data),
            zeropoint = self.zeropoint,
            mask = self._mask,
            psf = self._psf,
            psf_upscale = self.psf_upscale,
            note = self.note,
            window = self.window,
        )
    
    def get_window(self, window):
        indices = window.get_indices(self)
        return self.__class__(
            data = self.data[indices],
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            variance = self._variance[indices] if self.has_variance else None,
            mask = self._mask[indices] if self.has_mask else None,
            psf = self._psf,
            psf_upscale = self.psf_upscale,
            note = self.note,
            origin = (torch.max(self.origin[0], window.origin[0]),
                      torch.max(self.origin[1], window.origin[1]))
        )
    
    def reduce(self, scale):
        assert isinstance(scale, int) or scale.dtype is torch.int32

        if scale == 1:
            return self

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale
        if self.has_psf:
            PMS = self.psf.shape[0] // scale
            PNS = self.psf.shape[1] // scale

        return self.__class__(
            data = self.data[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)),
            pixelscale = self.pixelscale * scale,
            zeropoint = self.zeropoint,
            variance = self.variance[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)) if self.has_variance else None,
            mask = self.mask[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).max(axis=(1, 3)) if self.has_mask else None,
            psf = self.psf[:PMS*scale, :PNS*scale].reshape(PMS, scale, PNS, scale).sum(axis=(1, 3)) if self.has_psf else None,
            psf_upscale = self.psf_upscale,
            note = self.note,
            origin = self.origin,
        )

    def _save_image_list(self):
        image_list = super()._save_image_list()
        if self._psf is not None:
            psf_header = fits.Header()
            psf_header["IMAGE"] = "PSF"
            psf_header["UPSCALE"] = int(self.psf_upscale.detach().cpu().item())
            image_list.append(fits.ImageHDU(self._psf.detach().cpu().numpy(), header = psf_header))
        if self._variance is not None:
            var_header = fits.Header()
            var_header["IMAGE"] = "VARIANCE"
            image_list.append(fits.ImageHDU(self._variance.detach().cpu().numpy(), header = var_header))
        if self._mask is not None:
            mask_header = fits.Header()
            mask_header["IMAGE"] = "MASK"
            image_list.append(fits.ImageHDU(self._mask.detach().cpu().numpy().astype(int), header = mask_header))
        return image_list

    def load(self, filename):
        hdul = super().load(filename)

        for hdu in hdul:
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "PSF":
                self.set_psf(np.array(hdu.data, dtype = np.float64))
                self.psf_upscale = torch.tensor(hdu.header["UPSCALE"], dtype = torch.int32, device = AP_config.ap_device)
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "VARIANCE":
                self.set_variance(np.array(hdu.data, dtype = np.float64))
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "MASK":
                self.set_mask(np.array(hdu.data, dtype = bool))
        return hdul
    
class Target_Image_List(Image_List, Target_Image):
    
    @property
    def variance(self):
        return tuple(image.variance for image in self.image_list)
    @variance.setter
    def variance(self, variance):
        for image, var in zip(self.image_list, variance):
            image.set_variance(var)
    @property
    def has_variance(self):
        return any(image.has_variance for image in self.image_list)
    
    @property
    def mask(self):
        return tuple(image.mask for image in self.image_list)
    @mask.setter
    def mask(self, mask):
        for image, M in zip(self.image_list, mask):
            image.set_mask(M)
    @property
    def has_mask(self):
        return any(image.has_mask for image in self.image_list)
    
    @property
    def psf(self):
        return tuple(image.psf for image in self.image_list)
    @psf.setter
    def psf(self, psf):
        for image, P in zip(self.image_list, psf):
            image.set_psf(P)
    @property
    def has_psf(self):
        return any(image.has_psf for image in self.image_list)
    @property
    def psf_border(self):
        return tuple(image.psf_border for image in self.image_list)
    @property
    def psf_border_int(self):
        return tuple(image.psf_border_int for image in self.image_list)

    def set_variance(self, variance, img):
        self.image_list[img].set_variance(variance)
        
    def set_psf(self, psf, img):
        self.image_list[img].set_psf(psf)

    def set_mask(self, mask, img):
        self.image_list[img].set_mask(mask)

    def or_mask(self, mask):
        raise NotImplementedError()
    def and_mask(self, mask):
        raise NotImplementedError()
        
