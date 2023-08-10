from typing import List, Optional

import torch
import numpy as np
from torch.nn.functional import avg_pool2d

from .image_object import Image, Image_List
from .jacobian_image import Jacobian_Image, Jacobian_Image_List
from .model_image import Model_Image, Model_Image_List
from .psf_image import PSF_Image
from astropy.io import fits
from .. import AP_config

__all__ = ["Target_Image", "Target_Image_List"]


class Target_Image(Image):
    """Image object which represents the data to be fit by a model. It can
    include a variance image, mask, and PSF as anciliary data which
    describes the target image.

    """

    image_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.has_variance:
            self.set_variance(kwargs.get("variance", None))
        if not self.has_mask:
            self.set_mask(kwargs.get("mask", None))
        if not self.has_psf:
            self.set_psf(kwargs.get("psf", None), kwargs.get("psf_upscale", 1))

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
        return torch.zeros_like(self.data, dtype=torch.bool)

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
    def has_psf(self):
        try:
            return self._psf is not None
        except AttributeError:
            return False

    def set_variance(self, variance):
        if variance is None:
            self._variance = None
            return
        assert (
            variance.shape == self.data.shape
        ), "variance must have same shape as data"
        self._variance = (
            variance.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            if isinstance(variance, torch.Tensor)
            else torch.as_tensor(
                variance, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        )

    def set_psf(self, psf, psf_upscale=1):
        if psf is None:
            self._psf = None
            return
        if isinstance(psf, PSF_Image):
            self._psf = psf
            return

        self._psf = PSF_Image(
            psf,
            psf_upscale=psf_upscale,
            pixelscale=self.pixelscale / psf_upscale,
            identity=self.identity,
        )

    def set_mask(self, mask):
        if mask is None:
            self._mask = None
            return
        assert mask.shape == self.data.shape, "mask must have same shape as data"
        self._mask = (
            mask.to(dtype=torch.bool, device=AP_config.ap_device)
            if isinstance(mask, torch.Tensor)
            else torch.as_tensor(mask, dtype=torch.bool, device=AP_config.ap_device)
        )

    def to(self, dtype=None, device=None):
        super().to(dtype=dtype, device=device)
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device

        if self.has_variance:
            self._variance = self._variance.to(dtype=dtype, device=device)
        if self.has_psf:
            self._psf = self._psf.to(dtype=dtype, device=device)
        if self.has_mask:
            self._mask = self.mask.to(dtype=torch.bool, device=device)
        return self

    def or_mask(self, mask):
        self._mask = torch.logical_or(self.mask, mask)

    def and_mask(self, mask):
        self._mask = torch.logical_and(self.mask, mask)

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return super().copy(
            mask=self._mask,
            psf=self._psf,
            variance=self._variance,
            **kwargs,
        )

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is not filled with zeros.

        """
        return super().blank_copy(mask=self._mask, psf=self._psf, **kwargs)

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""
        indices = window.get_indices(self)
        return super().get_window(
            window=window,
            variance=self._variance[indices] if self.has_variance else None,
            mask=self._mask[indices] if self.has_mask else None,
            psf=self._psf,
            **kwargs,
        )

    def jacobian_image(
        self,
        parameters: Optional[List[str]] = None,
        data: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if parameters is None:
            data = None
            parameters = []
        elif data is None:
            data = torch.zeros(
                (*self.data.shape, len(parameters)),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        return Jacobian_Image(
            parameters=parameters,
            target_identity=self.identity,
            data=data,
            header=self.header,
            **kwargs,
        )

    def model_image(self, data: Optional[torch.Tensor] = None, **kwargs):
        return Model_Image(
            data=torch.zeros_like(self.data) if data is None else data,
            header=self.header,
            target_identity=self.identity,
            **kwargs,
        )

    def reduce(self, scale, **kwargs):
        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        return super().reduce(
            scale=scale,
            variance=self.variance[: MS * scale, : NS * scale]
            .reshape(MS, scale, NS, scale)
            .sum(axis=(1, 3))
            if self.has_variance
            else None,
            mask=self.mask[: MS * scale, : NS * scale]
            .reshape(MS, scale, NS, scale)
            .amax(axis=(1, 3))
            if self.has_mask
            else None,
            psf=self.psf.reduce(scale) if self.has_psf else None,
            **kwargs,
        )

    def expand(self, padding):
        raise NotImplementedError("expand not available for Target_Image yet")

    def _save_image_list(self):
        image_list = super()._save_image_list()
        if self._psf is not None:
            psf_header = fits.Header()
            self.psf._save_image_list(image_list, psf_header)
        if self._variance is not None:
            var_header = fits.Header()
            var_header["IMAGE"] = "VARIANCE"
            image_list.append(
                fits.ImageHDU(self._variance.detach().cpu().numpy(), header=var_header)
            )
        if self._mask is not None:
            mask_header = fits.Header()
            mask_header["IMAGE"] = "MASK"
            image_list.append(
                fits.ImageHDU(
                    self._mask.detach().cpu().numpy().astype(int), header=mask_header
                )
            )
        return image_list

    def load(self, filename):
        hdul = super().load(filename)

        for hdu in hdul:
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "PSF":
                self.set_psf(
                    PSF_Image(
                        np.array(hdu.data, dtype=np.float64),
                        psf_upscale=hdu.header["UPSCALE"],
                        pixelscale=self.pixelscale / hdu.header["UPSCALE"],
                    )
                )
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "VARIANCE":
                self.set_variance(np.array(hdu.data, dtype=np.float64))
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "MASK":
                self.set_mask(np.array(hdu.data, dtype=bool))
        return hdul


class Target_Image_List(Image_List, Target_Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert all(
            isinstance(image, Target_Image) for image in self.image_list
        ), f"Target_Image_List can only hold Target_Image objects, not {tuple(type(image) for image in self.image_list)}"

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

    def jacobian_image(
        self, parameters: List[str], data: Optional[List[torch.Tensor]] = None
    ):
        if data is None:
            data = [None] * len(self.image_list)
        return Jacobian_Image_List(
            list(
                image.jacobian_image(parameters, dat)
                for image, dat in zip(self.image_list, data)
            )
        )

    def model_image(self, data: Optional[List[torch.Tensor]] = None):
        if data is None:
            data = [None] * len(self.image_list)
        return Model_Image_List(
            list(
                image.model_image(data=dat) for image, dat in zip(self.image_list, data)
            )
        )

    def match_indices(self, other):
        indices = []
        if isinstance(other, Target_Image_List):
            for other_image in other.image_list:
                for isi, self_image in enumerate(self.image_list):
                    if other_image.identity == self_image.identity:
                        indices.append(isi)
                        break
                else:
                    indices.append(None)
        elif isinstance(other, Target_Image):
            for isi, self_image in enumerate(self.image_list):
                if other.identity == self_image.identity:
                    indices = isi
                    break
            else:
                indices = None
        return indices

    def __isub__(self, other):
        if isinstance(other, Target_Image_List):
            for other_image in other.image_list:
                for self_image in self.image_list:
                    if other_image.identity == self_image.identity:
                        self_image -= other_image
                        break
                else:
                    self.image_list.append(other_image)
        elif isinstance(other, Target_Image):
            for self_image in self.image_list:
                if other.identity == self_image.identity:
                    self_image -= other
                    break
        elif isinstance(other, Model_Image_List):
            for other_image in other.image_list:
                for self_image in self.image_list:
                    if other_image.target_identity == self_image.identity:
                        self_image -= other_image
                        break
        elif isinstance(other, Model_Image):
            for self_image in self.image_list:
                if other.target_identity == self_image.identity:
                    self_image -= other
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image -= other_image
        return self

    def __iadd__(self, other):
        if isinstance(other, Target_Image_List):
            for other_image in other.image_list:
                for self_image in self.image_list:
                    if other_image.identity == self_image.identity:
                        self_image += other_image
                        break
                else:
                    self.image_list.append(other_image)
        elif isinstance(other, Target_Image):
            for self_image in self.image_list:
                if other.identity == self_image.identity:
                    self_image += other
        elif isinstance(other, Model_Image_List):
            for other_image in other.image_list:
                for self_image in self.image_list:
                    if other_image.target_identity == self_image.identity:
                        self_image += other_image
                        break
        elif isinstance(other, Model_Image):
            for self_image in self.image_list:
                if other.target_identity == self_image.identity:
                    self_image += other
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image += other_image
        return self

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
