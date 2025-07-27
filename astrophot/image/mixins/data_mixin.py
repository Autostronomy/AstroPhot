from typing import Union

import torch
import numpy as np
from astropy.io import fits

from ...utils.initialize import auto_variance
from ... import config
from ...errors import SpecificationConflict
from ..image_object import Image
from ..window import Window


class DataMixin:

    def __init__(
        self,
        *args,
        mask=None,
        std=None,
        variance=None,
        weight=None,
        _mask=None,
        _weight=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if _mask is None:
            self.mask = mask
        else:
            self._mask = _mask
        if (std is not None) + (variance is not None) + (weight is not None) > 1:
            raise SpecificationConflict(
                "Can only define one of: std, variance, or weight for a given image."
            )

        if _weight is not None:
            self._weight = _weight
        elif std is not None:
            self.std = std
        elif variance is not None:
            self.variance = variance
        else:
            self.weight = weight

        # Set nan pixels to be masked automatically
        if torch.any(torch.isnan(self.data)).item():
            self._mask = self.mask | torch.isnan(self.data)

    @property
    def std(self):
        """Stores the standard deviation of the image pixels. This represents
        the uncertainty in each pixel value. It should always have the
        same shape as the image data. In the case where the standard
        deviation is not known, a tensor of ones will be created to
        stand in as the standard deviation values.

        The standard deviation is not stored directly, instead it is
        computed as :math:`\\sqrt{1/W}` where :math:`W` is the
        weights.

        """
        if self.has_variance:
            return torch.sqrt(self.variance)
        return torch.ones_like(self.data)

    @std.setter
    def std(self, std):
        if std is None:
            self._weight = None
            return
        if isinstance(std, str) and std == "auto":
            self.weight = "auto"
            return
        self.weight = 1 / std**2

    @property
    def has_std(self):
        """Returns True when the image object has stored standard deviation values. If
        this is False and the std property is called then a
        tensor of ones will be returned.

        """
        try:
            return self._weight is not None
        except AttributeError:
            return False

    @property
    def variance(self):
        """Stores the variance of the image pixels. This represents the
        uncertainty in each pixel value. It should always have the
        same shape as the image data. In the case where the variance
        is not known, a tensor of ones will be created to stand in as
        the variance values.

        The variance is not stored directly, instead it is
        computed as :math:`\\frac{1}{W}` where :math:`W` is the
        weights.

        """
        if self.has_variance:
            return torch.where(self._weight == 0, torch.inf, 1 / self._weight)
        return torch.ones_like(self.data)

    @variance.setter
    def variance(self, variance):
        if variance is None:
            self._weight = None
            return
        if isinstance(variance, str) and variance == "auto":
            self.weight = "auto"
            return
        self.weight = 1 / variance

    @property
    def has_variance(self):
        """Returns True when the image object has stored variance values. If
        this is False and the variance property is called then a
        tensor of ones will be returned.

        """
        try:
            return self._weight is not None
        except AttributeError:
            return False

    @property
    def weight(self):
        """Stores the weight of the image pixels. This represents the
        uncertainty in each pixel value. It should always have the
        same shape as the image data. In the case where the weight
        is not known, a tensor of ones will be created to stand in as
        the weight values.

        The weights are used to proprtionately scale residuals in the
        likelihood. Most commonly this shows up as a :math:`\\chi^2`
        like:

        .. math::

          \\chi^2 = (\\vec{y} - \\vec{f(\\theta)})^TW(\\vec{y} - \\vec{f(\\theta)})

        which can be optimized to find parameter values. Using the
        Jacobian, which in this case is the derivative of every pixel
        wrt every parameter, the weight matrix also appears in the
        gradient:

        .. math::

          \\vec{g} = J^TW(\\vec{y} - \\vec{f(\\theta)})

        and the hessian approximation used in Levenberg-Marquardt:

        .. math::

          H \\approx J^TWJ

        """
        if self.has_weight:
            return self._weight
        return torch.ones_like(self.data)

    @weight.setter
    def weight(self, weight):
        if weight is None:
            self._weight = None
            return
        if isinstance(weight, str) and weight == "auto":
            weight = 1 / auto_variance(self.data, self.mask).T
        self._weight = torch.transpose(
            torch.as_tensor(weight, dtype=config.DTYPE, device=config.DEVICE), 0, 1
        )
        if self._weight.shape != self.data.shape:
            self._weight = None
            raise SpecificationConflict(
                f"weight/variance must have same shape as data ({weight.shape} vs {self.data.shape})"
            )

    @property
    def has_weight(self):
        """Returns True when the image object has stored weight values. If
        this is False and the weight property is called then a
        tensor of ones will be returned.

        """
        try:
            return self._weight is not None
        except AttributeError:
            self._weight = None
            return False

    @property
    def mask(self):
        """The mask stores a tensor of boolean values which indicate any
        pixels to be ignored. These pixels will be skipped in
        likelihood evaluations and in parameter optimization. It is
        common practice to mask pixels with pathological values such
        as due to cosmic rays or satellites passing through the image.

        In a mask, a True value indicates that the pixel is masked and
        should be ignored. False indicates a normal pixel which will
        inter into most calculations.

        If no mask is provided, all pixels are assumed valid.

        """
        if self.has_mask:
            return self._mask
        return torch.zeros_like(self.data, dtype=torch.bool)

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = None
            return
        self._mask = torch.transpose(
            torch.as_tensor(mask, dtype=torch.bool, device=config.DEVICE), 0, 1
        )
        if self._mask.shape != self.data.shape:
            self._mask = None
            raise SpecificationConflict(
                f"mask must have same shape as data ({mask.shape} vs {self.data.shape})"
            )

    @property
    def has_mask(self):
        """
        Single boolean to indicate if a mask has been provided by the user.
        """
        try:
            return self._mask is not None
        except AttributeError:
            return False

    def to(self, dtype=None, device=None):
        """Converts the stored `Target_Image` data, variance, psf, etc to a
        given data type and device.

        """
        if dtype is not None:
            dtype = config.DTYPE
        if device is not None:
            device = config.DEVICE
        super().to(dtype=dtype, device=device)

        if self.has_weight:
            self._weight = self._weight.to(dtype=dtype, device=device)
        if self.has_mask:
            self._mask = self._mask.to(dtype=torch.bool, device=device)
        return self

    def copy_kwargs(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        kwargs = {"_mask": self._mask, "_weight": self._weight, **kwargs}
        return super().copy_kwargs(**kwargs)

    def get_window(self, other: Union[Image, Window], indices=None, **kwargs):
        """Get a sub-region of the image as defined by an other image on the sky."""
        if indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        return super().get_window(
            other,
            _weight=self._weight[indices] if self.has_weight else None,
            _mask=self._mask[indices] if self.has_mask else None,
            indices=indices,
            **kwargs,
        )

    def fits_images(self):
        images = super().fits_images()
        if self.has_weight:
            images.append(
                fits.ImageHDU(
                    torch.transpose(self.weight, 0, 1).detach().cpu().numpy(), name="WEIGHT"
                )
            )
        if self.has_mask:
            images.append(
                fits.ImageHDU(
                    torch.transpose(self.mask, 0, 1).detach().cpu().numpy().astype(int), name="MASK"
                )
            )
        return images

    def load(self, filename: str, hduext=0):
        """Load the image from a FITS file. This will load the data, WCS, and
        any ancillary data such as variance, mask, and PSF.

        """
        hdulist = super().load(filename, hduext=hduext)
        if "WEIGHT" in hdulist:
            self.weight = np.array(hdulist["WEIGHT"].data, dtype=np.float64)
        if "MASK" in hdulist:
            self.mask = np.array(hdulist["MASK"].data, dtype=bool)
        elif "DQ" in hdulist:
            self.mask = np.array(hdulist["DQ"].data, dtype=bool)
        return hdulist

    def reduce(self, scale, **kwargs):
        """Returns a new `Target_Image` object with a reduced resolution
        compared to the current image. `scale` should be an integer
        indicating how much to reduce the resolution. If the
        `Target_Image` was originally (48,48) pixels across with a
        pixelscale of 1 and `reduce(2)` is called then the image will
        be (24,24) pixels and the pixelscale will be 2. If `reduce(3)`
        is called then the returned image will be (16,16) pixels
        across and the pixelscale will be 3.

        """
        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        return super().reduce(
            scale=scale,
            _weight=(
                1
                / self.variance[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .sum(axis=(1, 3))
                if self.has_variance
                else None
            ),
            _mask=(
                self.mask[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .amax(axis=(1, 3))
                if self.has_mask
                else None
            ),
            **kwargs,
        )
