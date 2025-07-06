from typing import List, Optional, Union

import numpy as np
import torch
from astropy.io import fits

from .image_object import Image, ImageList
from .window import Window
from .jacobian_image import JacobianImage, JacobianImageList
from .model_image import ModelImage, ModelImageList
from .psf_image import PSFImage
from .. import AP_config
from ..utils.initialize import auto_variance
from ..errors import SpecificationConflict, InvalidImage

__all__ = ["TargetImage", "TargetImageList"]


class TargetImage(Image):
    """Image object which represents the data to be fit by a model. It can
    include a variance image, mask, and PSF as anciliary data which
    describes the target image.

    Target images are a basic unit of data in `AstroPhot`, they store
    the information collected from telescopes for which models are to
    be fit. There is minimal functionality in the `Target_Image`
    object itself, it is mostly defined in terms of how other objects
    interact with it.

    Basic usage:

    .. code-block:: python

      import astrophot as ap

      # Create target image
      image = ap.image.Target_Image(
          data="pixel data",
          wcs="astropy WCS object",
          variance="pixel uncertainties",
          psf="point spread function as PSF_Image object",
          mask=" True for pixels to ignore",
      )

      # Display the data
      fig, ax = plt.subplots()
      ap.plots.target_image(fig, ax, image)
      plt.show()

      # Save the image
      image.save("mytarget.fits")

      # Load the image
      image2 = ap.image.Target_Image(filename="mytarget.fits")

      # Make low resolution version
      lowrez = image.reduce(2)

    Some important information to keep in mind. First, providing an
    `astropy WCS` object is the best way to keep track of coordinates
    and pixel scale properties, especially when dealing with
    multi-band data. If images have relative positioning, rotation,
    pixel sizes, field of view this will all be handled automatically
    by taking advantage of `WCS` objects. Second, Providing accurate
    variance (or weight) maps is critical to getting a good fit to the
    data. This is a very common source of issues so it is worthwhile
    to review literature on how best to construct such a map. A good
    starting place is the FAQ for GALFIT:
    https://users.obs.carnegiescience.edu/peng/work/galfit/CHI2.html
    which is an excellent resource for all things image modeling. Just
    note that `AstroPhot` uses variance or weight maps, not sigma
    images. `AstroPhot` will not crete a variance map for the user, by
    default it will just assume uniform variance which is rarely
    accurate. Third, The PSF pixelscale must be a multiple of the
    image pixelscale. So if the image has a pixelscale of 1 then the
    PSF must have a pixelscale of 1, 1/2, 1/3, etc for anything to
    work out. Note that if the PSF pixelscale is finer than the image,
    then all modelling will be done at the higher resolution. This is
    recommended for accuracy though it can mean higher memory
    consumption.

    """

    image_count = 0

    def __init__(self, *args, mask=None, variance=None, psf=None, weight=None, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.has_mask:
            self.mask = mask
        if not self.has_weight and variance is None:
            self.weight = weight
        elif not self.has_variance:
            self.variance = variance
        if not self.has_psf:
            self.psf = psf

        # Set nan pixels to be masked automatically
        if torch.any(torch.isnan(self.data.value)).item():
            self.mask = self.mask | torch.isnan(self.data.value)

    @property
    def standard_deviation(self):
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
        return torch.ones_like(self.data.value)

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
        return torch.ones_like(self.data.value)

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
        return torch.ones_like(self.data.value)

    @weight.setter
    def weight(self, weight):
        if weight is None:
            self._weight = None
            return
        if isinstance(weight, str) and weight == "auto":
            weight = 1 / auto_variance(self.data.value, self.mask)
        if weight.shape != self.data.shape:
            raise SpecificationConflict(
                f"weight/variance must have same shape as data ({weight.shape} vs {self.data.shape})"
            )
        self._weight = torch.as_tensor(weight, dtype=AP_config.ap_dtype, device=AP_config.ap_device)

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
        return torch.zeros_like(self.data.value, dtype=torch.bool)

    @mask.setter
    def mask(self, mask):
        if mask is None:
            self._mask = None
            return
        if mask.shape != self.data.shape:
            raise SpecificationConflict(
                f"mask must have same shape as data ({mask.shape} vs {self.data.shape})"
            )
        self._mask = torch.as_tensor(mask, dtype=torch.bool, device=AP_config.ap_device)

    @property
    def has_mask(self):
        """
        Single boolean to indicate if a mask has been provided by the user.
        """
        try:
            return self._mask is not None
        except AttributeError:
            return False

    @property
    def has_psf(self):
        """Returns True when the target image object has a PSF model."""
        try:
            return self._psf is not None
        except AttributeError:
            return False

    @property
    def psf(self):
        """The PSF for the `Target_Image`. This is used to convolve the
        model with the PSF before evaluating the likelihood. The PSF
        should be a `PSF_Image` object or an `AstroPhot` PSF_Model.

        If no PSF is provided, then the image will not be convolved
        with a PSF and the model will be evaluated directly on the
        image pixels.

        """
        try:
            return self._psf
        except AttributeError:
            return None

    @psf.setter
    def psf(self, psf):
        """Provide a psf for the `Target_Image`. This is stored and passed to
        models which need to be convolved.

        The PSF doesn't need to have the same pixelscale as the
        image. It should be some multiple of the resolution of the
        `Target_Image` though. So if the image has a pixelscale of 1,
        the psf may have a pixelscale of 1, 1/2, 1/3, 1/4 and so on.

        """
        if hasattr(self, "_psf"):
            del self._psf  # remove old psf if it exists
        from ..models import Model

        if psf is None:
            self._psf = None
        elif isinstance(psf, PSFImage):
            self._psf = psf
        elif isinstance(psf, Model):
            self._psf = psf
        else:
            self._psf = PSFImage(
                data=psf,
                pixelscale=self.pixelscale,
                name=self.name + "_psf",
            )

    def to(self, dtype=None, device=None):
        """Converts the stored `Target_Image` data, variance, psf, etc to a
        given data type and device.

        """
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)

        if self.has_weight:
            self._weight = self._weight.to(dtype=dtype, device=device)
        if self.has_mask:
            self._mask = self.mask.to(dtype=torch.bool, device=device)
        return self

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        kwargs = {"mask": self._mask, "psf": self.psf, "weight": self._weight, **kwargs}
        return super().copy(**kwargs)

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        kwargs = {"mask": self._mask, "psf": self.psf, "weight": self._weight, **kwargs}
        return super().blank_copy(**kwargs)

    def get_window(self, other: Union[Image, Window], **kwargs):
        """Get a sub-region of the image as defined by an other image on the sky."""
        indices = self.get_indices(other if isinstance(other, Window) else other.window)
        return super().get_window(
            other,
            weight=self._weight[indices] if self.has_weight else None,
            mask=self._mask[indices] if self.has_mask else None,
            psf=self.psf,
            _indices=indices,
            **kwargs,
        )

    def fits_images(self):
        images = super().fits_images()
        if self.has_variance:
            images.append(fits.ImageHDU(self.weight.cpu().numpy(), name="WEIGHT"))
        if self.has_mask:
            images.append(fits.ImageHDU(self.mask.cpu().numpy(), name="MASK"))
        if self.has_psf:
            if isinstance(self.psf, PSFImage):
                images.append(
                    fits.ImageHDU(
                        self.psf.data.npvalue, name="PSF", header=fits.Header(self.psf.fits_info())
                    )
                )
            else:
                AP_config.ap_logger.warning("Unable to save PSF to FITS, not a PSF_Image.")
        return images

    def load(self, filename: str):
        """Load the image from a FITS file. This will load the data, WCS, and
        any ancillary data such as variance, mask, and PSF.

        """
        hdulist = super().load(filename)
        if "WEIGHT" in hdulist:
            self.weight = np.array(hdulist["WEIGHT"].data, dtype=np.float64)
        if "MASK" in hdulist:
            self.mask = np.array(hdulist["MASK"].data, dtype=bool)
        if "PSF" in hdulist:
            self.psf = PSFImage(
                data=np.array(hdulist["PSF"].data, dtype=np.float64),
                pixelscale=(
                    (hdulist["PSF"].header["CD1_1"], hdulist["PSF"].header["CD1_2"]),
                    (hdulist["PSF"].header["CD2_1"], hdulist["PSF"].header["CD2_2"]),
                ),
            )

    def jacobian_image(
        self,
        parameters: List[str],
        data: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Construct a blank `Jacobian_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
        if data is None:
            data = torch.zeros(
                (*self.data.shape, len(parameters)),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        kwargs = {
            "pixelscale": self.pixelscale,
            "crpix": self.crpix.value,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_jacobian",
            **kwargs,
        }
        return JacobianImage(parameters=parameters, data=data, **kwargs)

    def model_image(self, **kwargs):
        """
        Construct a blank `Model_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
        kwargs = {
            "data": torch.zeros_like(self.data.value),
            "pixelscale": self.pixelscale,
            "crpix": self.crpix.value,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_model",
            **kwargs,
        }
        return ModelImage(**kwargs)

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
            variance=(
                self.variance[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .sum(axis=(1, 3))
                if self.has_variance
                else None
            ),
            mask=(
                self.mask[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .amax(axis=(1, 3))
                if self.has_mask
                else None
            ),
            psf=self.psf if self.has_psf else None,
            **kwargs,
        )


class TargetImageList(ImageList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, TargetImage) for image in self.images):
            raise InvalidImage(
                f"Target_Image_List can only hold Target_Image objects, not {tuple(type(image) for image in self.images)}"
            )

    @property
    def variance(self):
        return tuple(image.variance for image in self.images)

    @variance.setter
    def variance(self, variance):
        for image, var in zip(self.images, variance):
            image.variance = var

    @property
    def has_variance(self):
        return any(image.has_variance for image in self.images)

    @property
    def weight(self):
        return tuple(image.weight for image in self.images)

    @weight.setter
    def weight(self, weight):
        for image, wgt in zip(self.images, weight):
            image.weight = wgt

    @property
    def has_weight(self):
        return any(image.has_weight for image in self.images)

    def jacobian_image(self, parameters: List[str], data: Optional[List[torch.Tensor]] = None):
        if data is None:
            data = tuple(None for _ in range(len(self.images)))
        return JacobianImageList(
            list(image.jacobian_image(parameters, dat) for image, dat in zip(self.images, data))
        )

    def model_image(self):
        return ModelImageList(list(image.model_image() for image in self.images))

    def match_indices(self, other):
        indices = []
        if isinstance(other, TargetImageList):
            for other_image in other.images:
                for isi, self_image in enumerate(self.images):
                    if other_image.identity == self_image.identity:
                        indices.append(isi)
                        break
                else:
                    indices.append(None)
        elif isinstance(other, TargetImage):
            for isi, self_image in enumerate(self.images):
                if other.identity == self_image.identity:
                    indices = isi
                    break
            else:
                indices = None
        return indices

    def __isub__(self, other):
        if isinstance(other, ImageList):
            for other_image in other.images:
                for self_image in self.images:
                    if other_image.identity == self_image.identity:
                        self_image -= other_image
                        break
        elif isinstance(other, Image):
            for self_image in self.images:
                if other.identity == self_image.identity:
                    self_image -= other
                    break
        else:
            for self_image, other_image in zip(self.images, other):
                self_image -= other_image
        return self

    def __iadd__(self, other):
        if isinstance(other, ImageList):
            for other_image in other.images:
                for self_image in self.images:
                    if other_image.identity == self_image.identity:
                        self_image += other_image
                        break
        elif isinstance(other, Image):
            for self_image in self.images:
                if other.identity == self_image.identity:
                    self_image += other
        else:
            for self_image, other_image in zip(self.images, other):
                self_image += other_image
        return self

    @property
    def mask(self):
        return tuple(image.mask for image in self.images)

    @mask.setter
    def mask(self, mask):
        for image, M in zip(self.images, mask):
            image.mask = M

    @property
    def has_mask(self):
        return any(image.has_mask for image in self.images)

    @property
    def psf(self):
        return tuple(image.psf for image in self.images)

    @psf.setter
    def psf(self, psf):
        for image, P in zip(self.images, psf):
            image.psf = P

    @property
    def has_psf(self):
        return any(image.has_psf for image in self.images)

    @property
    def psf_border(self):
        return tuple(image.psf_border for image in self.images)

    @property
    def psf_border_int(self):
        return tuple(image.psf_border_int for image in self.images)
