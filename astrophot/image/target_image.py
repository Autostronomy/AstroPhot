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
from ..errors import InvalidImage
from .mixins import DataMixin

__all__ = ["TargetImage", "TargetImageList"]


class TargetImage(DataMixin, Image):
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

    def __init__(self, *args, psf=None, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.has_psf:
            self.psf = psf

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

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        kwargs = {"psf": self.psf, **kwargs}
        return super().copy(**kwargs)

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        kwargs = {"psf": self.psf, **kwargs}
        return super().blank_copy(**kwargs)

    def get_window(self, other: Union[Image, Window], indices=None, **kwargs):
        """Get a sub-region of the image as defined by an other image on the sky."""
        return super().get_window(
            other,
            psf=self.psf,
            indices=indices,
            **kwargs,
        )

    def fits_images(self):
        images = super().fits_images()
        if self.has_psf:
            if isinstance(self.psf, PSFImage):
                images.append(
                    fits.ImageHDU(
                        self.psf.data.detach().cpu().numpy(),
                        name="PSF",
                        header=fits.Header(self.psf.fits_info()),
                    )
                )
            else:
                AP_config.ap_logger.warning("Unable to save PSF to FITS, not a PSF_Image.")
        return images

    def load(self, filename: str, hduext=0):
        """Load the image from a FITS file. This will load the data, WCS, and
        any ancillary data such as variance, mask, and PSF.

        """
        hdulist = super().load(filename, hduext=hduext)
        if "PSF" in hdulist:
            self.psf = PSFImage(
                data=np.array(hdulist["PSF"].data, dtype=np.float64),
                pixelscale=(
                    (hdulist["PSF"].header["CD1_1"], hdulist["PSF"].header["CD1_2"]),
                    (hdulist["PSF"].header["CD2_1"], hdulist["PSF"].header["CD2_2"]),
                ),
            )
        return hdulist

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
            "pixelscale": self.pixelscale.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_jacobian",
            **kwargs,
        }
        return JacobianImage(parameters=parameters, data=data, **kwargs)

    def model_image(self, upsample=1, pad=0, **kwargs):
        """
        Construct a blank `Model_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
        kwargs = {
            "data": torch.zeros(
                (self.data.shape[0] * upsample + 2 * pad, self.data.shape[1] * upsample + 2 * pad),
                dtype=self.data.dtype,
                device=self.data.device,
            ),
            "pixelscale": self.pixelscale.value / upsample,
            "crpix": (self.crpix + 0.5) * upsample + pad - 0.5,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_model",
            **kwargs,
        }
        return ModelImage(**kwargs)


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
