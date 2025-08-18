from typing import List, Optional

import numpy as np
from astropy.io import fits

from .image_object import Image, ImageList
from .jacobian_image import JacobianImage, JacobianImageList
from .model_image import ModelImage, ModelImageList
from .psf_image import PSFImage
from .. import config
from ..backend_obj import backend, ArrayLike
from ..errors import InvalidImage
from .mixins import DataMixin
from ..utils.decorators import combine_docstrings

__all__ = ["TargetImage", "TargetImageList"]


@combine_docstrings
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

    ```{python}
    import astrophot as ap

    # Create target image
    image = ap.image.Target_Image(
        data="pixel data",
        wcs="astropy WCS object",
        variance="pixel uncertainties",
        psf="point spread function as PSF_Image object",
        mask="True for pixels to ignore",
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
    ```

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
    def has_psf(self) -> bool:
        """Returns True when the target image object has a PSF model."""
        try:
            return self._psf is not None
        except AttributeError:
            return False

    @property
    def psf(self):
        """The PSF for the `TargetImage`. This is used to convolve the
        model with the PSF before evaluating the likelihood. The PSF
        should be a `PSFImage` object or an `AstroPhot` PSFModel.

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
        """Provide a psf for the `TargetImage`. This is stored and passed to
        models which need to be convolved.

        The PSF doesn't need to have the same pixelscale as the
        image. It should be some multiple of the resolution of the
        `TargetImage` though. So if the image has a pixelscale of 1,
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
                CD=self.CD,
                name=self.name + "_psf",
            )

    def copy_kwargs(self, **kwargs):
        kwargs = {"psf": self.psf, **kwargs}
        return super().copy_kwargs(**kwargs)

    def fits_images(self):
        images = super().fits_images()
        if self.has_psf:
            if isinstance(self.psf, PSFImage):
                images.append(
                    fits.ImageHDU(
                        backend.to_numpy(backend.transpose(self.psf.data, 1, 0)),
                        name="PSF",
                        header=fits.Header(self.psf.fits_info()),
                    )
                )
            else:
                config.logger.warning("Unable to save PSF to FITS, not a PSF_Image.")
        return images

    def load(self, filename: str, hduext: int = 0):
        """Load the image from a FITS file. This will load the data, WCS, and
        any ancillary data such as variance, mask, and PSF.

        """
        hdulist = super().load(filename, hduext=hduext)
        if "PSF" in hdulist:
            self.psf = PSFImage(
                data=np.array(hdulist["PSF"].data, dtype=np.float64),
                CD=(
                    (hdulist["PSF"].header["CD1_1"], hdulist["PSF"].header["CD1_2"]),
                    (hdulist["PSF"].header["CD2_1"], hdulist["PSF"].header["CD2_2"]),
                ),
            )
        return hdulist

    def jacobian_image(
        self,
        parameters: List[str],
        data: Optional[ArrayLike] = None,
        **kwargs,
    ) -> JacobianImage:
        """
        Construct a blank `JacobianImage` object formatted like this current `TargetImage` object. Mostly used internally.
        """
        if data is None:
            data = backend.zeros(
                (*self.data.shape, len(parameters)),
                dtype=config.DTYPE,
                device=config.DEVICE,
            )
        kwargs = {
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_jacobian",
            **kwargs,
        }
        return JacobianImage(parameters=parameters, _data=data, **kwargs)

    def model_image(self, upsample: int = 1, pad: int = 0, **kwargs) -> ModelImage:
        """
        Construct a blank `ModelImage` object formatted like this current `TargetImage` object. Mostly used internally.
        """
        kwargs = {
            "_data": backend.zeros(
                (self.data.shape[0] * upsample + 2 * pad, self.data.shape[1] * upsample + 2 * pad),
                dtype=config.DTYPE,
                device=config.DEVICE,
            ),
            "CD": self.CD.value / upsample,
            "crpix": (self.crpix + 0.5) * upsample + pad - 0.5,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_model",
            **kwargs,
        }
        return ModelImage(**kwargs)

    def psf_image(self, data: ArrayLike, upscale: int = 1, **kwargs) -> PSFImage:
        kwargs = {
            "data": data,
            "CD": self.CD.value / upscale,
            "identity": self.identity,
            "name": self.name + "_psf",
            **kwargs,
        }
        return PSFImage(**kwargs)

    def reduce(self, scale: int, **kwargs) -> "TargetImage":
        """Returns a new `TargetImage` object with a reduced resolution
        compared to the current image. `scale` should be an integer
        indicating how much to reduce the resolution. If the
        `TargetImage` was originally (48,48) pixels across with a
        pixelscale of 1 and `reduce(2)` is called then the image will
        be (24,24) pixels and the pixelscale will be 2. If `reduce(3)`
        is called then the returned image will be (16,16) pixels
        across and the pixelscale will be 3.

        """

        return super().reduce(
            scale=scale,
            psf=(self.psf.reduce(scale) if isinstance(self.psf, PSFImage) else None),
            **kwargs,
        )


class TargetImageList(ImageList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, (TargetImage, TargetImageList)) for image in self.images):
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

    def jacobian_image(
        self, parameters: List[str], data: Optional[List[ArrayLike]] = None
    ) -> JacobianImageList:
        if data is None:
            data = tuple(None for _ in range(len(self.images)))
        return JacobianImageList(
            list(image.jacobian_image(parameters, dat) for image, dat in zip(self.images, data))
        )

    def model_image(self) -> ModelImageList:
        return ModelImageList(list(image.model_image() for image in self.images))

    @property
    def mask(self):
        return tuple(image.mask for image in self.images)

    @mask.setter
    def mask(self, mask):
        for image, M in zip(self.images, mask):
            image.mask = M

    @property
    def has_mask(self) -> bool:
        return any(image.has_mask for image in self.images)

    @property
    def psf(self):
        return tuple(image.psf for image in self.images)

    @psf.setter
    def psf(self, psf):
        for image, P in zip(self.images, psf):
            image.psf = P

    @property
    def has_psf(self) -> bool:
        return any(image.has_psf for image in self.images)
