import numpy as np
import torch

from .. import AP_config
from .image_object import Image, ImageList
from ..errors import InvalidImage

__all__ = ["ModelImage", "ModelImageList"]


######################################################################
class ModelImage(Image):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """

    def __init__(self, *args, window=None, upsample=1, pad=0, **kwargs):
        if window is not None:
            kwargs["pixelscale"] = window.image.pixelscale / upsample
            kwargs["crpix"] = (
                (window.crpix.npvalue - np.array((window.i_low, window.j_low)) + 0.5) * upsample
                + pad
                - 0.5
            )
            kwargs["crval"] = window.image.crval.value
            kwargs["crtan"] = window.image.crtan.value
            kwargs["data"] = torch.zeros(
                (
                    (window.i_high - window.i_low) * upsample + 2 * pad,
                    (window.j_high - window.j_low) * upsample + 2 * pad,
                ),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            kwargs["zeropoint"] = window.image.zeropoint
            kwargs["identity"] = window.image.identity
            kwargs["name"] = window.image.name + "_model"
        super().__init__(*args, **kwargs)

    def clear_image(self):
        self.data._value = torch.zeros_like(self.data.value)

    def crop(self, pixels, **kwargs):
        """Crop the image by the number of pixels given. This will crop
        the image in all four directions by the number of pixels given.

        given data shape (N, M) the new shape will be:

        crop - int: crop the same number of pixels on all sides. new shape (N - 2*crop, M - 2*crop)
        crop - (int, int): crop each dimension by the number of pixels given. new shape (N - 2*crop[1], M - 2*crop[0])
        crop - (int, int, int, int): crop each side by the number of pixels given assuming (x low, x high, y low, y high). new shape (N - crop[2] - crop[3], M - crop[0] - crop[1])
        """
        if len(pixels) == 1:  # same crop in all dimension
            crop = pixels if isinstance(pixels, int) else pixels[0]
            data = self.data.value[
                crop : self.data.shape[0] - crop,
                crop : self.data.shape[1] - crop,
            ]
            crpix = self.crpix.value - crop
        elif len(pixels) == 2:  # different crop in each dimension
            data = self.data.value[
                pixels[1] : self.data.shape[0] - pixels[1],
                pixels[0] : self.data.shape[1] - pixels[0],
            ]
            crpix = self.crpix.value - pixels
        elif len(pixels) == 4:  # different crop on all sides
            data = self.data.value[
                pixels[2] : self.data.shape[0] - pixels[3],
                pixels[0] : self.data.shape[1] - pixels[1],
            ]
            crpix = self.crpix.value - pixels[0::2]  # fixme
        else:
            raise ValueError(
                f"Invalid crop shape {pixels}, must be (int,), (int, int), or (int, int, int, int)!"
            )
        return self.copy(data=data, crpix=crpix, **kwargs)

    def reduce(self, scale: int, **kwargs):
        """This operation will downsample an image by the factor given. If
        scale = 2 then 2x2 blocks of pixels will be summed together to
        form individual larger pixels. A new image object will be
        returned with the appropriate pixelscale and data tensor. Note
        that the window does not change in this operation since the
        pixels are condensed, but the pixel size is increased
        correspondingly.

        Parameters:
            scale: factor by which to condense the image pixels. Each scale X scale region will be summed [int]

        """
        if not isinstance(scale, int) and not (
            isinstance(scale, torch.Tensor) and scale.dtype is torch.int32
        ):
            raise SpecificationConflict(f"Reduce scale must be an integer! not {type(scale)}")
        if scale == 1:
            return self

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        data = (
            self.data.value[: MS * scale, : NS * scale]
            .reshape(MS, scale, NS, scale)
            .sum(axis=(1, 3))
        )
        pixelscale = self.pixelscale * scale
        crpix = (self.crpix.value + 0.5) / scale - 0.5
        return self.copy(
            data=data,
            pixelscale=pixelscale,
            crpix=crpix,
            **kwargs,
        )


######################################################################
class ModelImageList(ImageList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, ModelImage) for image in self.images):
            raise InvalidImage(
                f"Model_Image_List can only hold Model_Image objects, not {tuple(type(image) for image in self.images)}"
            )

    def clear_image(self):
        for image in self.images:
            image.clear_image()
