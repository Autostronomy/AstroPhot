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
from ..errors import SpecificationConflict, InvalidImage

__all__ = ["Target_Image", "Target_Image_List"]


class Target_Image(Image):
    """Image object which represents the data to be fit by a model. It can
    include a variance image, mask, and PSF as anciliary data which
    describes the target image.

    Target images are a basic unit of data in `AstroPhot`, they store
    the information collected from telescopes for which models are to
    be fit. There is minimial functionality in the `Target_Image`
    object itself, it is mostly defined in terms of how other objects
    interact with it.

    Basic usage:

    .. code-block:: python

      import astrophot as ap

      # Create target image
      image = ap.image.Target_Image(
          data = <pixel data>,
          wcs = <astropy WCS object>,
          variance = <pixel uncertainties>,
          psf = <point spread function as PSF_Image object>,
          mask = <pixels to ignore>,
      )

      # Display the data
      fig, ax = plt.subplots()
      ap.plots.target_image(fig, ax, image)
      plt.show()

      # Save the image
      image.save("mytarget.fits")

      # Load the image
      image2 = ap.image.Target_Image(filename = "mytarget.fits")

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.has_weight and "weight" in kwargs:
            self.set_weight(kwargs.get("weight", None))
        elif not self.has_variance and "variance" in kwargs:
            self.set_variance(kwargs.get("variance", None))
        if not self.has_mask:
            self.set_mask(kwargs.get("mask", None))
        if not self.has_psf:
            self.set_psf(kwargs.get("psf", None), kwargs.get("psf_upscale", 1))

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
        return torch.ones_like(self.data)

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
        self.set_variance(variance)

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
        self.set_weight(weight)

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
        as due to cosmic rays or satelites passing through the image.

        In a mask, a True value indicates that the pixel is masked and
        should be ignored. False indicates a normal pixel which will
        inter into most calculaitons.

        If no mask is provided, all pixels are assumed valid.

        """
        if self.has_mask:
            return self._mask
        return torch.zeros_like(self.data, dtype=torch.bool)

    @mask.setter
    def mask(self, mask):
        self.set_mask(mask)

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
    def psf(self):
        """Stores the point-spread-function for this target. This should be a
        `PSF_Image` object which represents the scattering of a point
        source of light. It can also be an `AstroPhot_Model` object
        which will contribute its own parameters to an optimization
        problem.

        The PSF stored for a `Target_Image` object is passed to all
        models applied to that target which have a `psf_mode` that is
        not `none`. This means they will all use the same PSF
        model. If one wishes to define a variable PSF across an image,
        then they should pass the PSF objects to the `AstroPhot_Model`'s
        directly instead of to a `Target_Image`.

        Raises:

          AttributeError: if this is called without a PSF defined

        """
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
        """
        Provide a variance tensor for the image. Variance is equal to $\\sigma^2$. This should have the same shape as the data.
        """
        if variance is None:
            self._weight = None
            return
        self.set_weight(1 / variance)

    def set_weight(self, weight):
        """Provide a weight tensor for the image. Weight is equal to $\\frac{1}{\\sigma^2}$. This should have the same
        shape as the data.

        """
        if weight is None:
            self._weight = None
            return
        if weight.shape != self.data.shape:
            raise SpecificationConflict(
                f"weight/variance must have same shape as data ({weight.shape} vs {self.data.shape})"
            )
        self._weight = (
            weight.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            if isinstance(weight, torch.Tensor)
            else torch.as_tensor(weight, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        )

    def set_psf(self, psf, psf_upscale=1):
        """Provide a psf for the `Target_Image`. This is stored and passed to
        models which need to be convolved.

        The PSF doesn't need to have the same pixelscale as the
        image. It should be some multiple of the resolution of the
        `Target_Image` though. So if the image has a pixelscale of 1,
        the psf may have a pixelscale of 1, 1/2, 1/3, 1/4 and so on.

        """
        if psf is None:
            self._psf = None
            return
        if isinstance(psf, PSF_Image):
            self._psf = psf
            return

        self._psf = PSF_Image(
            data=psf,
            psf_upscale=psf_upscale,
            pixelscale=self.pixelscale / psf_upscale,
            identity=self.identity,
        )

    def set_mask(self, mask):
        """
        Set the boolean mask which will indicate which pixels to ignore. A mask value of True means the pixel will be ignored.
        """
        if mask is None:
            self._mask = None
            return
        if mask.shape != self.data.shape:
            raise SpecificationConflict(
                f"mask must have same shape as data ({mask.shape} vs {self.data.shape})"
            )
        self._mask = (
            mask.to(dtype=torch.bool, device=AP_config.ap_device)
            if isinstance(mask, torch.Tensor)
            else torch.as_tensor(mask, dtype=torch.bool, device=AP_config.ap_device)
        )

    def to(self, dtype=None, device=None):
        """Converts the stored `Target_Image` data, variance, psf, etc to a
        given data type and device.

        """
        super().to(dtype=dtype, device=device)
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device

        if self.has_weight:
            self._weight = self._weight.to(dtype=dtype, device=device)
        if self.has_psf:
            self._psf = self._psf.to(dtype=dtype, device=device)
        if self.has_mask:
            self._mask = self.mask.to(dtype=torch.bool, device=device)
        return self

    def or_mask(self, mask):
        """
        Combines the currently stored mask with a provided new mask using the boolean `or` operator.
        """
        self._mask = torch.logical_or(self.mask, mask)

    def and_mask(self, mask):
        """
        Combines the currently stored mask with a provided new mask using the boolean `and` operator.
        """
        self._mask = torch.logical_and(self.mask, mask)

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return super().copy(
            mask=self._mask,
            psf=self._psf,
            weight=self._weight,
            **kwargs,
        )

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is not filled with zeros.

        """
        return super().blank_copy(mask=self._mask, psf=self._psf, **kwargs)

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""
        indices = self.window.get_self_indices(window)
        return super().get_window(
            window=window,
            weight=self._weight[indices] if self.has_weight else None,
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
        """
        Construct a blank `Jacobian_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
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
        """
        Construct a blank `Model_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
        return Model_Image(
            data=torch.zeros_like(self.data) if data is None else data,
            header=self.header,
            target_identity=self.identity,
            **kwargs,
        )

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
        """
        `Target_Image` doesn't have expand yet.
        """
        raise NotImplementedError("expand not available for Target_Image yet")

    def get_state(self):
        state = super().get_state()

        if self.has_weight:
            state["weight"] = self.weight.detach().cpu().tolist()
        if self.has_mask:
            state["mask"] = self.mask.detach().cpu().tolist()
        if self.has_psf:
            state["psf"] = self.psf.get_state()

        return state

    def set_state(self, state):
        super().set_state(state)

        self.weight = state.get("weight", None)
        self.mask = state.get("mask", None)
        if "psf" in state:
            self.psf = PSF_Image(state=state["psf"])

    def get_fits_state(self):
        states = super().get_fits_state()
        if self.has_weight:
            states.append(
                {
                    "DATA": self.weight.detach().cpu().numpy(),
                    "HEADER": {"IMAGE": "WEIGHT"},
                }
            )
        if self.has_mask:
            states.append(
                {
                    "DATA": self.mask.detach().cpu().numpy(),
                    "HEADER": {"IMAGE": "MASK"},
                }
            )
        if self.has_psf:
            states += self.psf.get_fits_state()

        return states

    def set_fits_state(self, states):
        super().set_fits_state(states)
        for state in states:
            if state["HEADER"]["IMAGE"] == "WEIGHT":
                self.weight = np.array(state["DATA"], dtype=np.float64)
            if state["HEADER"]["IMAGE"] == "mask":
                self.mask = np.array(state["DATA"], dtype=bool)
            if state["HEADER"]["IMAGE"] == "PSF":
                self.psf = PSF_Image(fits_state=states)


class Target_Image_List(Image_List, Target_Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, Target_Image) for image in self.image_list):
            raise InvalidImage(
                f"Target_Image_List can only hold Target_Image objects, not {tuple(type(image) for image in self.image_list)}"
            )

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
    def weight(self):
        return tuple(image.weight for image in self.image_list)

    @weight.setter
    def weight(self, weight):
        for image, wgt in zip(self.image_list, weight):
            image.set_weight(wgt)

    @property
    def has_weight(self):
        return any(image.has_weight for image in self.image_list)

    def jacobian_image(self, parameters: List[str], data: Optional[List[torch.Tensor]] = None):
        if data is None:
            data = [None] * len(self.image_list)
        return Jacobian_Image_List(
            list(image.jacobian_image(parameters, dat) for image, dat in zip(self.image_list, data))
        )

    def model_image(self, data: Optional[List[torch.Tensor]] = None):
        if data is None:
            data = [None] * len(self.image_list)
        return Model_Image_List(
            list(image.model_image(data=dat) for image, dat in zip(self.image_list, data))
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
