from typing import List

import torch

from .image_object import Image, Image_List
from .. import AP_config
from ..errors import SpecificationConflict, InvalidImage

__all__ = ["Jacobian_Image", "Jacobian_Image_List"]


######################################################################
class Jacobian_Image(Image):
    """Jacobian of a model evaluated in an image.

    Image object which represents the evaluation of a jacobian on an
    image. It takes the form of a 3D (Image x Nparameters)
    tensor. This object can be added other other Jacobian images to
    build up a full jacobian for a complex model.

    """

    def __init__(
        self,
        parameters: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.parameters = list(parameters)
        if len(self.parameters) != len(set(self.parameters)):
            raise SpecificationConflict("Every parameter should be unique upon jacobian creation")

    def flatten(self, attribute: str = "data"):
        if attribute in self.children:
            return getattr(self, attribute).value.reshape((-1, len(self.parameters)))
        return getattr(self, attribute).reshape((-1, len(self.parameters)))

    def copy(self, **kwargs):
        return super().copy(parameters=self.parameters, **kwargs)

    def __iadd__(self, other: "Jacobian_Image"):
        if not isinstance(other, Jacobian_Image):
            raise InvalidImage("Jacobian images can only add with each other, not: type(other)")

        # exclude null jacobian images
        if other.data.value is None:
            return self
        if self.data.value is None:
            return other

        self_indices = self.get_indices(other)
        other_indices = other.get_indices(self)
        for i, other_identity in enumerate(other.parameters):
            if other_identity in self.parameters:
                other_loc = self.parameters.index(other_identity)
            else:
                data = torch.zeros(
                    self.data.shape[0],
                    self.data.shape[1],
                    self.data.shape[2] + 1,
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                data[:, :, :-1] = self.data.value
                self.data = data
                self.parameters.append(other_identity)
                other_loc = -1
            self.data.value[self_indices[0], self_indices[1], other_loc] += other.data.value[
                other_indices[0], other_indices[1], i
            ]
        return self


######################################################################
class Jacobian_Image_List(Image_List, Jacobian_Image):
    """For joint modelling, represents Jacobians evaluated on a list of
    images.

    Stores jacobians evaluated on a number of image objects. Since
    jacobian images are aware of the target images they were evaluated
    on, it is possible to combine this object with other
    Jacobian_Image_List objects or even Jacobian_Image objects and
    everything will be sorted into the proper locations of the list,
    and image.

    """

    def flatten(self, attribute="data"):
        if len(self.images) > 1:
            for image in self.images[1:]:
                if self.images[0].parameters != image.parameters:
                    raise SpecificationConflict(
                        "Jacobian image list sub-images track different parameters. Please initialize with all parameters that will be used."
                    )
        return torch.cat(tuple(image.flatten(attribute) for image in self.images))
