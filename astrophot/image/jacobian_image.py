from typing import List

import torch

from .image_object import Image, ImageList
from ..errors import SpecificationConflict, InvalidImage

__all__ = ["JacobianImage", "JacobianImageList"]


######################################################################
class JacobianImage(Image):
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

    def copy(self, **kwargs):
        return super().copy(parameters=self.parameters, **kwargs)

    def __iadd__(self, other: "JacobianImage"):
        if not isinstance(other, JacobianImage):
            raise InvalidImage("Jacobian images can only add with each other, not: type(other)")

        self_indices = self.get_indices(other.window)
        other_indices = other.get_indices(self.window)
        for i, other_identity in enumerate(other.parameters):
            if other_identity in self.parameters:
                other_loc = self.parameters.index(other_identity)
            else:
                continue
            self._data[self_indices[0], self_indices[1], other_loc] += other.data[
                other_indices[0], other_indices[1], i
            ]
        return self


######################################################################
class JacobianImageList(ImageList):
    """For joint modelling, represents Jacobians evaluated on a list of
    images.

    Stores jacobians evaluated on a number of image objects. Since
    jacobian images are aware of the target images they were evaluated
    on, it is possible to combine this object with other
    Jacobian_Image_List objects or even Jacobian_Image objects and
    everything will be sorted into the proper locations of the list,
    and image.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, (JacobianImage, JacobianImageList)) for image in self.images):
            raise InvalidImage(
                f"JacobianImageList can only hold JacobianImage objects, not {tuple(type(image) for image in self.images)}"
            )

    def flatten(self, attribute="data"):
        if len(self.images) > 1:
            for image in self.images[1:]:
                if self.images[0].parameters != image.parameters:
                    raise SpecificationConflict(
                        "Jacobian image list sub-images track different parameters. Please initialize with all parameters that will be used."
                    )
        return torch.cat(tuple(image.flatten(attribute) for image in self.images), dim=0)
