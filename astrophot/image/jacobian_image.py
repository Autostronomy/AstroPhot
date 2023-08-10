import warnings
from typing import Optional, Union, List

import torch
from torch.nn.functional import pad

from .image_object import Image, Image_List
from .. import AP_config

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
        target_identity: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_identity = target_identity
        self.parameters = parameters
        assert len(self.parameters) == len(
            set(self.parameters)
        ), "Every parameter should be unique upon jacobian creation"

    def flatten(self, attribute: str = "data"):
        return getattr(self, attribute).reshape((-1, len(self.parameters)))

    def copy(self, **kwargs):
        return super().copy(
            parameters=self.parameters, target_identity=self.target_identity, **kwargs
        )

    def __add__(self, other):
        raise NotImplementedError("Jacobian images cannot add like this, use +=")

    def __sub__(self, other):
        raise NotImplementedError("Jacobian images cannot subtract")

    def __isub__(self, other):
        raise NotImplementedError("Jacobian images cannot subtract")

    def __iadd__(self, other):
        assert isinstance(
            other, Jacobian_Image
        ), "Jacobian images can only add with each other"

        # exclude null jacobian images
        if other.data is None:
            return self
        if self.data is None:
            return other

        full_window = self.window | other.window
        if full_window > self.window:
            warnings.warn("Jacobian image addition without full coverage")

        self_indices = other.window.get_indices(self)
        other_indices = self.window.get_indices(other)
        for i, other_identity in enumerate(other.parameters):
            if other_identity in self.parameters:
                other_loc = self.parameters.index(other_identity)
            else:
                self.set_data(
                    torch.cat(
                        (
                            self.data,
                            torch.zeros(
                                self.data.shape[0],
                                self.data.shape[1],
                                1,
                                dtype=AP_config.ap_dtype,
                                device=AP_config.ap_device,
                            ),
                        ),
                        dim=2,
                    ),
                    require_shape=False,
                )
                self.parameters.append(other_identity)
                other_loc = -1
            self.data[self_indices[0], self_indices[1], other_loc] += other.data[
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

    def __init__(self, image_list):
        super().__init__(image_list)

    def flatten(self, attribute="data"):
        if len(self.image_list) > 1:
            for image in self.image_list[1:]:
                assert (
                    self.image_list[0].parameters == image.parameters
                ), "Jacobian image list sub-images track different parameters. Please initialize with all parameters that will be used"
        return torch.cat(tuple(image.flatten(attribute) for image in self.image_list))

    def __add__(self, other):
        raise NotImplementedError("Jacobian images cannot add like this, use +=")

    def __sub__(self, other):
        raise NotImplementedError("Jacobian images cannot subtract")

    def __isub__(self, other):
        raise NotImplementedError("Jacobian images cannot subtract")

    def __iadd__(self, other):
        if isinstance(other, Jacobian_Image_List):
            for other_image in other.image_list:
                for self_image in self.image_list:
                    if other_image.target_identity == self_image.target_identity:
                        self_image += other_image
                        break
                else:
                    self.image_list.append(other_image)
        elif isinstance(other, Jacobian_Image):
            for self_image in self.image_list:
                if other.target_identity == self_image.target_identity:
                    self_image += other
                    break
            else:
                self.image_list.append(other_image)
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image += other_image
        return self
