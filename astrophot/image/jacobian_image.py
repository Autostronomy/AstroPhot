from typing import List, Union

from .image_object import Image, ImageList
from ..errors import SpecificationConflict, InvalidImage
from ..backend_obj import backend

__all__ = ("JacobianImage", "JacobianImageList")


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

    def match_parameters(self, other: Union["JacobianImage", "JacobianImageList", List]):
        self_i = []
        other_i = []
        other_parameters = other if isinstance(other, list) else other.parameters
        for i, other_param in enumerate(other_parameters):
            if other_param in self.parameters:
                self_i.append(self.parameters.index(other_param))
                other_i.append(i)
        return self_i, other_i

    def __iadd__(self, other: "JacobianImage"):
        if not isinstance(other, JacobianImage):
            raise InvalidImage("Jacobian images can only add with each other, not: type(other)")

        self_indices = self.get_indices(other.window)
        other_indices = other.get_indices(self.window)
        for self_i, other_i in zip(*self.match_parameters(other)):
            self._data = backend.add_at_indices(
                self._data,
                self_indices + (self_i,),
                other.data[other_indices[0], other_indices[1], other_i],
            )
        return self

    def plane_to_world(self, x, y):
        raise NotImplementedError(
            "JacobianImage does not support plane_to_world conversion. There is no meaningful world position of a PSF image."
        )

    def world_to_plane(self, ra, dec):
        raise NotImplementedError(
            "JacobianImage does not support world_to_plane conversion. There is no meaningful world position of a PSF image."
        )


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

    @property
    def parameters(self) -> List[str]:
        """List of parameters for the jacobian images in this list."""
        if not self.images:
            return []
        return self.images[0].parameters

    def flatten(self, attribute: str = "data"):
        if len(self.images) > 1:
            for image in self.images[1:]:
                if self.images[0].parameters != image.parameters:
                    raise SpecificationConflict(
                        "Jacobian image list sub-images track different parameters. Please initialize with all parameters that will be used."
                    )
        return backend.concatenate(tuple(image.flatten(attribute) for image in self.images), dim=0)

    def match_parameters(self, other: Union[JacobianImage, "JacobianImageList", List[str]]):
        self_i = []
        other_i = []
        other_parameters = other if isinstance(other, list) else other.parameters
        for i, other_param in enumerate(other_parameters):
            if other_param in self.parameters:
                self_i.append(self.parameters.index(other_param))
                other_i.append(i)
        return self_i, other_i
