from typing import Optional

from .group_model_object import Group_Model
from ..image import PSF_Image
from ..image import PSF_Image, Image, Window, Model_Image, Model_Image_List, Window_List
from ..errors import InvalidTarget
from ..param import Parameter_Node

__all__ = ["PSF_Group_Model"]


class PSF_Group_Model(Group_Model):

    model_type = f"psf {Group_Model.model_type}"
    usable = True
    normalize_psf = True

    @property
    def psf_mode(self):
        return "none"

    @psf_mode.setter
    def psf_mode(self, value):
        pass

    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar):
        if not (tar is None or isinstance(tar, PSF_Image)):
            raise InvalidTarget("Group_Model target must be a PSF_Image instance.")
        self._target = tar

        if hasattr(self, "models"):
            for model in self.models.values():
                model.target = tar

    def sample(
        self,
        image: Optional[Image] = None,
        window: Optional[Window] = None,
        parameters: Optional[Parameter_Node] = None,
    ):
        # Note: same as group model except working_image is normalized at the end
        self._param_tuple = None
        if image is None:
            sample_window = True
            image = self.make_model_image(window=window)
        else:
            sample_window = False
        if window is None:
            window = image.window

        working_image = image[window].blank_copy()

        if parameters is None:
            parameters = self.parameters

        for model in self.models.values():
            if window is not None and isinstance(window, Window_List):
                indices = self.target.match_indices(model.target)
                if isinstance(indices, (tuple, list)):
                    use_window = Window_List(
                        window_list=list(window.window_list[ind] for ind in indices)
                    )
                else:
                    use_window = window.window_list[indices]
            else:
                use_window = window
            if sample_window:
                # Will sample the model fit window then add to the image
                working_image += model(window=use_window, parameters=parameters[model.name])
            else:
                # Will sample the entire image
                model(working_image, window=use_window, parameters=parameters[model.name])

        if self.normalize_psf:
            working_image.data /= working_image.data.sum()
        image += working_image
        return image
