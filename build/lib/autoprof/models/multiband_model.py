from .group_model_object import Group_Model
from ..image import Model_Image_List, Target_Image_List, Window_List
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
from .. import AP_config

__all__ = ["Multiband_Model"]

class Multiband_Model(Group_Model):
    """A multi-band model functions similarly to a group model object,
    except that instead of a single target for all models in the model
    list, there is a unique target for each model. The target for a
    multiband model is a Target_Image_List object which stores a
    pointer to each target from the models in the multiband model
    model list. Similarly, the model image that is output from
    sampling the multiband model is a Model_Image_List object which
    stores a pointer to the model image output from each of the
    individual models. At a high level, the multiband model operates
    like a group model object and in most cases shouldn't be
    operationally different.

    """

    model_type = f"multiband"

    def sync_target(self):
        """Ensure that the target list object held by the multiband model
        matches the targets of the individual models that it holds.

        """
        
        for model, target in zip(self.model_list, self.target):
            model.target = target
        
    def make_model_image(self):
        """Makes a blank Model_Image_List object, typically for the purposes
        of sampling this model image will be filled by the individual
        sub models.

        """
        return Model_Image_List(list(model.make_model_image() for model in self.model_list))

    def _build_target_List(self):
        """Construct a Target_Image_List object from the targets already held
        by the individual sub models.

        """
        return Target_Image_List(list(model.target for model in self.model_list))
    
    @property
    def window(self):
        return Window_List(list(model.window for model in self.model_list))
    @window.setter
    def window(self, win):
        pass
    
    @property 
    def target(self):
        try:
            if self._target is None:
                self._target = self._build_target_List()
        except AttributeError:
            self._target = self._build_target_List()
        return self._target
    @target.setter
    def target(self, tar):
        self._target = tar
        assert tar is None or isinstance(tar, Target_Image_List), f"multiband model object needs Target_Image_List object, not {type(tar)}"

    @torch.no_grad()
    def initialize(self, targets = None):
        """Initialize the models, ensure that all parameters have valid
        values and any other ancilliary requirements are met before
        sampling/fitting the sub models.

        """
        if targets is None:
            targets = self.target

        for model, target in zip(self.model_list, targets):
            model.initialize(target)

    def sample(self, sample_image = None):
        """Fill the sample image object (a Model_Image_List) with samples
        from the various sub models. This is the method by which the
        model creates an image.

        """
        if sample_image is None:
            sample_window = True
            sample_image = self.make_model_image() 
        else:
            sample_window = False

        for model, sub_image in zip(self.model_list, sample_image):
            if sample_window:
                sub_image += model.sample()
            else:
                model.sample(sub_image)
        return sample_image

    def compute_loss(self, return_sample = False):
        """Compute the Chi^2 for the multiband model.

        """
        loss = 0
        samples = []
        for model in self.model_list:
            res = model.compute_loss(return_sample)
            if return_sample:
                loss += res[0]
                samples.append(res[1])
            else:
                loss += res
        if return_sample:
            return loss, Model_Image_List(samples)
        else:
            return loss

    def model_image_shapes(self):
        """Return the shape of all the sub model image windows.

        """
        shapes = []
        for model in self.model_list:
            shapes.append(model.window.get_shape_flip(model.target.pixelscale).detach().cpu().numpy())
        return np.array(shapes, dtype = int)
    def model_image_sizes(self):
        """Return the size of all the sub model image windows.

        """
        sizes = []
        for shape in self.model_image_shapes():
            sizes.append(int(np.prod(shape)))
        return np.array(sizes, dtype = int)
        
    def jacobian(self, parameters = None, as_representation = False, override_locked = False, flatten = False):
        """Compute the jacobian for the full multiband model object. Unless
        flatten is True, the jacobian will have the same shape as the
        individual images for the multiple bands plus an extra
        dimension which holds the values for each parameter.

        """
        if parameters is not None:
            self.set_parameters(parameters, override_locked = override_locked, as_representation = as_representation)        
        sub_jacs = []
        for model in self.model_list:
            sub_jacs.append(model.jacobian(as_representation = as_representation, override_locked = override_locked, flatten = flatten))
        if flatten:
            img_sizes = self.model_image_sizes()
            full_jac = torch.zeros((sum(img_sizes),) + (np.sum(self.parameter_vector_len(override_locked = override_locked)),), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            param_map, param_vec_map = self.sub_model_parameter_map(override_locked = override_locked)
            for ijac, jac, p_map, vec_map in zip(range(len(sub_jacs)), sub_jacs, param_map, param_vec_map):
                for imodel, imulti in enumerate(vec_map):
                    full_jac[np.sum(img_sizes[:ijac]):np.sum(img_sizes[:ijac+1]),imulti] = jac[:,imodel]
            return full_jac
        return sub_jacs

    
