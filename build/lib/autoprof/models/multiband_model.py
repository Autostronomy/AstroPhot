from .group_model_object import Group_Model
from ..image import Model_Image_List, Target_Image_List
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    individual models.

    """

    model_type = f"multiband {Group_Model.model_type}"

    def sync_target(self):
        for model, target in zip(self.model_list, self.target):
            model.target = target
        
    def make_model_image(self):
        return Model_Image_List(list(model.make_model_image() for model in self.model_list), dtype = self.dtype, device = self.device)

    def _build_target_List(self):
        return Target_Image_List(list(model.target for model in self.model_list), dtype = self.dtype, device = self.device)
    
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
        if tar is None:
            self._target = None
            return
        assert isinstance(tar, Target_Image_List)
        self._target = tar.to(dtype = self.dtype, device = self.device)

    @torch.no_grad()
    def initialize(self, targets = None):
        if targets is None:
            targets = self.target

        for model, target in zip(self.model_list, targets):
            model.initialize(target)

    def sample(self, sample_images = None):

        if sample_images is None:
            sample_window = True
            sample_images = self.make_model_image() 
        else:
            sample_window = False

        for model, sample_image in zip(self.model_list, sample_images):
            if sample_window:
                sample_image += model.sample()
            else:
                model.sample(sample_image)
        return sample_images

    def compute_loss(self, return_sample = False):

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
            return loss, Model_Image_List(samples, dtype = self.dtype, device = self.device)
        else:
            return loss

    def model_image_shapes(self):
        shapes = []
        for model in self.model_list:
            shapes.append(model.window.get_shape_flip(model.target.pixelscale).detach().cpu().numpy())
        return np.array(shapes, dtype = int)
    def model_image_sizes(self):
        sizes = []
        for shape in self.model_image_shapes():
            sizes.append(int(np.prod(shape)))
        return np.array(sizes, dtype = int)
        
    def jacobian(self, parameters = None, as_representation = False, override_locked = False, flatten = False):
        if parameters is not None:
            self.set_parameters(parameters, override_locked = override_locked, as_representation = as_representation)        
        sub_jacs = []
        for model in self.model_list:
            sub_jacs.append(model.jacobian(as_representation = as_representation, override_locked = override_locked, flatten = flatten))
        if flatten:
            img_sizes = self.model_image_sizes()
            full_jac = torch.zeros((sum(img_sizes),) + (np.sum(self.parameter_vector_len(override_locked = override_locked)),), dtype = self.dtype, device = self.device)
            param_map, param_vec_map = self.sub_model_parameter_map(override_locked = override_locked)
            for ijac, jac, p_map, vec_map in zip(range(len(sub_jacs)), sub_jacs, param_map, param_vec_map):
                for imodel, imulti in enumerate(vec_map):
                    full_jac[np.sum(img_sizes[:ijac]):np.sum(img_sizes[:ijac+1]),imulti] = jac[:,imodel]
            return full_jac
        return sub_jacs

    
