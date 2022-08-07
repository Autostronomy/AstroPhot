import numpy as np
from model_object import BaseModel

class SuperModel(BaseModel):

    model_type = "supermodel"
    
    def __init__(self, name, target, locked, *models, **kwargs):
        self.models = models
        # figure out window
        super().__init__(name, target, window, locked, **kwargs)

    def build_parameters(self):
        for model in self.models:
            for P in model.parameters:
                
        
    def initialize(self, target = None):
        for model in self.models:
            model.initialize(target)

    def finalize(self):
        for model in self.models:
            model.finalize()

    def sample_model(self, sample_image = None):
        for model in self.models:
            model.sample_image(sample_image)

    def integrate_model(self):
        for model in self.models:
            model.integrate_model()

    def convolve_psf(self, psf = None):
        for model in self.models:
            model.convolve_psf(psf)

    def add_integrated_model(self):
        for model in self.models:
            model.add_integrated_model()

    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return
        
        for model in self.models:
            model.compute_loss(data)

        self.loss = defaultdict(0)
        for model in self.models:
            for key in model.loss:
                self.loss[key] += model.loss[key]

    def step_iteration(self):
        for model in self.models:
            model.step_iteration()

    def set_target(self, target):
        for model in self.models:
            model.set_target(target)

    def set_window(self, window = None, index_units = True):
        for model in self.models:
            model.set_window(window, index_units)

    def scale_window(self, scale):
        for model in self.models:
            model.scale_window(scale)

    def update_locked(self, locked):
        for model in self.models:
            model.update_locked(locked)

    def build_parameter_specs(self): # fixme
        pass
    def build_parameter_qualities(self): # fixme
        pass
    def build_parameters(self): # fixme
        pass
            
    def get_parameters(self, exclude_fixed = False, quality = None): # fixme
        for model in self.models:
            model.get_parameters(exclude_fixed, quality)

    def save_model(self, fileobject):
        for model in self.models:
            model.save_model(fileobject)

    def __getitem__(self, key): # fixme
        for model in self.models:
            model.__getitem__(key)

    from ._model_methods import _set_default_parameters
        
        
        
        
        
        
        
        
    
    
        
        
        
        
