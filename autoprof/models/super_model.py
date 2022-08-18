import numpy as np
from model_object import BaseModel

class SuperModel(BaseModel):

    model_type = "supermodel"
    parameter_specs = {
        "center": {"units": "arcsec", "fixed": True, "uncertainty": 0.0}
    }
    parameter_qualities = {}
    global_loss = False
    global_psf = False
    
    def __init__(self, name, target, locked, models, **kwargs):
        self.models = models
        self.update_window()
        super().__init__(name, target, self.window, locked, **kwargs)

    def update_window(self, include_locked = False):
        new_window = None
        for model in self.models:
            if model.locked and not include_locked:
                continue
            if new_window is None:
                new_window = deepcopy(model.window)
            else:
                new_window |= model.window
        self.window = new_window

    def initialize(self, target = None):
        for model in self.models:
            model.initialize(target)

    def finalize(self):
        for model in self.models:
            model.finalize()
        # In case parameters were adjusted internally, sync up the pointer values
        for P in self.parameters:
            if isinstance(self.parameters[P], Pointing_Parameter):
                self.parameters[P].sync()

    def sample_model(self, sample_image = None):
        for model in self.models:
            model.sample_image(sample_image)
        if sample_image is None:
            sample_image = self.model_image
        super().sample_model(sample_image)
        if sample_image is self.model_image:
            for model in self.models:
                sample_image += model.model_image
                
    def convolve_psf(self, working_image, psf = None):
        if self.global_psf:
            super().convolve_psf(psf)
            return
        for model in self.models:
            model.convolve_psf(psf)

    def integrate_model(self, working_image, psf = None):
        if self.global_integrate:
            super().integrate_model()
            return
        for model in self.models:
            model.integrate_model()

    def add_integrated_model(self):
        for model in self.models:
            model.add_integrated_model()

    def compute_loss(self, data):
        if self.global_loss:
            super().compute_loss(data)
            return
        # If the image is locked, no need to compute the loss
        if self.locked:
            return
        
        for model in self.models:
            model.compute_loss(data)
            self.loss += model.loss

    def step_iteration(self):
        super().step_iteration()
        for model in self.models:
            model.step_iteration()
        # In case parameters were adjusted internally, sync up the pointer values
        for P in self.parameters:
            self.parameters[P].sync()

    def set_target(self, target):
        super().set_target(target)
        for model in self.models:
            model.set_target(target)

    def set_window(self, window = None, index_units = True):
        for model in self.models:
            model.set_window(window, index_units)
        self.update_window()
        super().set_window(self.window)
        
    def scale_window(self, scale):    
        for model in self.models:
            model.scale_window(scale)
        self.update_window()

    def update_locked(self, locked):
        super().update_locked(locked)
        for model in self.models:
            model.update_locked(locked)

    @classmethod
    def build_parameter_specs(cls, user_specs = None):
        parameter_specs = super().build_parameter_specs(user_specs)
        for model in self.models:
            for P in model.parameter_specs:
                parameter_specs[f"{model.name}|{P}"] = model.parameter_specs[P]
        return parameter_specs
                
    @classmethod
    def build_parameter_qualities(cls):
        parameter_qualities = super().build_parameter_qualities()
        for model in self.models:
            for P in model.parameter_qualities:
                parameter_qualities[f"{model.name}|{P}"] = model.parameter_qualities[P]
        return parameter_qualities
                
    def build_parameters(self):
        super().build_parameters()
        for model in self.models:
            for P in model.parameters:
                name = f"{model.name}|{P}"
                if isinstance(model.parameters[P], Parameter_Array):
                    self.parameters[name] = Pointing_Parameter_Array(name, model.parameters[P])
                else:
                    self.parameters[name] = Pointing_Parameter(name, model.parameters[P])
            
    def save_model(self, fileobject):
        for model in self.models:
            model.save_model(fileobject)

        
        
        
        
        
        
        
        
    
    
        
        
        
        
