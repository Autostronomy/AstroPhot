import torch

from .core_model import AutoPhot_Model

__all__ = ("Auxiliary_Model",)

class Auxiliary_Model(AutoPhot_Model):
    """A class for models which do not themselves produce images, instead
    they control behaviour of the fitting problem. Very little
    functionality is included in this base class as Auxiliary models
    are each, by design, very unique.

    The Auxiliary_Model does define the parameter system for these
    models, which all end in "*" to indicate that they are auxiliary
    models.

    """

    model_type = f"aux {AutoPhot_Model.model_type}"

    useable = False

    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    parameter_specs = {}
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ()

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        
        # Set any user defined attributes for the model
        for kwarg in kwargs:
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        # If loading from a file, get model configuration then exit __init__
        if "filename" in kwargs:
            self.load(kwargs["filename"])
            return

        self.parameter_specs = self.build_parameter_specs(
            kwargs.get("parameters", None)
        )
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.parameters.set_values(kwargs["parameters"])
    
        for P in self.parameters:
            assert P.name[-1] == "*", "auxiliary model parameter names must have '*' at the end"
