from .model_object import BaseModel

__all__ = ["Star_Model"]

class Star_Model(BaseModel):
    """Prototype star model, to be subclassed by other star models which
    define specific behavior.

    """
    model_type = f"star {BaseModel.model_type}"
    psf_mode = 'none'
    parameter_specs = {
        "flux": {"units": "flux"},
    }

    
