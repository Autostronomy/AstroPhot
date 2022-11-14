from .model_object import BaseModel

__all__ = ["Star_Model"]

class Star_Model(BaseModel):
    """Prototype star model, to be subclassed by other star models which
    define specific behavior.

    """
    model_type = f"star {BaseModel.model_type}"
    psf_mode = "none"
    integrate_mode = "none"
    
    def radius_metric(self, X, Y):
        return torch.sqrt((torch.abs(X)+1e-6)**2 + (torch.abs(Y)+1e-6)**2) # epsilon added for numerical stability of gradient
    
