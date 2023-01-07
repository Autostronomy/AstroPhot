from .model_object import BaseModel

__all__ = ["Star_Model"]

class Star_Model(BaseModel):
    """Prototype star model, to be subclassed by other star models which
    define specific behavior. Mostly this just requires that no
    standard PSF convolution is applied to this model as that is to be
    handled internally by the star model. By default, the PSF object
    has no integration mode since it will always be evaluated at the
    resolution of the PSF provided, this the integration has in effect
    already been done when constructing hte PSF.

    """
    model_type = f"star {BaseModel.model_type}"
    integrate_mode = "none"
    
    def radius_metric(self, X, Y):
        return torch.sqrt((torch.abs(X)+1e-8)**2 + (torch.abs(Y)+1e-8)**2) # epsilon added for numerical stability of gradient

    @property
    def psf_mode(self):
        return "none"
    @psf_mode.setter
    def psf_mode(self, val):
        pass
