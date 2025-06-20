from caskade import Param as CParam
import torch


class Param(CParam):
    """
    A class that extends the Caskade Param class to include additional functionality.
    This class is used to define parameters for models in the AstroPhot package.
    """

    def __init__(self, *args, uncertainty=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty = uncertainty
        self.saveattrs.add("uncertainty")

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, uncertainty):
        if uncertainty is None:
            self._uncertainty = None
        else:
            self._uncertainty = torch.as_tensor(uncertainty)
