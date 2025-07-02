from caskade import Param as CParam
import torch


class Param(CParam):
    """
    A class that extends the Caskade Param class to include additional functionality.
    This class is used to define parameters for models in the AstroPhot package.
    """

    def __init__(self, *args, uncertainty=None, prof=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty = uncertainty
        self.saveattrs.add("uncertainty")
        self.prof = prof
        self.saveattrs.add("prof")

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, uncertainty):
        if uncertainty is None:
            self._uncertainty = None
        else:
            self._uncertainty = torch.as_tensor(uncertainty)

    @property
    def prof(self):
        return self._prof

    @prof.setter
    def prof(self, prof):
        if prof is None:
            self._prof = None
        else:
            self._prof = torch.as_tensor(prof)

    @property
    def initialized(self):
        """Check if the parameter is initialized."""
        if self.pointer:
            return True
        if self.value is not None:
            return True
        return False
