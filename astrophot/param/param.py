from caskade import Param as CParam
from ..backend_obj import backend


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
            self._uncertainty = backend.as_array(uncertainty)

    @property
    def prof(self):
        return self._prof

    @prof.setter
    def prof(self, prof):
        if prof is None:
            self._prof = None
        else:
            self._prof = backend.as_array(prof)

    @property
    def initialized(self):
        """Check if the parameter is initialized."""
        if self.pointer:
            return True
        if self.value is not None:
            return True
        return False

    def is_valid(self, value):
        if self.valid[0] is not None and backend.any(value <= self.valid[0]):
            return False
        if self.valid[1] is not None and backend.any(value >= self.valid[1]):
            return False
        return True

    def soft_valid(self, value):
        if self.valid[0] is None and self.valid[1] is None:
            return value
        if self.valid[0] is not None and self.valid[1] is not None:
            vrange = 0.1 * (self.valid[1] - self.valid[0])
            smin = self.valid[0] + 0.1 * vrange
            smax = self.valid[1] - 0.1 * vrange
        elif self.valid[0] is not None:
            smin = self.valid[0] + 0.1
            smax = None
        elif self.valid[1] is not None:
            smin = None
            smax = self.valid[1] - 0.1
        return backend.clamp(value, min=smin, max=smax)
