from math import prod
import numpy as np

from caskade import Param as CParam
from ..backend_obj import backend
from .. import config


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
    def name_array(self):
        numel = max(1, prod(self.shape))
        if numel == 1:
            return np.array(self.name)
        names = [f"{self.name}_{i}" for i in range(numel)]
        return np.array(names).reshape(self.shape)

    @property
    def initialized(self):
        """Check if the parameter is initialized."""
        if self.pointer:
            return True
        if self.value is not None:
            return True
        return False

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


class PSFParam(CParam):
    """
    A class that extends the Param class to include additional functionality specific to PSF parameters.
    This class is used to define PSF parameters for models in the AstroPhot package.
    """

    def __init__(self, *args, upsample=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample = upsample
        self.saveattrs.add("upsample")

    @Param.value.getter
    def value(self):
        value = super().value
        if value is None:
            value = backend.ones((1, 1), dtype=config.DTYPE, device=config.DEVICE)
        return value

    @value.setter
    def value(self, value):
        from ..image import PSFImage
        from ..models import Model

        if isinstance(value, PSFImage):

            def getimage(p):
                return p.image._data

            self.unlink(tuple(self.children))
            self.link("image", value)
            value = getimage
        elif isinstance(value, Model):

            def getmodel(p):
                return p.model()._data

            self.unlink(tuple(self.children))
            self.link("model", value)
            value = getmodel
        Param.value.fset(self, value)

    @property
    def upsample(self) -> int:
        if len(self.children) > 0:
            return next(iter(self.children.values)).upsample
        return self._upsample

    @upsample.setter
    def upsample(self, value: int):
        if value < 1:
            raise ValueError("upsample factor must be a positive integer.")
        self._upsample = int(value)

    @property
    def pixelscale(self) -> float:
        return 1.0 / self.upsample

    @property
    def psf_pad(self) -> int:
        return max(self.value.shape[-2:]) // 2
