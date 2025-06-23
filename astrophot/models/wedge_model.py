import numpy as np
import torch

from .galaxy_model_object import GalaxyModel

__all__ = ["WedgeGalaxy"]


class WedgeGalaxy(GalaxyModel):
    """Variant of the ray model where no smooth transition is performed
    between regions as a function of theta, instead there is a sharp
    trnasition boundary. This may be desirable as it cleanly
    separates where the pixel information is going. Due to the sharp
    transition though, it may cause unusual behaviour when fitting. If
    problems occur, try fitting a ray model first then fix the center,
    PA, and q and then fit the wedge model. Essentially this breaks
    down the structure fitting and the light profile fitting into two
    steps. The wedge model, like the ray model, defines no extra
    parameters, however a new option can be supplied on instantiation
    of the wedge model which is "wedges" or the number of wedges in
    the model.

    """

    _model_type = "segments"
    usable = False
    _options = ("segmentss", "symmetric_wedges")

    def __init__(self, *args, symmetric_wedges=True, segments=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric_wedges = symmetric_wedges
        self.segments = segments

    def polar_model(self, R, T):
        model = torch.zeros_like(R)
        if self.segments % 2 == 0 and self.symmetric_wedges:
            for w in range(self.segments):
                angles = (T - (w * np.pi / self.segments)) % np.pi
                indices = torch.logical_or(
                    angles < (np.pi / (2 * self.segments)),
                    angles >= (np.pi * (1 - 1 / (2 * self.segments))),
                )
                model[indices] += self.iradial_model(w, R[indices])
        elif self.segments % 2 == 1 and self.symmetric_wedges:
            for w in range(self.segments):
                angles = (T - (w * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / (2 * self.segments)),
                    angles >= (np.pi * (2 - 1 / (2 * self.segments))),
                )
                model[indices] += self.iradial_model(w, R[indices])
                angles = (T - (np.pi + w * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / (2 * self.segments)),
                    angles >= (np.pi * (2 - 1 / (2 * self.segments))),
                )
                model[indices] += self.iradial_model(w, R[indices])
        else:
            for w in range(self.segments):
                angles = (T - (w * 2 * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / self.segments),
                    angles >= (np.pi * (2 - 1 / self.segments)),
                )
                model[indices] += self.iradial_model(w, R[indices])
        return model

    def brightness(self, x, y):
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))
