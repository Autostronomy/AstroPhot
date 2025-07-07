from ..param import forward
from . import func
from ..utils.interpolate import interp2d


class DistortImageMixin:
    """
    DistortImage is a subclass of Image that applies a distortion to the image.
    This is typically used for images that have been distorted by a telescope or camera.
    """

    @forward
    def pixel_to_plane(self, i, j, crtan):
        di = interp2d(self.distortion_ij[0], i, j)
        dj = interp2d(self.distortion_ij[1], i, j)
        return func.pixel_to_plane_linear(i + di, j + dj, *self.crpix, self.pixelscale, *crtan)

    @forward
    def plane_to_pixel(self, x, y, crtan):
        I, J = func.plane_to_pixel_linear(x, y, *self.crpix, self.pixelscale, *crtan)
        dI = interp2d(self.distortion_IJ[0], I, J)
        dJ = interp2d(self.distortion_IJ[1], I, J)
        return I + dI, J + dJ
