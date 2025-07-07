from target_image import TargetImage
from distort_image import DistortImageMixin
from . import func


class SIPTargetImage(DistortImageMixin, TargetImage):

    def __init__(self, *args, sipA=(), sipB=(), sipAP=(), sipBP=(), pixel_area_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sipA = sipA
        self.sipB = sipB
        self.sipAP = sipAP
        self.sipBP = sipBP

        i, j = self.pixel_center_meshgrid()
        u, v = i - self.crpix[0], j - self.crpix[1]
        self.distortion_ij = func.sip_delta(u, v, self.sipA, self.sipB)
        self.distortion_IJ = func.sip_delta(u, v, self.sipAP, self.sipBP)  # fixme maybe

        if pixel_area_map is None:
            self.update_pixel_area_map()
        else:
            self._pixel_area_map = pixel_area_map

    @property
    def pixel_area_map(self):
        return self._pixel_area_map

    def update_pixel_area_map(self):
        """
        Update the pixel area map based on the current SIP coefficients.
        """
        i, j = self.pixel_corner_meshgrid()
        x, y = self.pixel_to_plane(i, j)

        # 1: [:-1, :-1]
        # 2: [:-1, 1:]
        # 3: [1:, 1:]
        # 4: [1:, :-1]
        A = 0.5 * (
            x[:-1, :-1] * y[:-1, 1:]
            + x[:-1, 1:] * y[1:, 1:]
            + x[1:, 1:] * y[1:, :-1]
            + x[1:, :-1] * y[:-1, :-1]
            - (
                x[:-1, 1:] * y[:-1, :-1]
                + x[1:, 1:] * y[:-1, 1:]
                + x[1:, :-1] * y[1:, 1:]
                + x[:-1, :-1] * y[1:, :-1]
            )
        )
        self._pixel_area_map = A.abs()
