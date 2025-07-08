import torch

from .target_image import TargetImage
from .mixins import SIPMixin


class SIPTargetImage(SIPMixin, TargetImage):
    """
    A TargetImage with SIP distortion coefficients.
    This class is used to represent a target image with SIP distortion coefficients.
    It inherits from TargetImage and SIPMixin.
    """

    def jacobian_image(self, **kwargs):
        kwargs = {
            "pixel_area_map": self.pixel_area_map,
            "sipA": self.sipA,
            "sipB": self.sipB,
            "sipAP": self.sipAP,
            "sipBP": self.sipBP,
            "distortion_ij": self.distortion_ij,
            "distortion_IJ": self.distortion_IJ,
            **kwargs,
        }
        return super().jacobian_image(**kwargs)

    def model_image(self, upsample=1, pad=0, **kwargs):
        new_area_map = self.pixel_area_map
        new_distortion_ij = self.distortion_ij
        new_distortion_IJ = self.distortion_IJ
        if upsample > 1:
            new_area_map = self.pixel_area_map.repeat_interleave(upsample, dim=0)
            new_area_map = new_area_map.repeat_interleave(upsample, dim=1)
            new_area_map = new_area_map / upsample**2
            U = torch.nn.Upsample(scale_factor=upsample, mode="bilinear", align_corners=False)
            new_distortion_ij = U(self.distortion_ij)
            new_distortion_IJ = U(self.distortion_IJ)
        if pad > 0:
            new_area_map = torch.nn.functional.pad(
                new_area_map, (pad, pad, pad, pad), mode="replicate"
            )
            new_distortion_ij = torch.nn.functional.pad(
                new_distortion_ij, (pad, pad, pad, pad), mode="replicate"
            )
            new_distortion_IJ = torch.nn.functional.pad(
                new_distortion_IJ, (pad, pad, pad, pad), mode="replicate"
            )
        kwargs = {
            "pixel_area_map": new_area_map,
            "sipA": self.sipA,
            "sipB": self.sipB,
            "sipAP": self.sipAP,
            "sipBP": self.sipBP,
            "distortion_ij": new_distortion_ij,
            "distortion_IJ": new_distortion_IJ,
            **kwargs,
        }
        return super().model_image(**kwargs)
