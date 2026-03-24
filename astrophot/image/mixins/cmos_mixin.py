from typing import Optional, Tuple

from .. import func
from ... import config
from ...param import forward


class CMOSMixin:
    """
    A mixin class for CMOS image processing. This class can be used to add
    CMOS-specific functionality to image processing classes.
    """

    def __init__(
        self,
        *args,
        subpixel_loc: Tuple[float, float] = (0, 0),
        subpixel_scale: float = 1.0,
        filename: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, filename=filename, **kwargs)
        if filename is not None:
            return
        self.subpixel_loc = subpixel_loc
        self.subpixel_scale = subpixel_scale

    @property
    def base_scale(self):
        """Get the base scale of the image, which is the subpixel scale."""
        return self.subpixel_scale

    @forward
    def pixel_collecting_area(self, I_, J_, upsample):
        # CMOS pixels only sensitive in sub area, so scale the pixel collecting area
        return self.pixel_area * self.subpixel_scale**2 / upsample**2

    def pixel_center_meshgrid(self, window=None, pad=0, upsample=1):
        """Get a meshgrid of pixel coordinates in the image, centered on the pixel grid."""
        if window is None:
            window = self.window
        return func.cmos_pixel_center_meshgrid(
            window.extent, pad, upsample, self.subpixel_loc, config.DTYPE, config.DEVICE
        )

    def copy(self, **kwargs):
        return super().copy(
            subpixel_loc=self.subpixel_loc, subpixel_scale=self.subpixel_scale, **kwargs
        )

    def fits_info(self):
        info = super().fits_info()
        info["SPIXLOC1"] = self.subpixel_loc[0]
        info["SPIXLOC2"] = self.subpixel_loc[1]
        info["SPIXSCL"] = self.subpixel_scale
        return info

    def load(self, filename: str, hduext: int = 0):
        hdulist = super().load(filename, hduext=hduext)
        if "SPIXLOC1" in hdulist[hduext].header:
            self.subpixel_loc = (
                hdulist[0].header.get("SPIXLOC1", 0),
                hdulist[0].header.get("SPIXLOC2", 0),
            )
        if "SPIXSCL" in hdulist[hduext].header:
            self.subpixel_scale = hdulist[0].header.get("SPIXSCL", 1.0)
        return hdulist
