from typing import List, Optional

import torch
import numpy as np
from torch.nn.functional import avg_pool2d

from .image_object import Image, Image_List
from astropy.io import fits
from .. import AP_config

__all__ = ["PSF_Image"]


class PSF_Image(Image):
    """Image object which represents a model of PSF. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert torch.all((torch.tensor(self.data.shape) % 2) == 1), "psf must have odd shape"

        self.psf_upscale = torch.as_tensor(
            kwargs.get("psf_upscale", 1), dtype=torch.int32, device=AP_config.ap_device
        )

        # set the band
        self.band = kwargs.get("band", None)

    @property
    def psf_border_int(self):
        return torch.ceil(
            (
                1
                + torch.flip(
                    torch.tensor(
                        self.data.shape,
                        dtype=AP_config.ap_dtype,
                        device=AP_config.ap_device,
                    ),
                    (0,),
                )
                / self.psf_upscale
            )
            / 2
        ).int()

    @property
    def psf_border(self):
        return self.pixelscale * self.psf_border_int

    def _save_image_list(self, image_list, psf_header):
        psf_header["IMAGE"] = "PSF"
        psf_header["UPSCALE"] = int(self.psf_upscale.detach().cpu().item())
        image_list.append(
            fits.ImageHDU(self.data.detach().cpu().numpy(), header=psf_header)
        )
        
        
