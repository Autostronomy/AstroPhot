from typing import Optional

from scipy.stats import iqr
import torch
import numpy as np

from .model_object import Component_Model
from ._shared_methods import select_target
from ..utils.initialize import isophotes
from ..utils.angle_operations import Angle_Average
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.conversions.coordinates import (
    Rotate_Cartesian,
    Axis_Ratio_Cartesian,
)

__all__ = ["Edgeon_Model"]


class Edgeon_Model(Component_Model):
    """General Edge-On galaxy model to be subclassed for any specific
    representation such as radial light profile or the structure of
    the galaxy on the sky. Defines an edgeon galaxy as an object with
    a position angle, no inclination information is included.

    """

    model_type = f"edgeon {Component_Model.model_type}"
    parameter_specs = {
        "PA": {
            "units": "rad",
            "limits": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.06,
        },
    }
    _parameter_order = Component_Model._parameter_order + ("PA",)
    useable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self, target=None, parameters: Optional["Parameter_Group"] = None, **kwargs
    ):
        super().initialize(target=target, parameters=parameters)
        if parameters["PA"].value is not None:
            return
        target_area = target[self.window]
        edge = np.concatenate(
            (
                target_area.data[:, 0].detach().cpu().numpy(),
                target_area.data[:, -1].detach().cpu().numpy(),
                target_area.data[0, :].detach().cpu().numpy(),
                target_area.data[-1, :].detach().cpu().numpy(),
            )
        )
        edge_average = np.median(edge)
        edge_scatter = iqr(edge, rng=(16, 84)) / 2
        icenter = target_area.world_to_pixel(parameters["center"].value)

        iso_info = isophotes(
            target_area.data.detach().cpu().numpy() - edge_average,
            (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
            threshold=3 * edge_scatter,
            pa=0.0,
            q=1.0,
            n_isophotes=15,
        )
        parameters["PA"].set_value(
            (
                -(
                    (
                        Angle_Average(
                            list(
                                iso["phase2"]
                                for iso in iso_info[-int(len(iso_info) / 3) :]
                            )
                        )
                        / 2
                    )
                    + target.north
                )
            )
            % np.pi,
            override_locked=True,
        )

    @default_internal
    def transform_coordinates(self, X, Y, image=None, parameters=None):
        return Rotate_Cartesian(-(parameters["PA"].value - image.north), X, Y)

    @default_internal
    def evaluate_model(
        self,
        X=None,
        Y=None,
        image: "Image" = None,
        parameters: "Parameter_Group" = None,
        **kwargs,
    ):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        XX, YY = self.transform_coordinates(X, Y, image=image, parameters=parameters)

        return self.brightness_model(
            torch.abs(XX), torch.abs(YY), image=image, parameters=parameters
        )


class Edgeon_Sech(Edgeon_Model):
    """An edgeon profile where the vertical distribution is a sech^2
    profile, subclasses define the radial profile.

    """

    model_type = f"sech2 {Edgeon_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "hs": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Edgeon_Model._parameter_order + ("I0", "hs")
    useable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self, target=None, parameters: Optional["Parameter_Group"] = None, **kwargs
    ):
        super().initialize(target=target, parameters=parameters)
        if (parameters["I0"].value is not None) and (
            parameters["hs"].value is not None
        ):
            return
        target_area = target[self.window]
        icenter = target_area.world_to_pixel(parameters["center"].value)

        if parameters["I0"].value is None:
            parameters["I0"].set_value(
                torch.log10(
                    torch.mean(
                        target_area.data[
                            int(icenter[0]) - 2 : int(icenter[0]) + 2,
                            int(icenter[1]) - 2 : int(icenter[1]) + 2,
                        ]
                    )
                    / target.pixel_area.item()
                ),
                override_locked=True,
            )
            parameters["I0"].set_uncertainty(
                torch.std(
                    target_area.data[
                        int(icenter[0]) - 2 : int(icenter[0]) + 2,
                        int(icenter[1]) - 2 : int(icenter[1]) + 2,
                    ]
                )
                / (torch.abs(parameters["I0"].value) * target.pixel_area),
                override_locked=True,
            )
        if parameters["hs"].value is None:
            parameters["hs"].set_value(
                torch.max(self.window.shape) * 0.1, override_locked=True
            )
            parameters["hs"].set_value(parameters["hs"].value / 2, override_locked=True)

    @default_internal
    def brightness_model(self, X, Y, image=None, parameters=None):
        return (
            (image.pixel_area * 10 ** parameters["I0"].value)
            * self.radial_model(X, image=image, parameters=parameters)
            / (torch.cosh((Y + self.softening) / parameters["hs"].value) ** 2)
        )


class Edgeon_Isothermal(Edgeon_Sech):
    """A self-gravitating locally-isothermal edgeon disk. This comes from
    van der Kruit & Searle 1981.

    """

    model_type = f"isothermal {Edgeon_Sech.model_type}"
    parameter_specs = {
        "rs": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Edgeon_Sech._parameter_order + ("rs",)
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self, target=None, parameters: Optional["Parameter_Group"] = None, **kwargs
    ):
        super().initialize(target=target, parameters=parameters)
        if parameters["rs"].value is not None:
            return
        parameters["rs"].set_value(
            torch.max(self.window.shape) * 0.4, override_locked=True
        )
        parameters["rs"].set_value(parameters["rs"].value / 2, override_locked=True)

    @default_internal
    def radial_model(self, R, image=None, parameters=None):
        Rscaled = torch.abs((R + self.softening) / parameters["rs"].value)
        return (
            Rscaled
            * torch.exp(-Rscaled)
            * torch.special.scaled_modified_bessel_k1(Rscaled)
        )
