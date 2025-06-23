import torch

from .galaxy_model_object import GalaxyModel

# from .warp_model import Warp_Galaxy

__all__ = [
    "SuperEllipseGalaxy",
    # "SuperEllipse_Warp"
]


class SuperEllipseGalaxy(GalaxyModel):
    """Expanded galaxy model which includes a superellipse transformation
    in its radius metric. This allows for the expression of "boxy" and
    "disky" isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation, especially
    for early-type galaxies. The functional form for this is:

    R = (|X|^C + |Y|^C)^(1/C)

    where R is the new distance metric, X Y are the coordinates, and C
    is the coefficient for the superellipse. C can take on any value
    greater than zero where C = 2 is the standard distance metric, 0 <
    C < 2 creates disky or pointed perturbations to an ellipse, and C
    > 2 transforms an ellipse to be more boxy.

    Parameters:
        C: superellipse distance metric parameter.

    """

    _model_type = "superellipse"
    _parameter_specs = {
        "C": {"units": "none", "value": 2.0, "uncertainty": 1e-2, "valid": (0, None)},
    }
    usable = False

    def radius_metric(self, x, y, C):
        return torch.pow(x.abs().pow(C) + y.abs().pow(C), 1.0 / C)


# class SuperEllipse_Warp(Warp_Galaxy):
#     """Expanded warp model which includes a superellipse transformation
#     in its radius metric. This allows for the expression of "boxy" and
#     "disky" isophotes instead of pure ellipses. This is a common
#     extension of the standard elliptical representation, especially
#     for early-type galaxies. The functional form for this is:

#     R = (|X|^C + |Y|^C)^(1/C)

#     where R is the new distance metric, X Y are the coordinates, and C
#     is the coefficient for the superellipse. C can take on any value
#     greater than zero where C = 2 is the standard distance metric, 0 <
#     C < 2 creates disky or pointed perturbations to an ellipse, and C
#     > 2 transforms an ellipse to be more boxy.

#     Parameters:
#         C0: superellipse distance metric parameter where C0 = C-2 so that a value of zero is now a standard ellipse.


#     """

#     model_type = f"superellipse {Warp_Galaxy.model_type}"
#     parameter_specs = {
#         "C0": {"units": "C-2", "value": 0.0, "uncertainty": 1e-2, "limits": (-2, None)},
#     }
#     _parameter_order = Warp_Galaxy._parameter_order + ("C0",)
#     usable = False

#     @default_internal
#     def radius_metric(self, X, Y, image=None, parameters=None):
#         return torch.pow(
#             torch.pow(torch.abs(X), parameters["C0"].value + 2.0)
#             + torch.pow(torch.abs(Y), parameters["C0"].value + 2.0),
#             1.0 / (parameters["C0"].value + 2.0),
#         )  # epsilon added for numerical stability of gradient
