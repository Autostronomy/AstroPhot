from .mixins import (
    SersicPSFMixin,
    ExponentialPSFMixin,
    GaussianPSFMixin,
    FerrerPSFMixin,
    KingPSFMixin,
    MoffatPSFMixin,
    NukerPSFMixin,
    SplinePSFMixin,
    RadialMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    InclinedMixin,
)
from .psf_model_object import PSFModel

radial_models = (
    SersicPSFMixin,
    ExponentialPSFMixin,
    GaussianPSFMixin,
    FerrerPSFMixin,
    KingPSFMixin,
    MoffatPSFMixin,
    NukerPSFMixin,
    SplinePSFMixin,
)

EllipseMixin = type("EllipseMixin", (InclinedMixin,), {"usable": False, "_model_type": "ellipse"})
__all__ = []
for mixin in radial_models:
    # PSF Model
    g_mixin = type(mixin.__name__[:-5], (mixin, RadialMixin, PSFModel), {"usable": True})
    globals()[g_mixin.__name__] = g_mixin
    __all__.append(g_mixin.__name__)

    # Ellipse PSF Model
    g_mixin = type(
        mixin.__name__[:-5] + "Ellipse",
        (mixin, EllipseMixin, RadialMixin, PSFModel),
        {"usable": True},
    )
    globals()[g_mixin.__name__] = g_mixin
    __all__.append(g_mixin.__name__)

    for n, p in zip(
        ("SuperEllipse", "FourierEllipse", "Warp"),
        (SuperEllipseMixin, FourierEllipseMixin, WarpMixin),
    ):
        # Galaxy Model with additional perturbation mixin
        g_mixin = type(
            mixin.__name__[:-5] + n,
            (mixin, InclinedMixin, RadialMixin, p, PSFModel),
            {"usable": True},
        )
        globals()[g_mixin.__name__] = g_mixin
        __all__.append(g_mixin.__name__)
