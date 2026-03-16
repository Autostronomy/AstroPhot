from .mixins import (
    SersicPSFMixin,
    iSersicPSFMixin,
    ExponentialPSFMixin,
    iExponentialPSFMixin,
    GaussianPSFMixin,
    iGaussianPSFMixin,
    FerrerPSFMixin,
    iFerrerPSFMixin,
    KingPSFMixin,
    iKingPSFMixin,
    MoffatPSFMixin,
    iMoffatPSFMixin,
    NukerPSFMixin,
    iNukerPSFMixin,
    SplinePSFMixin,
    iSplinePSFMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    TruncationMixin,
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

__all__ = []
for mixin in radial_models:
    # PSF Model
    g_mixin = type(mixin.__name__[:-5], (mixin, RadialMixin, PSFModel), {"usable": True})
    globals()[g_mixin.__name__] = g_mixin
    __all__.append(g_mixin.__name__)

    # Ellipse PSF Model
    g_mixin = type(
        mixin.__name__[:-5] + "Ellipse",
        (mixin, InclinedMixin, RadialMixin, PSFModel),
        {"usable": True, "_model_type": "ellipse"},
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
