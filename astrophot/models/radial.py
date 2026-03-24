from .mixins import (
    SersicMixin,
    iSersicMixin,
    ExponentialMixin,
    iExponentialMixin,
    GaussianMixin,
    iGaussianMixin,
    FerrerMixin,
    iFerrerMixin,
    KingMixin,
    iKingMixin,
    MoffatMixin,
    iMoffatMixin,
    NukerMixin,
    iNukerMixin,
    SplineMixin,
    iSplineMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    TruncationMixin,
)
from .galaxy_model_object import GalaxyModel

radial_models = (
    SersicMixin,
    ExponentialMixin,
    GaussianMixin,
    FerrerMixin,
    KingMixin,
    MoffatMixin,
    NukerMixin,
    SplineMixin,
)

__all__ = []
for mixin in radial_models:
    # Galaxy Model
    g_mixin = type(
        mixin.__name__[:-5] + "Galaxy", (mixin, RadialMixin, GalaxyModel), {"usable": True}
    )
    globals()[g_mixin.__name__] = g_mixin
    __all__.append(g_mixin.__name__)

    # Truncated Galaxy Model
    t_mixin = type(
        "T" + mixin.__name__[:-5] + "Galaxy",
        (TruncationMixin, mixin, RadialMixin, GalaxyModel),
        {"usable": True},
    )
    globals()[t_mixin.__name__] = t_mixin
    __all__.append(t_mixin.__name__)

    for n, p in zip(
        ("SuperEllipse", "FourierEllipse", "Warp"),
        (SuperEllipseMixin, FourierEllipseMixin, WarpMixin),
    ):
        # Galaxy Model with additional perturbation mixin
        g_mixin = type(
            mixin.__name__[:-5] + n, (mixin, RadialMixin, p, GalaxyModel), {"usable": True}
        )
        globals()[g_mixin.__name__] = g_mixin
        __all__.append(g_mixin.__name__)

        # Truncated Galaxy Model with additional perturbation mixin
        t_mixin = type(
            "T" + mixin.__name__[:-5] + n,
            (TruncationMixin, mixin, RadialMixin, p, GalaxyModel),
            {"usable": True},
        )
        globals()[t_mixin.__name__] = t_mixin
        __all__.append(t_mixin.__name__)

iradial_models = (
    iSersicMixin,
    iExponentialMixin,
    iGaussianMixin,
    iFerrerMixin,
    iKingMixin,
    iMoffatMixin,
    iNukerMixin,
    iSplineMixin,
)

for mixin in iradial_models:
    # Ray Galaxy Model
    r_mixin = type(mixin.__name__[1:-5] + "Ray", (mixin, RayMixin, GalaxyModel), {"usable": True})
    globals()[r_mixin.__name__] = r_mixin
    __all__.append(r_mixin.__name__)

    # Wedge Galaxy Model
    w_mixin = type(
        mixin.__name__[1:-5] + "Wedge", (mixin, WedgeMixin, GalaxyModel), {"usable": True}
    )
    globals()[w_mixin.__name__] = w_mixin
    __all__.append(w_mixin.__name__)
