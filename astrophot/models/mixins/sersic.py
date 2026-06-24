from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import sersic_np
from .. import func


def _x0_func(model, R, F):
    return 2.0, R[4], 10 ** F[4]


class SersicMixin:
    """Sersic radial light profile (Sersic 1963).

    This is a classic profile used widely in galaxy modelling. It can be a good
    starting point for many extended objects. The functional form of the Sersic
    profile is defined as:

    .. math::

       I(R) = I_e \\exp(- b_n((R/R_e)^{1/n} - 1))

    It is a generalization of a gaussian, exponential, and de-Vaucouleurs
    profile. The Sersic index ``n`` controls the shape of the profile, with ``n=1``
    being an exponential profile, ``n=4`` being a de-Vaucouleurs profile, and
    ``n=0.5`` being a Gaussian profile.

    :param n: Sersic index which controls the shape of the brightness profile
    :param Re: half light radius, also called effective radius [arcsec]
    :param Ie: intensity at the half light radius [flux/arcsec^2]
    """

    _model_type = "sersic"
    _parameter_specs = {
        "n": {
            "units": "none",
            "valid": (0.36, 8),
            "shape": (),
            "dynamic": True,
            "description": "Sersic index which controls the shape of the brightness profile",
        },
        "Re": {
            "units": "arcsec",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "half light radius, also called effective radius [arcsec]",
        },
        "Ie": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "intensity at the half light radius [flux/arcsec^2]",
        },
    }

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self, self.target[self.window], sersic_np, ("n", "Re", "Ie"), _x0_func
        )

    @forward
    def radial_model(self, R: ArrayLike, n: ArrayLike, Re: ArrayLike, Ie: ArrayLike) -> ArrayLike:
        return func.sersic(R, n, Re, Ie)


class iSersicMixin:
    """Sersic radial light profile (Sersic 1963).

    This is a classic profile used widely in galaxy modelling. It can be a good
    starting point for many extended objects. The functional form of the Sersic
    profile is defined as:

    .. math::

       I(R) = I_e \\exp(- b_n((R/R_e)^{1/n} - 1))

    It is a generalization of a gaussian, exponential, and de-Vaucouleurs
    profile. The Sersic index ``n`` controls the shape of the profile, with ``n=1``
    being an exponential profile, ``n=4`` being a de-Vaucouleurs profile, and
    ``n=0.5`` being a Gaussian profile.

    ``n``, ``Re``, and ``Ie`` are batched by their first dimension, allowing for
    multiple Sersic profiles to be defined at once.

    :param n: Sersic index which controls the shape of the brightness profile
    :param Re: half light radius, also called effective radius [arcsec]
    :param Ie: intensity at the half light radius [flux/arcsec^2]
    """

    _model_type = "sersic"
    _parameter_specs = {
        "n": {
            "units": "none",
            "valid": (0.36, 8),
            "shape": (None,),
            "dynamic": True,
            "description": "Sersic index which controls the shape of the brightness profile",
        },
        "Re": {
            "units": "arcsec",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "half light radius, also called effective radius",
        },
        "Ie": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "intensity at the half light radius, also called effective intensity",
        },
    }

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=sersic_np,
            params=("n", "Re", "Ie"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(
        self, i: int, R: ArrayLike, n: ArrayLike, Re: ArrayLike, Ie: ArrayLike
    ) -> ArrayLike:
        return func.sersic(R, n[i], Re[i], Ie[i])


class SersicPSFMixin:
    """Sersic radial light profile (Sersic 1963).

    This is a classic profile used widely in galaxy modelling, though it
    is simply a generalization of a Gaussian. It can be a good starting
    point for many objects. The functional form of the Sersic profile is
    defined as:

    .. math::

       I(R) = I_e \\exp(- b_n((R/R_e)^{1/n} - 1))

    It includes the gaussian, exponential, and de-Vaucouleurs
    profiles. The Sersic index ``n`` controls the shape of the profile, with ``n=1``
    being an exponential profile, ``n=4`` being a de-Vaucouleurs profile, and
    ``n=0.5`` being a Gaussian profile.

    :param n: Sersic index which controls the shape of the brightness profile
    :param Re: half light radius, also called effective radius [pix]
    :param Ie: intensity at the half light radius, also called effective intensity [flux/pix^2]
    """

    _model_type = "sersic"
    _parameter_specs = {
        "n": {
            "units": "none",
            "valid": (0.36, 8),
            "shape": (),
            "dynamic": True,
            "description": "Sersic index which controls the shape of the brightness profile",
        },
        "Re": {
            "units": "pix",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "half light radius, also called effective radius",
        },
        "Ie": {
            "units": "flux/pix^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": False,
            "value": 1.0,
            "description": "intensity at the half light radius, also called effective intensity",
        },
    }

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self, self.target[self.window], sersic_np, ("n", "Re", "Ie"), _x0_func
        )

    @forward
    def radial_model(self, R: ArrayLike, n: ArrayLike, Re: ArrayLike, Ie: ArrayLike) -> ArrayLike:
        return func.sersic(R, n, Re, Ie)
