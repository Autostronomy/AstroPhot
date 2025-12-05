import pytest

import astrophot as ap
from astrophot.param import Param

from utils import make_basic_sersic


def test_param():

    a = Param("a", value=1.0, uncertainty=0.1, valid=(0, 2), prof=1.0)
    assert isinstance(a.uncertainty, ap.backend.array_type), "uncertainty should be a tensor"
    assert isinstance(a.prof, ap.backend.array_type), "prof should be a tensor"
    assert a.initialized, "parameter should be marked as initialized"
    assert a.soft_valid(a.value) == a.value, "soft valid should return the value if not near limits"
    assert (
        a.soft_valid(-1 * ap.backend.ones_like(a.value)) > a.valid[0]
    ), "soft valid should push values inside the limits"
    assert (
        a.soft_valid(3 * ap.backend.ones_like(a.value)) < a.valid[1]
    ), "soft valid should push values inside the limits"

    b = Param("b", value=[2.0, 3.0], uncertainty=[0.1, 0.1], valid=(1, None))
    assert ap.backend.all(
        b.soft_valid(-1 * ap.backend.ones_like(b.value)) > b.valid[0]
    ), "soft valid should push values inside the limits"
    assert b.prof is None

    c = Param("c", value=lambda P: P.a.value, valid=(None, 4.0))
    c.link(a)
    assert c.initialized, "pointer should be marked as initialized"
    assert c.uncertainty is None


def test_module():

    target = make_basic_sersic()
    model1 = ap.Model(name="test model 1", model_type="sersic galaxy model", target=target)
    model2 = ap.Model(name="test model 2", model_type="sersic galaxy model", target=target)
    model = ap.Model(name="test", model_type="group model", target=target, models=[model1, model2])
    model.initialize()

    U = ap.backend.ones_like(model.get_values()) * 0.1
    model.fill_dynamic_value_uncertainties(U)

    paramsu = model.get_values(attribute="uncertainty")
    assert ap.backend.all(ap.backend.isfinite(paramsu)), "All parameters should be finite"

    paramsn = model.build_params_array_names()
    assert all(isinstance(name, str) for name in paramsn), "All parameter names should be strings"

    paramsun = model.build_params_array_units()
    assert all(isinstance(unit, str) for unit in paramsun), "All parameter units should be strings"

    index = model.dynamic_params_array_index(model2.q)
    assert index == [9], "Parameter index should be correct"

    with pytest.raises(ValueError):
        model.dynamic_params_array_index(5.0)  # Not a Param instance
