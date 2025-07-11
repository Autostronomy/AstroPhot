from astrophot.param import Param
import torch


def test_param():

    a = Param("a", value=1.0, uncertainty=0.1, valid=(0, 2), prof=1.0)
    assert a.is_valid(1.5), "value should be valid"
    assert isinstance(a.uncertainty, torch.Tensor), "uncertainty should be a tensor"
    assert isinstance(a.prof, torch.Tensor), "prof should be a tensor"
    assert a.initialized, "parameter should be marked as initialized"
    assert a.soft_valid(a.value) == a.value, "soft valid should return the value if not near limits"
    assert (
        a.soft_valid(-1 * torch.ones_like(a.value)) > a.valid[0]
    ), "soft valid should push values inside the limits"
    assert (
        a.soft_valid(3 * torch.ones_like(a.value)) < a.valid[1]
    ), "soft valid should push values inside the limits"

    b = Param("b", value=[2.0, 3.0], uncertainty=[0.1, 0.1], valid=(1, None))
    assert not b.is_valid(0.5), "value should not be valid"
    assert b.is_valid(10.5), "value should be valid"
    assert torch.all(
        b.soft_valid(-1 * torch.ones_like(b.value)) > b.valid[0]
    ), "soft valid should push values inside the limits"
    assert b.prof is None

    c = Param("c", value=lambda P: P.a.value, valid=(None, 4.0))
    c.link(a)
    assert c.initialized, "pointer should be marked as initialized"
    assert c.is_valid(0.5), "value should be valid"
    assert c.uncertainty is None
