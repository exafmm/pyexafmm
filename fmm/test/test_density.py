import numpy as np
import pytest

import fmm.density as density


@pytest.mark.parametrize(
    "surface, dnsity",
    [
        ("foo", "bar")
    ]
)
def test_charge_errors(surface, dnsity):
    with pytest.raises(Exception) as e_info:
        obj = density.Charge(surface, dnsity)
        assert "`surface` and `density` must be numpy arrays" in str(e_info.value)


@pytest.mark.parametrize(
    "surface, dnsity",
    [
        (np.array([[0, 0, 0]]), np.array([1]))
    ]
)
def test_charge_instantiation(surface, dnsity):
    obj = density.Charge(surface, dnsity)
    assert isinstance(obj, density.AbstractDensity)


@pytest.mark.parametrize(
    "surface, dnsity",
    [
        ("foo", "bar")
    ]
)
def test_potential_errors(surface, dnsity):
    with pytest.raises(Exception) as e_info:
        obj = density.Potential(surface, dnsity)
        assert "`surface` and `density` must be numpy arrays" in str(e_info.value)


@pytest.mark.parametrize(
    "surface, dnsity, indices",
    [
        (np.array([[0, 0, 0]]), np.array([1]), None),

    ]
)
def test_potential_instantiation(surface, dnsity, indices):
    obj = density.Potential(surface, dnsity, indices)
    assert isinstance(obj, density.AbstractDensity)

    if indices is None:
        assert isinstance(obj.indices, set)
