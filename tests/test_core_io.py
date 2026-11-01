"""Core tests: technology construction and the input-output identity."""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.io_model import build_technology, ensure_viable, toy_technology
from planlab.core.state import Technology


def test_toy_technology_satisfies_io_identity():
    tech = toy_technology(7)
    assert tech.n == 7
    resid = tech.base_output - tech.A @ tech.base_output - tech.base_final
    assert np.max(np.abs(resid)) <= 1e-9 * tech.base_output.max()


def test_column_sums_strictly_below_one():
    tech = toy_technology(9)
    colsum = tech.A.sum(axis=0)
    assert np.all(colsum < 1.0)
    assert np.all(colsum > 0.0)


def test_technology_rejects_non_productive_matrix():
    A = np.full((2, 2), 0.7)          # column sums 1.4 -> not productive
    with pytest.raises(ValueError, match="productive"):
        Technology(
            sectors=("a", "b"),
            A=A,
            labor=np.array([0.1, 0.1]),
            kappa=np.array([1.0, 1.0]),
            consumption_weights=np.ones(2),
            essentials=np.zeros(2),
            base_output=np.array([1.0, 1.0]),
            base_final=np.array([0.1, 0.1]),
        )


def test_technology_rejects_negative_coefficients():
    with pytest.raises(ValueError, match="negative"):
        Technology(
            sectors=("a", "b"),
            A=np.array([[0.1, -0.1], [0.0, 0.1]]),
            labor=np.array([0.1, 0.1]),
            kappa=np.array([1.0, 1.0]),
            consumption_weights=np.ones(2),
            essentials=np.zeros(2),
            base_output=np.array([1.0, 1.0]),
            base_final=np.array([0.9, 0.9]),
        )


def test_build_technology_computes_final_demand_from_identity():
    A = np.array([[0.1, 0.2], [0.3, 0.1]])
    x = np.array([100.0, 50.0])
    tech = build_technology(
        sectors=["a", "b"], A=A, labor=[0.5, 0.4], kappa=[2.0, 2.0],
        base_output=x, base_final=None,
    )
    expected = x - A @ x
    assert np.allclose(tech.base_final, expected)
    tech.validate_identity(x, tech.base_final)


def test_build_technology_rejects_inconsistent_final_demand():
    A = np.array([[0.1, 0.2], [0.3, 0.1]])
    with pytest.raises(ValueError, match="inconsistent"):
        build_technology(
            sectors=["a", "b"], A=A, labor=[0.5, 0.4], kappa=[2.0, 2.0],
            base_output=np.array([100.0, 50.0]), base_final=np.array([10.0, 10.0]),
        )


def test_ensure_viable_scales_down_when_residual_negative():
    # good 1 is used much more than it is produced: x - A x < 0 for sector 1
    A = np.array([[0.05, 0.90], [0.05, 0.05]])
    assert np.all(A.sum(axis=0) < 1.0)
    x = np.array([1.0, 100.0])
    residual_before = x - A @ x
    assert residual_before.min() < 0
    A2, theta = ensure_viable(A, x)
    assert 0.0 < theta < 1.0
    assert np.all(x - A2 @ x >= -1e-9)


def test_technology_helpers():
    tech = toy_technology(4)
    assert tech.n == 4
    L = tech.leontief_inverse()
    assert np.allclose(L @ (np.eye(4) - tech.A), np.eye(4), atol=1e-9)
    cap = tech.capacity_from_capital(tech.kappa * tech.base_output)
    assert np.allclose(cap, tech.base_output)
    with pytest.raises(KeyError):
        tech.sector_index("does-not-exist")


def test_sector_names_must_be_unique():
    from planlab.core.io_model import normalise_sector_names

    with pytest.raises(ValueError):
        normalise_sector_names(["a", "a"])
    with pytest.raises(ValueError):
        normalise_sector_names(["a", ""])


def test_essential_flags_are_booleanised():
    tech = toy_technology(4)
    assert set(np.unique(tech.essentials)).issubset({0.0, 1.0})
