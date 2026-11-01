"""Core tests: the feasibility programme (the short side of the market)."""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.feasibility import feasible_output, is_feasible


def simple_A():
    # one sector uses the other; both productive
    return np.array([[0.2, 0.4], [0.3, 0.1]])


def test_feasible_targets_returned_unchanged():
    A = simple_A()
    ub = np.array([10.0, 10.0])
    inv = np.array([5.0, 5.0])
    x, status = feasible_output(A, ub, inv)
    assert status == "infeasible-ub"
    assert np.allclose(x, ub)


def test_infeasible_targets_are_reduced_and_balance_holds():
    A = simple_A()
    ub = np.array([100.0, 1.0])     # sector 1 badly under-provisioned
    inv = np.zeros(2)
    x, status = feasible_output(A, ub, inv)
    assert status == "optimal"
    assert np.all(x <= ub + 1e-9)
    assert np.all(A @ x - x <= inv + 1e-8)


def test_monotone_in_capacity_bound():
    A = simple_A()
    inv = np.array([1.0, 1.0])
    _, _ = feasible_output(A, np.array([5.0, 5.0]), inv)
    low, _ = feasible_output(A, np.array([1.0, 5.0]), inv)
    high, _ = feasible_output(A, np.array([3.0, 5.0]), inv)
    assert np.all(high >= low - 1e-9)


def test_reserve_reduces_feasible_output():
    A = simple_A()
    ub = np.array([10.0, 10.0])
    inv = np.array([2.0, 2.0])
    without, _ = feasible_output(A, ub, inv, reserve=np.zeros(2))
    with_reserve, _ = feasible_output(A, ub, inv, reserve=np.array([2.0, 2.0]))
    assert np.sum(with_reserve) <= np.sum(without) + 1e-9


def test_capacity_zero_gives_zero_output():
    A = simple_A()
    x, status = feasible_output(A, np.zeros(2), np.zeros(2))
    assert np.allclose(x, 0.0)


def test_labour_constraint_binds():
    A = simple_A()
    ub = np.array([10.0, 10.0])
    inv = np.array([100.0, 100.0])
    labor = np.array([1.0, 1.0])
    x, status = feasible_output(A, ub, inv, labor=labor, labour_supply=5.0)
    assert float(labor @ x) <= 5.0 + 1e-6


def test_is_feasible_matches_programme():
    A = simple_A()
    inv = np.array([1.0, 1.0])
    # balanced vector: A x - x <= 0 <= inventory, so feasible
    assert is_feasible(A, np.array([2.0, 2.0]), inv, np.zeros(2))
    # a lopsided vector makes good 2 scarce: A x - x = [-39.6, 14.1] > inventory
    assert not is_feasible(A, np.array([50.0, 1.0]), inv, np.zeros(2))


def test_empty_technology():
    x, status = feasible_output(np.zeros((0, 0)), np.zeros(0), np.zeros(0))
    assert x.shape == (0,)
    assert status == "empty"
