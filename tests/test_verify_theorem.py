"""Tests for the independent verification suite (owner: T6 verifier).

These tests exercise the *verifier's own* code: the theorem re-derivation, the
from-scratch one-period solver, the accounting helpers and the statistics.  They
are fast (a few seconds) and never depend on results/.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verification.check1_theorem import (boundary_schur, companion_radius,
                                         g_closed_form, log_growth, schur_stable)
from verification.check2_engine import (independent_period, ration_by_tier,
                                        solve_goods_balance)


# ---------------------------------------------------------------------------
# check 1 — the theorem
# ---------------------------------------------------------------------------
def test_closed_form_known_values():
    assert g_closed_form(0) == pytest.approx(2.0, abs=1e-15)
    assert g_closed_form(1) == pytest.approx(1.0, abs=1e-15)
    assert g_closed_form(2) == pytest.approx((math.sqrt(5) - 1) / 2, abs=1e-15)
    assert g_closed_form(3) == pytest.approx(0.4450418679126288, abs=1e-15)


def test_closed_form_is_decreasing_and_asymptotic():
    values = [g_closed_form(m) for m in range(0, 200)]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))
    assert g_closed_form(200) * (2 * 200 + 1) / math.pi == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("m", [0, 1, 2, 3, 5, 8])
def test_root_free_schur_boundary_matches_closed_form(m):
    assert boundary_schur(m) == pytest.approx(g_closed_form(m), abs=1e-8)


@pytest.mark.parametrize("m", [1, 3, 6])
def test_spectral_radius_crosses_one_at_boundary(m):
    gb = g_closed_form(m)
    assert companion_radius(m, 0.97 * gb) < 1.0
    assert companion_radius(m, 1.03 * gb) > 1.0
    assert companion_radius(m, gb) == pytest.approx(1.0, abs=1e-8)


def test_time_domain_growth_changes_sign():
    gb = g_closed_form(2)
    below, _ = log_growth(0.95 * gb, 2, steps=2000, warmup=400)
    above, _ = log_growth(1.05 * gb, 2, steps=2000, warmup=400)
    assert below < 0.0 < above


def test_schur_test_rejects_non_monic():
    with pytest.raises(ValueError):
        schur_stable(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# check 2 — the from-scratch period
# ---------------------------------------------------------------------------
def _tiny_technology():
    from planlab.core.io_model import toy_technology

    return toy_technology(6)


def test_ration_by_tier_is_ladder_consistent():
    supply = np.array([10.0, 5.0])
    claims = np.array([[1.0, 1.0],     # tier 0
                       [2.0, 2.0],     # tier 0
                       [8.0, 8.0],     # tier 1
                       [8.0, 8.0]])    # tier 2
    tiers = np.array([0.0, 0.0, 1.0, 2.0])
    grants = ration_by_tier(supply, claims, tiers)
    assert np.all(grants.sum(axis=0) <= supply + 1e-12)
    # tier 0 fully served; the remaining supply goes to tier 1 only
    assert grants[:2].sum(axis=0) == pytest.approx(claims[:2].sum(axis=0))
    assert grants[3].sum() == pytest.approx(0.0)
    assert grants[2] == pytest.approx(np.array([7.0, 2.0]))


def test_independent_period_reproduces_engine_on_toy_technology():
    from planlab.core.engine import Simulation
    from planlab.core.scenarios import scenario_from_dict
    from planlab.mechanisms import get_mechanism

    tech = _tiny_technology()
    sc = scenario_from_dict({
        "id": "verify_toy", "mechanism": "material_balance",
        "technology": {"toy": 6}, "periods": 8, "seeds": [0], "burn_in": 2,
        "info": {"lag": 0, "noise": 0.02},
        "shocks": [{"type": "capacity", "sector": "s02", "start": 3, "duration": 2,
                    "size": 0.6}],
    })
    captured = {}
    original = Simulation._reconcile

    def wrapper(self, **kwargs):
        snap = {key: np.array(kwargs[key], dtype=float, copy=True)
                for key in ("capacity", "inventory", "demand", "gov_demand", "invest_demand")}
        snap["targets"] = np.array(kwargs["plan"].targets, dtype=float)
        snap["claims"] = np.array(kwargs["alloc"].claims, dtype=float)
        snap["tiers"] = np.array(kwargs["alloc"].tiers, dtype=float)
        snap["labels"] = tuple(kwargs["alloc"].labels)
        snap["labour"] = float(kwargs["labour_supply"])
        out = original(self, **kwargs)
        captured[int(self.state.t)] = (snap, out)
        return out

    Simulation._reconcile = wrapper
    try:
        mech = get_mechanism("material_balance", tech, {})
        Simulation(tech, mech, sc, seed=0).run()
    finally:
        Simulation._reconcile = original

    assert len(captured) == 8
    worst = 0.0
    for t, (snap, out) in captured.items():
        mine = independent_period(
            A=tech.A, labor=tech.labor, targets=snap["targets"],
            capacity=snap["capacity"], inventory=snap["inventory"],
            labour_supply=snap["labour"], claims=snap["claims"], tiers=snap["tiers"],
            labels=snap["labels"], demand=snap["demand"],
            gov_demand=snap["gov_demand"], invest_demand=snap["invest_demand"],
            base_output=tech.base_output, max_stock_ratio=sc.max_stock_ratio,
        )
        x, Z, C, G, Inv, S_end, unmet = out[:7]
        for a, b in ((x, mine["x"]), (Z, mine["Z"]), (C, mine["C"]), (G, mine["G"]),
                     (Inv, mine["invest"]), (S_end, mine["inventory_end"]),
                     (unmet, mine["unmet"])):
            scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale)
    assert worst < 1e-10


def test_solve_goods_balance_respects_bounds_and_constraints():
    from planlab.core.io_model import toy_technology

    tech = _tiny_technology()
    ub = np.asarray(tech.base_output) * 1.3
    r = np.zeros(tech.n)
    x, obj = solve_goods_balance(tech.A, ub, r, np.zeros(tech.n), tech.labor,
                                 1e9 * float(tech.labor @ ub))
    assert np.all(x <= ub + 1e-9)
    assert np.all(tech.A @ x - x <= r + 1e-9 * max(1.0, float(np.max(ub))))
    assert obj == pytest.approx(float(np.sum(x)), rel=1e-12)
