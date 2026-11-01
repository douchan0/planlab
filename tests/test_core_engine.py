"""Core tests: engine accounting, feasibility and determinism.

These are the tests that protect the scientific validity of every number in the
paper.  If the balance identity leaks, the welfare comparisons are meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import AccountingError, run_scenario
from planlab.core.io_model import toy_technology
from planlab.core.scenarios import scenario_from_dict

from conftest import balance_residual, make_scenario


@pytest.mark.parametrize("mechanism", ["material_balance", "lp_central"])
def test_calm_conditions_reproduce_base_year(toy_tech, mechanism):
    """With no shock, no noise and no lag the base year must be a fixed point."""
    sc = make_scenario(mechanism, id=f"calm_{mechanism}", periods=20, burn_in=5)
    res = run_scenario(sc, technology=toy_tech)
    last = res.records[-1]
    assert np.allclose(last.x, toy_tech.base_output, rtol=1e-6)
    assert res.run_metrics["mean_shortage_rate"] < 1e-9
    assert res.run_metrics["mean_plan_fulfilment"] > 0.999
    # inventory is stationary, not accumulating or draining
    ratio = last.inventory_end / toy_tech.base_output
    assert np.allclose(ratio, sc.stock_ratio, atol=1e-6)


@pytest.mark.parametrize("mechanism", ["material_balance", "lp_central"])
def test_balance_identity_holds_every_period(toy_tech, mechanism):
    sc = make_scenario(
        mechanism,
        id=f"shock_{mechanism}",
        periods=45,
        shocks=[{"type": "capacity", "sector": "s02", "start": 10, "duration": 6, "size": 0.6}],
    )
    res = run_scenario(sc, technology=toy_tech)
    worst = max(balance_residual(r) for r in res.records)
    assert worst < 1e-8, f"balance identity violated by {worst:.3e}"


def test_no_nan_or_inf_anywhere(toy_tech):
    sc = make_scenario(
        "material_balance",
        id="nan_check",
        periods=40,
        info={"lag": 3, "noise": 0.15},
        demand={"growth": 0.01, "ar1": 0.5, "sigma": 0.05},
        shocks=[
            {"type": "capacity", "sector": "s03", "start": 8, "duration": 10, "size": 0.5},
            {"type": "demand", "sector": "s01", "start": 15, "duration": 5, "size": 1.3},
        ],
    )
    res = run_scenario(sc, technology=toy_tech)
    for r in res.records:
        for arr in (r.x, r.Z, r.C, r.G, r.invest, r.inventory_end, r.demand, r.unmet, r.price):
            assert np.all(np.isfinite(arr))
        assert np.all(r.x >= 0)
        assert np.all(r.C >= 0)
        assert np.all(r.inventory_end >= -1e-9)


def test_same_seed_same_result(toy_tech):
    sc = make_scenario(
        "material_balance", id="det", periods=25,
        info={"lag": 2, "noise": 0.1}, demand={"sigma": 0.05, "ar1": 0.3},
    )
    a = run_scenario(sc, technology=toy_tech, seed=7)
    b = run_scenario(sc, technology=toy_tech, seed=7)
    for ra, rb in zip(a.records, b.records):
        assert np.array_equal(ra.x, rb.x)
        assert np.array_equal(ra.price, rb.price)


def test_different_seed_different_path(toy_tech):
    sc = make_scenario(
        "material_balance", id="det2", periods=25,
        info={"lag": 2, "noise": 0.1}, demand={"sigma": 0.05, "ar1": 0.3},
    )
    a = run_scenario(sc, technology=toy_tech, seed=1)
    b = run_scenario(sc, technology=toy_tech, seed=2)
    assert not np.array_equal(a.records[-1].x, b.records[-1].x)


def test_common_random_numbers_across_mechanisms(toy_tech):
    """Two mechanisms with the same seed must face the same exogenous path."""
    from planlab.core.engine import Simulation
    from planlab.mechanisms import get_mechanism

    sc = make_scenario(
        "material_balance", id="crn", periods=12,
        info={"lag": 0, "noise": 0.0},
        demand={"sigma": 0.1, "ar1": 0.4, "income_elasticity": 0.0},
        household={"backlog_persistence": 0.0},
    )
    demands = {}
    for mech_name in ("material_balance", "lp_central"):
        mech = get_mechanism(mech_name, toy_tech, {})
        res = Simulation(toy_tech, mech, sc, seed=42).run()
        demands[mech_name] = np.array([r.demand for r in res.records])
    assert np.allclose(demands["material_balance"], demands["lp_central"])


def test_capacity_shock_bites(toy_tech):
    sc = make_scenario(
        "material_balance", id="bite", periods=30,
        shocks=[{"type": "capacity", "sector": "s02", "start": 10, "duration": 6, "size": 0.5}],
    )
    res = run_scenario(sc, technology=toy_tech)
    during = res.records[11]
    assert during.x[1] < 0.75 * toy_tech.base_output[1]
    assert res.run_metrics["mean_shortage_rate"] > 0.0
    assert res.run_metrics["shock_peak_shortage"] > 0.05


def test_observation_is_delayed_and_labelled(toy_tech):
    from planlab.core.engine import build_observation

    hist = {
        "demand": [np.full(3, float(i)) for i in range(12)],
        "capacity": [np.full(3, float(i)) for i in range(12)],
        "inventory": [np.zeros(3) for _ in range(12)],
        "price": [np.ones(3) for _ in range(12)],
        "backlog": [np.zeros(3) for _ in range(12)],
        "labour": [1.0] * 12,
        "shortage": [np.zeros(3) for _ in range(12)],
    }
    obs = build_observation(
        t=8, lag=3, noise=0.0, rng=np.random.default_rng(0), hist=hist,
        observe_capacity=True, observe_inventory=True, observe_backlog=True,
        fallback_capacity=np.zeros(3), fallback_inventory=np.zeros(3),
        fallback_backlog=np.zeros(3), fallback_labour=1.0,
    )
    assert obs.meta["observed_period"] == 8 - 1 - 3
    assert obs.meta["effective_lag"] == 4
    assert np.allclose(obs.demand, hist["demand"][4])


def test_noise_zero_means_exact_observation():
    from planlab.core.engine import build_observation

    hist = {k: [np.full(2, 3.0) for _ in range(6)] for k in
            ("demand", "capacity", "inventory", "price", "backlog", "shortage")}
    hist["labour"] = [2.0] * 6
    obs = build_observation(
        t=3, lag=0, noise=0.0, rng=np.random.default_rng(0), hist=hist,
        observe_capacity=True, observe_inventory=True, observe_backlog=True,
        fallback_capacity=np.zeros(2), fallback_inventory=np.zeros(2),
        fallback_backlog=np.zeros(2), fallback_labour=1.0,
    )
    assert np.allclose(obs.demand, 3.0)
    assert np.isclose(obs.price[0], 3.0)


def test_unobserved_capacity_falls_back_to_baseline():
    from planlab.core.engine import build_observation

    hist = {k: [np.full(2, 9.0) for _ in range(5)] for k in
            ("demand", "capacity", "inventory", "price", "backlog", "shortage")}
    hist["labour"] = [1.0] * 5
    obs = build_observation(
        t=2, lag=0, noise=0.0, rng=np.random.default_rng(0), hist=hist,
        observe_capacity=False, observe_inventory=True, observe_backlog=True,
        fallback_capacity=np.full(2, 1.5), fallback_inventory=np.zeros(2),
        fallback_backlog=np.zeros(2), fallback_labour=1.0,
    )
    assert np.allclose(obs.capacity, 1.5)


def test_accounting_error_is_raised_on_a_leaky_engine(toy_tech):
    """A deliberately leaky mechanism must be caught, not silently tolerated."""
    from planlab.core.engine import Simulation
    from planlab.mechanisms.material_balance import MaterialBalance

    class Leaky(MaterialBalance):
        name = "leaky"

    sc = make_scenario("material_balance", id="leak", periods=4, burn_in=1)
    sim = Simulation(toy_tech, Leaky(toy_tech, {}), sc, seed=0)
    original = sim._reconcile

    def drop_one_good(*args, **kwargs):
        out = list(original(*args, **kwargs))
        out[5] = np.zeros_like(out[5])            # lose all end-of-period stocks
        return tuple(out)

    sim._reconcile = drop_one_good
    with pytest.raises(AccountingError):
        sim.run()


def test_lag_does_not_break_the_engine(toy_tech):
    sc = make_scenario("material_balance", id="lag6", periods=30, info={"lag": 6, "noise": 0.0})
    res = run_scenario(sc, technology=toy_tech)
    assert res.periods == 30
    assert np.all(np.isfinite(res.records[-1].x))


def test_burn_in_excludes_early_periods():
    from planlab.core.metrics import run_metrics

    sc = make_scenario("material_balance", id="burn", periods=20, burn_in=15)
    assert sc.burn_in == 15
    with pytest.raises(ValueError):
        scenario_from_dict({
            "id": "bad", "mechanism": "material_balance",
            "technology": {"toy": 6}, "periods": 10, "burn_in": 10,
        })
