"""Contract tests every institution must satisfy to be comparable in PlanLab.

The scientific claim of the project is that the *institution* is a replaceable module
on a shared substrate, so mechanism comparisons are identified.  These tests are what
defends that claim: identical technology, identical observation, only the coordination
rule differs.  They are deliberately written against the public interface (the engine
and :class:`~planlab.mechanisms.base.CoordinationMechanism`), never against private
attributes, so that they keep working when an institution is rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import balance_residual, run_scenario
from planlab.core.state import Observation, State
from planlab.mechanisms import get_mechanism

from conftest import make_scenario

#: the four institutions delivered with this test suite
MECHANISMS = ["rationing_priority", "tatonnement", "market", "hybrid_dual_track"]

#: one 40% capacity shock to a single sector, the Lead's reference impulse
SHOCK = [{"type": "capacity", "sector": "s02", "start": 10, "duration": 6, "size": 0.6}]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def observation(tech, *, t: int = 6, lag: int = 0, price=None, shortage=None,
                demand=None, capacity=None, inventory=None, backlog=None,
                labour: float | None = None, gov=None, invest=None) -> Observation:
    """A calm, fully specified observation (what the engine hands a mechanism)."""
    n = tech.n
    if labour is None:  # the labour supply the base year actually needs
        labour = float(np.dot(tech.labor, tech.base_output))
    tf = np.asarray(tech.base_final, dtype=float)
    return Observation(
        t=t,
        lag=lag,
        demand=np.asarray(tf * 0.78 if demand is None else demand, dtype=float),
        capacity=np.asarray(tech.base_output if capacity is None else capacity, dtype=float),
        inventory=np.asarray(0.15 * tech.base_output if inventory is None else inventory,
                            dtype=float),
        price=np.asarray(np.ones(n) if price is None else price, dtype=float),
        backlog=np.asarray(np.zeros(n) if backlog is None else backlog, dtype=float),
        labour=float(labour),
        shortage=np.asarray(np.zeros(n) if shortage is None else shortage, dtype=float),
        gov=np.asarray(0.10 * tf if gov is None else gov, dtype=float),
        invest=np.asarray(0.12 * tf if invest is None else invest, dtype=float),
        meta={"observed_period": max(0, t - 1 - lag), "effective_lag": lag + 1},
    )


def calm_state(tech, *, stock_ratio: float = 0.15) -> State:
    return State.initial(tech, stock_ratio=stock_ratio)


# ---------------------------------------------------------------------------
# base year
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_calm_conditions_reproduce_base_year(toy_tech, mechanism):
    """No shock, no noise, no lag: the base year must be an exact fixed point."""
    sc = make_scenario(mechanism, id=f"calm_{mechanism}", periods=20, burn_in=5)
    res = run_scenario(sc, technology=toy_tech)
    last = res.records[-1]
    assert np.allclose(last.x, toy_tech.base_output, rtol=1e-6), mechanism
    assert res.run_metrics["mean_shortage_rate"] < 1e-9
    assert res.run_metrics["mean_plan_fulfilment"] > 0.999
    ratio = last.inventory_end / toy_tech.base_output
    assert np.allclose(ratio, sc.stock_ratio, atol=1e-6)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_calm_is_a_fixed_point_at_every_lag(toy_tech, mechanism):
    """Calm reproduction must not depend on the information regime."""
    for lag in (0, 2, 6):
        sc = make_scenario(mechanism, id=f"calmlag_{mechanism}_{lag}", periods=15,
                           burn_in=5, info={"lag": lag, "noise": 0.0})
        res = run_scenario(sc, technology=toy_tech)
        assert np.allclose(res.records[-1].x, toy_tech.base_output, rtol=1e-6)


# ---------------------------------------------------------------------------
# shock response: contains the impulse, keeps the books
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_capacity_shock_runs_clean_and_is_contained(toy_tech, mechanism):
    sc = make_scenario(mechanism, id=f"shock_{mechanism}", periods=45, shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    worst = max(balance_residual(r) for r in res.records)
    assert worst < 1e-8, f"balance identity violated by {worst:.3e}"
    for r in res.records:
        for arr in (r.x, r.Z, r.C, r.G, r.invest, r.inventory_end, r.demand, r.unmet,
                    r.price, r.plan_targets):
            assert np.all(np.isfinite(arr))
        assert np.all(r.x >= 0.0) and np.all(r.C >= 0.0) and np.all(r.unmet >= 0.0)
        assert np.all(r.price > 0.0)
    m = res.run_metrics
    assert m["mean_shortage_rate"] > 0.0, "the shock must bite"
    assert m["shock_peak_shortage"] > 0.05
    # contained disruption, not a self-reinforcing collapse
    assert m["mean_shortage_rate"] < 0.15, m["mean_shortage_rate"]
    tail = float(np.mean([r.metrics["shortage_rate"] for r in res.records[-10:]]))
    assert tail < 0.02, f"shortage did not recover (tail {tail:.3f})"


@pytest.mark.parametrize("mechanism", MECHANISMS)
@pytest.mark.parametrize("lag", [2, 4, 6])
def test_shock_with_long_lag_stays_finite(toy_tech, mechanism, lag):
    sc = make_scenario(
        mechanism, id=f"shocklag_{mechanism}_{lag}", periods=50, shocks=SHOCK,
        info={"lag": lag, "noise": 0.0},
        demand={"ar1": 0.3, "sigma": 0.02},
    )
    res = run_scenario(sc, technology=toy_tech)
    last = res.records[-1]
    assert np.all(np.isfinite(last.x)) and np.all(np.isfinite(last.price))
    assert max(balance_residual(r) for r in res.records) < 1e-8
    assert res.run_metrics["mean_shortage_rate"] < 0.30


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_noise_and_shocks_do_not_produce_nan(toy_tech, mechanism):
    sc = make_scenario(
        mechanism, id=f"nan_{mechanism}", periods=40,
        info={"lag": 3, "noise": 0.15},
        demand={"growth": 0.01, "ar1": 0.5, "sigma": 0.05},
        shocks=[
            {"type": "capacity", "sector": "s03", "start": 8, "duration": 10, "size": 0.5},
            {"type": "demand", "sector": "s01", "start": 15, "duration": 5, "size": 1.3},
        ],
    )
    res = run_scenario(sc, technology=toy_tech)
    for r in res.records:
        for arr in (r.x, r.Z, r.C, r.G, r.invest, r.inventory_end, r.demand, r.unmet,
                    r.price):
            assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# spoilage: what cannot be used or stored must be zero, and must stay bounded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_no_spoilage_in_calm_conditions(toy_tech, mechanism):
    """A mechanism whose claims are notional demand never over-produces into a glut."""
    sc = make_scenario(mechanism, id=f"calmspoil_{mechanism}", periods=25, burn_in=5)
    res = run_scenario(sc, technology=toy_tech)
    assert res.run_metrics["mean_spoilage_rate"] == 0.0
    assert res.run_metrics["mean_spoilage"] == 0.0
    for r in res.records:
        assert np.all(np.asarray(r.spoilage) == 0.0)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_no_spoilage_under_measurement_noise(toy_tech, mechanism):
    """The information channel alone must not create unsaleable output."""
    sc = make_scenario(
        mechanism, id=f"noisespoil_{mechanism}", periods=40, burn_in=20,
        info={"lag": 3, "noise": 0.06},
        demand={"ar1": 0.3, "sigma": 0.02},
    )
    res = run_scenario(sc, technology=toy_tech)
    assert res.run_metrics["mean_spoilage_rate"] < 1e-12
    for r in res.records:
        assert np.all(np.asarray(r.spoilage) <= 1e-12)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_spoilage_and_storage_stay_bounded_after_a_shock(toy_tech, mechanism):
    """Storage is finite (``max_stock_ratio``): nothing may hide in the inventory."""
    sc = make_scenario(mechanism, id=f"spoilshock_{mechanism}", periods=45, shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    cap = sc.max_stock_ratio * toy_tech.base_output
    for r in res.records:
        assert np.all(r.inventory_end <= cap + 1e-9)
        assert np.all(np.asarray(r.spoilage) >= -1e-12)
    # bounded, and small: the over-claim channel (backlog re-submission) is the only one
    assert res.run_metrics["mean_spoilage_rate"] < 0.01


# ---------------------------------------------------------------------------
# determinism / no leakage between replicates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_same_seed_same_trajectory(toy_tech, mechanism):
    sc = make_scenario(
        mechanism, id=f"det_{mechanism}", periods=20,
        info={"lag": 2, "noise": 0.1}, demand={"sigma": 0.05, "ar1": 0.3},
    )
    a = run_scenario(sc, technology=toy_tech, seed=7)
    b = run_scenario(sc, technology=toy_tech, seed=7)
    for ra, rb in zip(a.records, b.records):
        assert np.array_equal(ra.x, rb.x)
        assert np.array_equal(ra.price, rb.price)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_fresh_instance_per_registry_call(toy_tech, mechanism):
    """Mechanisms carry instruments across periods; sharing one would leak state."""
    a = get_mechanism(mechanism, toy_tech, {})
    b = get_mechanism(mechanism, toy_tech, {})
    assert a is not b
    assert np.array_equal(a.targets, b.targets)
    assert np.allclose(a.targets, toy_tech.base_output)


# ---------------------------------------------------------------------------
# state is read-only; policy uses the observation, not the truth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_mechanism_does_not_mutate_state(toy_tech, mechanism):
    mech = get_mechanism(mechanism, toy_tech, {})
    state = calm_state(toy_tech)
    before = state.copy()
    obs = observation(toy_tech)
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    mech.update_prices(obs, state, plan, alloc, realisation(toy_tech))
    for name in ("x", "inventory", "capital", "capacity", "price", "backlog",
                 "gov_demand"):
        assert np.array_equal(getattr(state, name), getattr(before, name)), name
    assert state.t == before.t


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_policy_uses_only_the_observation(toy_tech, mechanism):
    """Unobserved state (this period's truth) must not move the plan or the claims."""
    mech_a = get_mechanism(mechanism, toy_tech, {})
    mech_b = get_mechanism(mechanism, toy_tech, {})
    obs = observation(toy_tech)
    clean = calm_state(toy_tech)
    dirty = calm_state(toy_tech)
    # wreck every field the mechanism is *not* allowed to use for policy
    dirty.x = np.full(toy_tech.n, 3.0)
    dirty.inventory = np.full(toy_tech.n, 1e6)
    dirty.capacity = np.full(toy_tech.n, 1e-3)
    dirty.backlog = np.full(toy_tech.n, 1e3)
    dirty.gov_demand = np.full(toy_tech.n, 1e3)
    dirty.capital = np.full(toy_tech.n, 1e6)
    plan_a = mech_a.form_plan(obs, clean)
    plan_b = mech_b.form_plan(obs, dirty)
    assert np.allclose(plan_a.targets, plan_b.targets)
    supply = clean.inventory + plan_a.targets
    alloc_a = mech_a.allocate(obs, clean, plan_a, supply)
    alloc_b = mech_b.allocate(obs, dirty, plan_b, supply)
    assert np.allclose(alloc_a.claims, alloc_b.claims)
    assert np.allclose(alloc_a.tiers, alloc_b.tiers)


# ---------------------------------------------------------------------------
# plan / allocation contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_plan_and_allocation_invariants(toy_tech, mechanism):
    mech = get_mechanism(mechanism, toy_tech, {})
    state = calm_state(toy_tech)
    obs = observation(toy_tech)
    plan = mech.form_plan(obs, state)
    n = toy_tech.n
    assert plan.targets.shape == (n,) and plan.final_targets.shape == (n,)
    assert np.all(plan.targets >= 0.0) and np.all(plan.final_targets >= 0.0)
    assert isinstance(plan.meta, dict) and plan.meta.get("status") == "ok"
    assert plan.meta["effective_lag"] == obs.lag + 1

    supply = state.inventory + plan.targets
    alloc = mech.allocate(obs, state, plan, supply)
    m = alloc.claims.shape[0]
    assert alloc.claims.shape == (m, n) and m >= n
    assert alloc.tiers.shape == (m,)
    assert len(alloc.labels) == m
    assert np.all(alloc.claims >= 0.0)
    assert all(lab.startswith("inter:") for lab in alloc.labels[:n])
    assert any(lab.startswith("final:") for lab in alloc.labels)
    parts = alloc.split_grants(np.zeros_like(alloc.claims))
    assert set(parts) == {"intermediates", "final", "gov", "invest", "stock"}
    assert parts["intermediates"].shape == (n, n)
    # a shipped mechanism must not need the engine to scale its claims down in calm
    total_final = alloc.claims[alloc.role_mask("final:"), :].sum(axis=0)
    assert np.all(total_final <= toy_tech.base_final * 1.05 + 1e-9)


#: parameter dictionaries each institution must reject at construction
BAD_PARAMS = {
    "rationing_priority": [
        {"quota_speed": 0.0}, {"quota_speed": 1.5}, {"revision_interval": 0},
        {"capacity_ceiling": 0.0}, {"price_gain": -0.1}, {"backlog_carry": 2.0},
        {"min_target_ratio": 3.0, "max_target_ratio": 2.0},
        {"priority_sectors": "not_a_sector"},
    ],
    "tatonnement": [
        {"price_gain": -0.1}, {"price_gain": 3.0}, {"supply_gain": 2.0},
        {"relative_price_clip": 0.0}, {"claim_elasticity": -1.0},
    ],
    "market": [
        {"claim_elasticity": 0.0}, {"price_step": 0.0}, {"price_step": 1.5},
        {"clearing_tol": 0.0}, {"max_clearing_iters": 0}, {"price_floor": 5.0,
                                                          "price_ceiling": 1.0},
    ],
    "hybrid_dual_track": [
        {"theta": -0.1}, {"theta": 1.5}, {"market_gain": -1.0}, {"admin_gain": 3.0},
        {"price_floor": 5.0, "price_ceiling": 1.0},
    ],
}


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_invalid_parameters_raise(toy_tech, mechanism):
    """Every documented parameter range must be enforced at construction time."""
    for params in BAD_PARAMS[mechanism]:
        with pytest.raises(ValueError):
            get_mechanism(mechanism, toy_tech, params)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_valid_parameters_are_accepted_and_recorded(toy_tech, mechanism):
    valid = {
        "rationing_priority": {"quota_speed": 0.25, "revision_interval": 3},
        "tatonnement": {"price_gain": 0.05},
        "market": {"claim_elasticity": 1.25},
        "hybrid_dual_track": {"theta": 0.0},
    }[mechanism]
    mech = get_mechanism(mechanism, toy_tech, valid)
    for key, value in valid.items():
        assert float(mech.config[key]) == pytest.approx(float(value))


def realisation(tech):
    """A :class:`Realisation` snapshot for the price-update signature."""
    from planlab.core.state import Realisation

    n = tech.n
    x = np.asarray(tech.base_output, dtype=float)
    return Realisation(
        x=x,
        Z=np.asarray(tech.A, dtype=float) * x.reshape(1, -1),
        C=np.asarray(tech.base_final, dtype=float),
        G=np.zeros(n),
        invest=np.zeros(n),
        inventory_end=0.15 * x,
        demand=np.asarray(tech.base_final, dtype=float),
        unmet=np.zeros(n),
        supply=x.copy(),
        capacity=x.copy(),
    )
