"""Tests specific to the dual-track hybrid (``hybrid_dual_track``).

``theta`` must span the two pure institutions: at ``theta = 1`` the market track is empty
and nothing but the administered price exists, at ``theta = 0`` every final claimant is
served pro rata at the residual-market price.  The diagnostics the experiment layer
reports (plan-track share, market-track price dispersion) must be exposed.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import run_scenario
from planlab.mechanisms import get_mechanism

from conftest import make_scenario
from test_mech_contract import SHOCK, calm_state, observation


def _alloc(tech, theta, obs=None):
    mech = get_mechanism("hybrid_dual_track", tech, {"theta": theta})
    state = calm_state(tech)
    obs = observation(tech) if obs is None else obs
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    return mech, plan, alloc


def test_theta_one_is_a_pure_plan(toy_tech):
    mech, plan, alloc = _alloc(toy_tech, 1.0)
    assert alloc.meta["market_track_claims"] == 0.0
    assert alloc.meta["plan_track_claims"] > 0.0
    assert plan.meta["plan_track_share"] == pytest.approx(1.0)
    sc = make_scenario("hybrid_dual_track", id="dual_plan", periods=40, burn_in=5,
                       shocks=SHOCK, mechanism_params={"theta": 1.0})
    res = run_scenario(sc, technology=toy_tech)
    for r in res.records:
        assert np.array_equal(r.price, np.ones(toy_tech.n))
    assert res.run_metrics["mean_price_dispersion"] == 0.0


def test_theta_zero_is_a_pure_market(toy_tech):
    mech, plan, alloc = _alloc(toy_tech, 0.0)
    assert alloc.meta["plan_track_claims"] == 0.0
    assert alloc.meta["market_track_claims"] > 0.0
    # every *funded* final claimant sits in the one market tier: pro rata, no priorities
    n = toy_tech.n
    market_tier = alloc.tiers[n + 4]
    funded = alloc.claims.sum(axis=1) > 0.0
    final_funded = alloc.role_mask("final:") & funded
    assert final_funded.any()
    assert np.all(alloc.tiers[final_funded] == market_tier)
    sc = make_scenario("hybrid_dual_track", id="dual_mkt", periods=40, burn_in=5,
                       shocks=SHOCK, mechanism_params={"theta": 0.0})
    res = run_scenario(sc, technology=toy_tech)
    assert res.run_metrics["mean_price_dispersion"] > 0.1


def test_plan_track_rows_outrank_market_track_rows(toy_tech):
    _, _, alloc = _alloc(toy_tech, 0.5)
    n = toy_tech.n
    plan_tiers = alloc.tiers[n:n + 4]
    market_tiers = alloc.tiers[n + 4:n + 8]
    assert np.all(plan_tiers < np.min(market_tiers))


@pytest.mark.parametrize("theta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_theta_brackets_the_plan_and_market_outcomes(toy_tech, theta):
    """The endpoints must be the pure institutions, and the middle must lie between."""
    sc = make_scenario("hybrid_dual_track", id=f"dual_{theta}", periods=40, burn_in=5,
                       shocks=SHOCK, mechanism_params={"theta": theta})
    res = run_scenario(sc, technology=toy_tech)
    m = res.run_metrics
    if theta == 0.0:
        assert m["mean_price_dispersion"] > 0.1
        assert m["mean_price_volatility"] > 0.0
    elif theta == 1.0:
        assert m["mean_price_dispersion"] == 0.0
        assert m["mean_price_volatility"] == 0.0
    else:
        assert 0.0 < m["mean_price_dispersion"] < 0.5
    assert m["mean_shortage_rate"] < 0.15
    assert np.isfinite(m["mean_welfare"])


def test_theta_monotonically_shrinks_the_market_signal(toy_tech):
    disp = []
    for theta in (0.0, 0.5, 1.0):
        sc = make_scenario("hybrid_dual_track", id=f"duald_{theta}", periods=40, burn_in=5,
                           shocks=SHOCK, mechanism_params={"theta": theta})
        disp.append(run_scenario(sc, technology=toy_tech).run_metrics["mean_price_dispersion"])
    assert disp[0] > disp[1] > disp[2]


def test_plan_track_protects_essentials(toy_tech):
    """The pure plan track ranks essentials above other final demand; the pure market
    does not, so essentials must fare no worse under ``theta = 1``."""
    ess = {}
    for theta in (0.0, 1.0):
        sc = make_scenario("hybrid_dual_track", id=f"dualess_{theta}", periods=40,
                           burn_in=5, shocks=SHOCK, mechanism_params={"theta": theta})
        ess[theta] = run_scenario(sc, technology=toy_tech).run_metrics[
            "mean_shortage_essentials"]
    assert ess[1.0] <= ess[0.0]


def test_diagnostics_are_exposed_for_the_experiment_layer(toy_tech):
    _, plan, alloc = _alloc(toy_tech, 0.4)
    assert plan.meta["theta"] == pytest.approx(0.4)
    assert 0.0 <= plan.meta["plan_track_share"] <= 1.0
    assert plan.meta["market_price_dispersion"] >= 0.0
    assert alloc.meta["theta"] == pytest.approx(0.4)
    assert alloc.meta["plan_track_claims"] > 0.0
    assert alloc.meta["market_track_claims"] > 0.0
    assert "plan_track_share" in alloc.meta
    assert np.isfinite(alloc.meta["market_plan_price_ratio"])


def test_plan_track_share_is_theta_in_calm_conditions(toy_tech):
    for theta in (0.0, 0.25, 0.5, 1.0):
        _, plan, _ = _alloc(toy_tech, theta)
        assert plan.meta["plan_track_share"] == pytest.approx(theta, abs=1e-9)
