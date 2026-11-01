"""Tests specific to the competitive market (``market``).

"Clearing" under a Leontief technology is the final-use flow market (intermediates are
technically determined), so the tests pin down three things: the auctioneer really
iterates when the flow is scarce and leaves the price alone when it is not, claims are
the price-mediated demand at the posted price (never more than notional demand), and the
posted price stays inside the band the mechanism claims to honour.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import run_scenario
from planlab.mechanisms import get_mechanism

from conftest import make_scenario
from test_mech_contract import SHOCK, calm_state, observation


def allocation_at(tech, obs, mechanism="market", params=None):
    mech = get_mechanism(mechanism, tech, params or {})
    state = calm_state(tech)
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    return mech, plan, alloc


def test_calm_market_clears_without_moving_the_price(toy_tech):
    obs = observation(toy_tech)
    mech, plan, alloc = allocation_at(toy_tech, obs)
    assert plan.meta["clearing_iters"] == 1.0
    assert plan.meta["clearing_gap"] < 1e-12
    assert np.allclose(mech.prices, np.ones(toy_tech.n))


def test_auctioneer_iterates_when_the_flow_is_scarce(toy_tech):
    """A binding supply must trigger real within-period iteration."""
    scarce = observation(toy_tech, capacity=0.5 * toy_tech.base_output)
    mech, plan, alloc = allocation_at(toy_tech, scarce)
    assert plan.meta["clearing_iters"] > 1.0
    assert plan.meta["clearing_gap"] <= mech.config["clearing_tol"] * 1.5
    assert np.all(mech.prices > 1.0)


def test_claims_are_price_mediated_and_never_above_notional_demand(toy_tech):
    calm_obs = observation(toy_tech)
    scarce_obs = observation(toy_tech, capacity=0.5 * toy_tech.base_output)
    _, _, calm_alloc = allocation_at(toy_tech, calm_obs)
    mech, _, scarce_alloc = allocation_at(toy_tech, scarce_obs)
    notional = calm_obs.demand
    mask = calm_alloc.role_mask("final:")
    calm_claim = calm_alloc.claims[mask, :].sum(axis=0)
    scarce_claim = scarce_alloc.claims[mask, :].sum(axis=0)
    assert np.allclose(calm_claim, notional, rtol=1e-9)
    assert np.all(scarce_claim <= notional + 1e-9), "no claimant may bid above notional demand"
    assert np.all(scarce_claim < 0.99 * notional), "the price must ration demand"
    assert np.all(mech.prices > 1.0)


def test_posted_price_respects_the_documented_band(toy_tech):
    """A nearly exhausted market must post its ceiling, not an unbounded price."""
    mech = get_mechanism("market", toy_tech, {"price_floor": 0.05, "price_ceiling": 20.0})
    state = calm_state(toy_tech)
    obs = observation(toy_tech, capacity=1e-6 * toy_tech.base_output)
    plan = mech.form_plan(obs, state)
    assert np.all(mech.prices <= 20.0 + 1e-12)
    assert np.all(mech.prices >= 0.05 - 1e-12)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    assert np.all(np.isfinite(alloc.claims))


def test_engine_run_keeps_prices_inside_the_band(toy_tech):
    sc = make_scenario("market", id="mkt_band", periods=40, burn_in=5, shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    prices = np.array([r.price for r in res.records])
    assert np.all(prices >= 0.05) and np.all(prices <= 20.0)
    assert np.all(np.isfinite(prices))


def test_market_recovers_after_the_shock(toy_tech):
    sc = make_scenario("market", id="mkt_recover", periods=45, burn_in=5, shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    tail = float(np.mean([r.metrics["shortage_rate"] for r in res.records[-8:]]))
    assert tail < 0.02
    last_prices = np.array([np.mean(r.price) for r in res.records[-5:]])
    assert np.all(np.abs(np.log(last_prices)) < 0.10)


def test_market_is_faster_to_react_than_a_frozen_price_economy(toy_tech):
    """The price mechanism must move the shock-period price, not just the quantities."""
    sc = make_scenario("market", id="mkt_react", periods=20, burn_in=5, shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    during = np.array([np.max(r.price) for r in res.records[10:15]])
    assert np.all(during > 1.0)


def test_clearing_tolerance_and_iteration_cap_are_settable(toy_tech):
    loose = get_mechanism("market", toy_tech, {"clearing_tol": 0.2, "max_clearing_iters": 3})
    scarce = observation(toy_tech, capacity=0.5 * toy_tech.base_output)
    state = calm_state(toy_tech)
    plan = loose.form_plan(scarce, state)
    assert plan.meta["clearing_iters"] <= 3.0
