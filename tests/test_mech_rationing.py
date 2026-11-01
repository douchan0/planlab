"""Tests specific to administrative quota rationing (``rationing_priority``).

The institution's identifying features are the *hard quota* (an approved document, not a
resource balance), the strict priority ladder, and administered prices that do not move.
``revision_interval`` is the lag-sensitivity treatment: a plan document fixed for several
quarters carries a stale approved capacity, so a longer interval must cost more after a
capacity shock.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import ration_by_priority, run_scenario
from planlab.mechanisms import get_mechanism

from conftest import make_scenario
from test_mech_contract import SHOCK, calm_state, observation


def _plan(tech, params=None, obs=None):
    mech = get_mechanism("rationing_priority", tech, params or {})
    state = calm_state(tech)
    obs = observation(tech) if obs is None else obs
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    return mech, plan, alloc


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def test_ladder_order_is_strict(toy_tech):
    _, plan, alloc = _plan(toy_tech)
    n = toy_tech.n
    priority = np.asarray(toy_tech.essentials, dtype=float) > 0.5
    tier_inter_priority = alloc.tiers[:n][priority].min()
    tier_inter_other = alloc.tiers[:n][~priority].min()
    assert tier_inter_priority < tier_inter_other
    tier_essential = alloc.tiers[alloc.labels.index("final:essential")].min()
    tier_other = alloc.tiers[alloc.labels.index("final:other")].min()
    tier_gov = alloc.tiers[alloc.labels.index("gov:00")].min()
    tier_invest = alloc.tiers[alloc.labels.index("invest:00")].min()
    assert tier_inter_other < tier_essential < tier_other < tier_gov < tier_invest


def test_engine_rationing_respects_the_ladder_when_supply_binds(toy_tech):
    """A lower tier never receives a smaller fill ratio than a higher tier."""
    _, plan, alloc = _plan(toy_tech)
    supply = 0.45 * (calm_state(toy_tech).inventory + plan.targets)
    grants = ration_by_priority(supply, alloc.claims, alloc.tiers)
    claimed = alloc.claims.sum(axis=0)
    filled = grants.sum(axis=0)
    assert np.all(filled <= supply + 1e-9)
    # per good, per tier: ratio must be non-increasing in the tier level
    for good in range(toy_tech.n):
        order = np.argsort(alloc.tiers, kind="stable")
        ratios = []
        for row in order:
            claim = alloc.claims[row, good]
            if claim <= 1e-12:
                continue
            ratios.append((alloc.tiers[row], grants[row, good] / claim))
        for tier_lo, r_lo in ratios:
            for tier_hi, r_hi in ratios:
                if tier_lo < tier_hi:
                    assert r_lo >= r_hi - 1e-9, (good, tier_lo, tier_hi)
    assert np.all(claimed > supply)


# ---------------------------------------------------------------------------
# the quota document
# ---------------------------------------------------------------------------
def test_quota_follows_observed_needs(toy_tech):
    obs = observation(toy_tech, demand=1.30 * observation(toy_tech).demand)
    _, plan, _ = _plan(toy_tech, {"capacity_ceiling": 2.0}, obs)
    assert np.any(plan.targets > 1.02 * toy_tech.base_output)


def test_quota_is_capped_by_the_approved_capacity(toy_tech):
    """``capacity_ceiling = 1`` means the quota cannot exceed approved capacity."""
    obs = observation(toy_tech, demand=1.30 * observation(toy_tech).demand)
    _, tight, _ = _plan(toy_tech, {"capacity_ceiling": 1.0}, obs)
    _, loose, _ = _plan(toy_tech, {"capacity_ceiling": 2.0}, obs)
    assert np.all(tight.targets <= toy_tech.base_output + 1e-9)
    assert np.any(loose.targets > tight.targets + 1e-9)


def test_revision_interval_controls_how_often_the_document_changes(toy_tech):
    mech = get_mechanism("rationing_priority", toy_tech, {"revision_interval": 3})
    state = calm_state(toy_tech)
    obs = observation(toy_tech)
    revised = []
    for _ in range(9):
        plan = mech.form_plan(obs, state)
        revised.append(plan.meta["revised"])
    assert revised == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_longer_revision_interval_costs_more_after_a_capacity_shock(toy_tech):
    means = {}
    for interval in (1, 4, 8):
        sc = make_scenario("rationing_priority", id=f"quota_iv{interval}", periods=45,
                           burn_in=5, shocks=SHOCK,
                           mechanism_params={"revision_interval": interval})
        means[interval] = run_scenario(sc, technology=toy_tech).run_metrics[
            "mean_shortage_rate"]
    assert means[1] < means[4] < means[8], means


def test_administered_prices_do_not_move(toy_tech):
    sc = make_scenario("rationing_priority", id="quota_prices", periods=30, burn_in=5,
                       shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    for r in res.records:
        assert np.array_equal(r.price, np.ones(toy_tech.n))


def test_administered_price_revision_can_be_switched_on(toy_tech):
    sc = make_scenario("rationing_priority", id="quota_prices_on", periods=30, burn_in=5,
                       shocks=SHOCK, mechanism_params={"price_gain": 0.2})
    res = run_scenario(sc, technology=toy_tech)
    assert max(np.max(r.price) for r in res.records) > 1.0


def test_priority_sector_selection_is_configurable(toy_tech):
    _, default_plan, default_alloc = _plan(toy_tech)
    mech = get_mechanism("rationing_priority", toy_tech, {"priority_sectors": "all"})
    state = calm_state(toy_tech)
    obs = observation(toy_tech)
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    assert np.allclose(alloc.tiers[:toy_tech.n], alloc.tiers[:toy_tech.n][0])
    # "essentials" must split the intermediate block in two tiers on this substrate
    assert len(np.unique(default_alloc.tiers[:toy_tech.n])) == 2
