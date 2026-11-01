"""Tests specific to the Lange price board (``tatonnement``).

The price rule is the object of Proposition 1, so it is tested as a *specification*
and not merely as behaviour: the log price must move by exactly ``price_gain`` times the
shortage rate that the observation carries, with no extra period of lag and with no use
of current-period truth.  A regression here would silently invalidate the
theory-to-simulation comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import run_scenario
from planlab.mechanisms import get_mechanism
from planlab.mechanisms.tatonnement import Tatonnement

from conftest import make_scenario
from test_mech_contract import SHOCK, calm_state, observation, realisation


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------
def test_price_rule_is_exactly_the_proposition_one_rule(toy_tech):
    """log p_{t+1} = log p_t + price_gain * shortage_observed, and nothing else."""
    gain = 0.37
    mech = get_mechanism("tatonnement", toy_tech, {"price_gain": gain})
    state = calm_state(toy_tech)
    state.price = np.array([1.20, 0.80, 1.00, 1.50, 0.90, 1.10])
    shortage = np.array([0.50, 0.00, 0.25, 1.00, 0.10, 0.00])
    obs = observation(toy_tech, shortage=shortage, price=state.price)
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    got = mech.update_prices(obs, state, plan, alloc, realisation(toy_tech))
    expected = state.price * np.exp(gain * shortage)
    assert np.allclose(got, expected, rtol=1e-13)


def test_gain_is_a_clean_scalar(toy_tech):
    """Doubling ``price_gain`` must exactly double the log-price move."""
    state = calm_state(toy_tech)
    shortage = np.full(toy_tech.n, 0.2)
    obs = observation(toy_tech, shortage=shortage)
    moves = {}
    for gain in (0.1, 0.2, 0.4):
        mech = get_mechanism("tatonnement", toy_tech, {"price_gain": gain})
        plan = mech.form_plan(obs, state)
        alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
        p = mech.update_prices(obs, state, plan, alloc, realisation(toy_tech))
        moves[gain] = np.log(p) - np.log(state.price)
    assert np.allclose(moves[0.2], 2.0 * moves[0.1], rtol=1e-12)
    assert np.allclose(moves[0.4], 4.0 * moves[0.1], rtol=1e-12)


def test_price_gain_zero_freezes_prices(toy_tech):
    """``price_gain = 0`` must be exactly a fixed-administered-price economy."""
    sc = make_scenario("tatonnement", id="taton_frozen", periods=30, burn_in=5,
                       shocks=SHOCK, mechanism_params={"price_gain": 0.0})
    res = run_scenario(sc, technology=toy_tech)
    for r in res.records:
        assert np.array_equal(r.price, np.ones(toy_tech.n))
    assert res.run_metrics["mean_price_dispersion"] == 0.0
    assert res.run_metrics["mean_price_volatility"] == 0.0


def test_no_extra_lag_and_no_current_period_truth(toy_tech):
    """The update may depend on ``obs.shortage`` only — not on t, lag or the truth."""
    state = calm_state(toy_tech)
    shortage = np.array([0.4, 0.1, 0.0, 0.3, 0.0, 0.9])
    obs_a = observation(toy_tech, t=4, lag=0, shortage=shortage)
    obs_b = observation(toy_tech, t=40, lag=6, shortage=shortage)
    # a realisation that contradicts the observation in every field
    lie = realisation(toy_tech)
    from planlab.core.state import Realisation
    lie = Realisation(
        x=np.zeros(toy_tech.n), Z=np.zeros((toy_tech.n, toy_tech.n)),
        C=np.zeros(toy_tech.n), G=np.zeros(toy_tech.n), invest=np.zeros(toy_tech.n),
        inventory_end=np.zeros(toy_tech.n), demand=np.full(toy_tech.n, 1e6),
        unmet=np.full(toy_tech.n, 1e6), supply=np.full(toy_tech.n, 1e6),
        capacity=np.zeros(toy_tech.n),
    )
    prices = []
    for obs in (obs_a, obs_b):
        mech = get_mechanism("tatonnement", toy_tech, {"price_gain": 0.2})
        plan = mech.form_plan(obs, state)
        alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
        prices.append(mech.update_prices(obs, state, plan, alloc, lie))
    assert np.array_equal(prices[0], prices[1])
    assert np.allclose(prices[0], np.exp(0.2 * shortage))


def test_the_shortage_the_board_sees_is_the_reported_one(toy_tech):
    """The signal must be the realised shortage of ``t - 1 - lag`` (no extra lag)."""
    seen: list[np.ndarray] = []

    class Spy(Tatonnement):
        name = "tatonnement_spy"

        def form_plan(self, obs, state):
            seen.append(np.asarray(obs.shortage, dtype=float).copy())
            return super().form_plan(obs, state)

    lag = 2
    sc = make_scenario("tatonnement", id="taton_spy", periods=22, burn_in=5, shocks=SHOCK,
                       info={"lag": lag, "noise": 0.0})
    from planlab.core.engine import Simulation

    sim = Simulation(toy_tech, Spy(toy_tech, {}), sc, seed=0)
    res = sim.run()
    assert len(seen) == sc.periods
    for t, signal in enumerate(seen, start=1):
        src = max(0, t - 1 - lag)
        record = res.records[src - 1] if src >= 1 else None
        if record is None:
            expected = np.zeros(toy_tech.n)
        else:
            expected = np.where(record.demand > 0,
                                record.unmet / np.maximum(record.demand, 1e-300), 0.0)
        assert np.allclose(signal, expected, atol=1e-12), f"t={t}"


def test_fallback_without_a_shortage_vector(toy_tech):
    """Defensive path: an observation without ``shortage`` must not break the board."""
    mech = get_mechanism("tatonnement", toy_tech, {"price_gain": 0.2})
    state = calm_state(toy_tech)
    obs = observation(toy_tech, shortage=None)
    # force the None branch (the constructor of Observation stores the array we pass)
    obs = type(obs)(**{**obs.__dict__, "shortage": None})
    plan = mech.form_plan(obs, state)
    alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
    p = mech.update_prices(obs, state, plan, alloc, realisation(toy_tech))
    assert p.shape == (toy_tech.n,)
    assert np.all(np.isfinite(p)) and np.all(p > 0.0)


# ---------------------------------------------------------------------------
# mechanism-specific behaviour
# ---------------------------------------------------------------------------
def test_investment_claims_are_price_mediated(toy_tech):
    """Dear goods are bought for capital formation in smaller quantities."""
    mech = get_mechanism("tatonnement", toy_tech, {})
    state = calm_state(toy_tech)
    flat = np.ones(toy_tech.n)
    skewed = np.ones(toy_tech.n)
    skewed[0] = 2.0
    claims = []
    for price in (flat, skewed):
        obs = observation(toy_tech, price=price)
        mech = get_mechanism("tatonnement", toy_tech, {})
        plan = mech.form_plan(obs, state)
        alloc = mech.allocate(obs, state, plan, state.inventory + plan.targets)
        row = alloc.claims[alloc.role_mask("invest:"), :].sum(axis=0)
        claims.append(row)
    assert claims[1][0] < 0.9 * claims[0][0], "the dear good must be bought less"
    assert np.all(np.isfinite(claims[1])) and np.all(claims[1] > 0.0)


def test_shocked_sector_price_rises_most(toy_tech):
    """A capacity shock must show up where it happened: in that good's price."""
    sc = make_scenario("tatonnement", id="taton_price", periods=30, burn_in=5,
                       shocks=SHOCK)
    res = run_scenario(sc, technology=toy_tech)
    during = res.records[13]          # inside the shock window
    other = np.delete(during.price, 1)
    assert during.price[1] > 1.2
    assert during.price[1] > np.max(other)


def test_higher_gain_gives_larger_price_swings(toy_tech):
    """Sanity on the treatment parameter: the knob does something monotone."""
    amps = []
    for gain in (0.05, 0.2, 0.5):
        sc = make_scenario("tatonnement", id=f"taton_amp_{gain}", periods=40, burn_in=5,
                           shocks=SHOCK, mechanism_params={"price_gain": gain})
        amps.append(run_scenario(sc, technology=toy_tech).run_metrics["oscillation_amplitude"])
    assert amps[0] < amps[1] < amps[2]
