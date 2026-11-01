"""Tests for the reference institution (``material_balance``) that make its
documented asymmetry *checkable* rather than argued.

The claim the paper needs (H1') is one-sidedness: a balancing round can only ever move
a target **down**, because ``core.feasibility.feasible_output`` maximises subject to
``0 <= x <= ub`` and therefore returns ``balanced <= ub``.  Combined with a proportional
revision that can push targets **up**, the two features interact asymmetrically: under
measurement noise the plan ratchets downwards and the revision cannot undo it.  Either
feature alone is harmless.  These tests are the executable version of that sentence.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import run_scenario
from planlab.mechanisms import get_mechanism

from conftest import make_scenario
from test_mech_contract import calm_state, observation


#: an observation that makes the plan infeasible for a *real* reason — a labour
#: supply 20% below the base year — so the balancing office has something to cut even
#: when the revision itself is the carried-over plan
def harsh_observation(tech, *, lag: int = 0, shortage: float = 0.4):
    return observation(
        tech,
        lag=lag,
        demand=0.86 * tech.base_final,
        capacity=0.75 * tech.base_output,
        inventory=0.05 * tech.base_output,
        backlog=0.15 * tech.base_final,
        labour=0.80 * float(np.dot(tech.labor, tech.base_output)),
        shortage=np.full(tech.n, shortage),
    )


@pytest.mark.parametrize("revision_gain", [0.0, 0.3, 1.0])
def test_balancing_rounds_can_only_revise_targets_downward(toy_tech, revision_gain):
    """``rounds > 0`` must never produce a target above the ``rounds = 0`` revision."""
    params = {"revision_gain": revision_gain}
    pure = get_mechanism("material_balance", toy_tech, {**params, "balance_rounds": 0})
    balanced = get_mechanism("material_balance", toy_tech, {**params, "balance_rounds": 3})
    state = calm_state(toy_tech)
    obs = harsh_observation(toy_tech)

    revision = pure.form_plan(obs, state).targets
    smoothed = balanced.form_plan(obs, state).targets

    assert np.all(smoothed <= revision + 1e-12), "a balancing round raised a target"
    assert np.any(smoothed < revision - 1e-9), "the hostile observation must bind somewhere"


def test_balancing_does_nothing_when_the_revision_is_already_feasible(toy_tech):
    """One-sidedness, other direction: no cut is applied if nothing is infeasible."""
    params = {"revision_gain": 0.1}
    pure = get_mechanism("material_balance", toy_tech, {**params, "balance_rounds": 0})
    balanced = get_mechanism("material_balance", toy_tech, {**params, "balance_rounds": 3})
    state = calm_state(toy_tech)
    obs = observation(toy_tech, shortage=np.zeros(toy_tech.n),
                      inventory=2.0 * toy_tech.base_output)
    a = pure.form_plan(obs, state).targets
    b = balanced.form_plan(obs, state).targets
    assert np.allclose(a, b, rtol=1e-9)


def test_rounds_zero_is_the_pure_proportional_revision(toy_tech):
    """With zero rounds the plan is exactly ``T(1 + eta s)``, clipped — as documented."""
    eta, clip, max_ratio, min_ratio = 0.3, 0.5, 2.5, 0.25
    mech = get_mechanism(
        "material_balance", toy_tech,
        {"revision_gain": eta, "revision_clip": clip, "balance_rounds": 0},
    )
    state = calm_state(toy_tech)
    shortage = np.array([0.9, 0.0, 0.2, 1.0, 0.05, 0.4])
    capacity = 0.8 * toy_tech.base_output
    obs = observation(toy_tech, shortage=shortage, capacity=capacity)
    got = mech.form_plan(obs, state).targets

    carried = toy_tech.base_output.copy()
    step = np.clip(eta * shortage, -clip, clip)
    expected = carried * (1.0 + step)
    expected = np.minimum(expected, max_ratio * carried)
    expected = np.maximum(expected, min_ratio * carried)
    expected = np.minimum(expected, np.maximum(capacity, 0.0) * 1.5 + 1e-12)
    assert np.allclose(got, expected, rtol=1e-12)


def _run(tech, gain, rounds, *, noise=0.03, periods=60, burn_in=20):
    sc = make_scenario(
        "material_balance", id=f"ratchet_{gain}_{rounds}", periods=periods,
        burn_in=burn_in, info={"lag": 0, "noise": noise},
        demand={"ar1": 0.3, "sigma": 0.02},
        mechanism_params={"revision_gain": gain, "balance_rounds": rounds},
    )
    res = run_scenario(sc, technology=tech)
    xs = np.array([r.x for r in res.records])
    ratio = float(np.mean(xs[burn_in:] / tech.base_output[None, :]))
    return res, ratio


def test_the_downward_ratchet_needs_both_the_gain_and_the_rounds(toy_tech):
    """The paper's H1' interaction, made checkable.

    Under measurement noise, a proportional revision with no balancing, and balancing
    with no revision, both cost (almost) nothing.  Together they ratchet the plan down:
    the revision answers noise by pushing targets up, the balancing office can only cut
    them, so the plan loses ground it never regains.
    """
    res_gain, ratio_gain = _run(toy_tech, 1.0, 0)
    res_rounds, ratio_rounds = _run(toy_tech, 0.0, 3)
    res_both, ratio_both = _run(toy_tech, 1.0, 3)

    # each feature alone is harmless
    assert ratio_gain > 0.99
    assert ratio_rounds > 0.99
    # together they cost output, shortage and (bounded) spoilage
    assert ratio_both < 0.93, ratio_both
    assert ratio_both < ratio_gain - 0.05
    assert ratio_both < ratio_rounds - 0.05
    assert res_both.run_metrics["mean_shortage_rate"] > res_gain.run_metrics[
        "mean_shortage_rate"]
    assert res_both.run_metrics["mean_spoilage_rate"] > 0.0
    assert res_gain.run_metrics["mean_spoilage_rate"] == 0.0
    assert res_rounds.run_metrics["mean_spoilage_rate"] == 0.0


def test_the_ratchet_is_monotone_in_the_number_of_rounds(toy_tech):
    """More balancing effort against the same noisy revision costs more output."""
    ratios = [_run(toy_tech, 1.0, rounds)[1] for rounds in (0, 1, 3)]
    assert ratios[0] > ratios[1] > ratios[2], ratios
