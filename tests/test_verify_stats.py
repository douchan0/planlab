"""Tests for the verifier's independent statistics (check 5) and accounting helpers
(check 3).  No dependency on results/; everything is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from verification.check5_stats import (bootstrap_ci_mean, cliffs_delta, holm,
                                       spearman_permutation, spearman_rho,
                                       wilcoxon_signed_rank)


def test_wilcoxon_exact_matches_scipy():
    from scipy import stats

    rng = np.random.default_rng(3)
    for _ in range(10):
        d = rng.standard_normal(12) * 0.4 + 0.2
        d[np.abs(d) < 0.02] += 0.1
        mine = wilcoxon_signed_rank(d)["p_exact"]
        ref = stats.wilcoxon(d, zero_method="wilcox", method="exact").pvalue
        assert mine == pytest.approx(ref, abs=1e-12)


def test_wilcoxon_all_zero_pairs():
    out = wilcoxon_signed_rank(np.zeros(10))
    assert out["n"] == 0
    assert np.isnan(out["statistic"])


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222])
    assert np.allclose(holm(p), multipletests(p, method="holm")[1], atol=1e-12)


def test_holm_monotone_and_bounded():
    p = np.array([0.5, 0.001, 0.2, np.nan, 0.04])
    adjusted = holm(p)
    assert np.isnan(adjusted[3])
    finite = adjusted[np.isfinite(adjusted)]
    assert np.all(finite >= 0.0) and np.all(finite <= 1.0)
    assert adjusted[1] <= adjusted[4] <= adjusted[2] <= adjusted[0] + 1e-15


def test_spearman_matches_scipy_with_ties():
    from scipy import stats

    rng = np.random.default_rng(11)
    x = np.round(rng.standard_normal(30), 1)
    y = np.round(0.5 * x + rng.standard_normal(30), 1)
    assert spearman_rho(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)


def test_permutation_p_matches_enumeration():
    from itertools import permutations

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.5, 1.0, 1.2, 2.4, 2.1])
    rho = spearman_rho(x, y)
    hits = sum(abs(spearman_rho(x, y[list(p)])) >= abs(rho) - 1e-15
               for p in permutations(range(y.size)))
    exact = hits / 120.0
    got = spearman_permutation(x, y, n_perm=20_000, seed=5)["p_perm"]
    assert got == pytest.approx(exact, abs=0.02)


def test_bootstrap_deterministic_and_stable():
    rng = np.random.default_rng(2)
    values = rng.standard_normal(25)
    a = bootstrap_ci_mean(values, seed=123)
    b = bootstrap_ci_mean(values, seed=123)
    c = bootstrap_ci_mean(values, seed=124)
    assert a == b
    width = a[1] - a[0]
    assert abs(c[0] - a[0]) < 0.2 * width and abs(c[1] - a[1]) < 0.2 * width


def test_cliffs_delta_extremes():
    assert cliffs_delta([3, 4, 5], [1, 2]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2], [3, 4, 5]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check 3 helpers
# ---------------------------------------------------------------------------
def test_metric_definitions_against_hand_example():
    from verification.check3_accounting import _plan_fulfilment, _shortage_rate, _welfare

    demand = np.array([10.0, 20.0])
    unmet = np.array([1.0, 3.0])
    assert _shortage_rate(demand, unmet) == pytest.approx(4.0 / 30.0)
    x = np.array([10.0, 10.0])
    targets = np.array([10.0, 20.0])
    assert _plan_fulfilment(x, targets) == pytest.approx(0.75)
    C = np.array([1.0, 2.0])
    base_final = np.array([100.0, 50.0])
    weights = np.array([1.2, 1.0])
    c0 = 1e-3 * base_final
    expected = 1.2 * np.log(1.0 + c0[0]) + 1.0 * np.log(2.0 + c0[1])
    assert _welfare(base_final, weights, C) == pytest.approx(expected)


def test_recover_recorded_A_handles_flat_padding():
    from verification.check3_accounting import recover_recorded_A

    A6 = np.arange(36, dtype=float).reshape(6, 6) / 100.0
    padded = np.full((20, 20), np.nan)
    padded.ravel()[:36] = A6.ravel()          # the runner's flat-fill layout
    rec = {"A": padded, "n_sectors": np.int64(6)}
    recovered, was_flat = recover_recorded_A(rec)
    assert was_flat
    assert np.allclose(recovered, A6)


def test_ladder_violation_detector():
    from verification.check3_accounting import _ladder_violations

    claims = np.array([[1.0], [1.0], [1.0], [1.0]])
    tiers = np.array([0.0, 0.0, 1.0, 2.0])
    # good ladder: tier 0 capped => later tiers get nothing
    grants = np.array([[0.3], [0.3], [0.0], [0.0]])
    assert _ladder_violations(grants, claims, tiers) == []
    # violation: a later tier is served after tier 0 was capped
    grants_bad = np.array([[0.3], [0.3], [0.2], [0.0]])
    assert _ladder_violations(grants_bad, claims, tiers) == [0]


def test_cell_tech_prefers_the_authoritative_spec():
    from verification._rerun import cell_tech_spec

    cell = {"tech": {"toy": 6},
            "scenario": {"technology": {"source": "eurostat", "n_sectors": 20}}}
    assert cell_tech_spec(cell) == {"toy": 6}
    assert cell_tech_spec({"scenario": {"technology": {"source": "eurostat"}}}) == {
        "source": "eurostat"}
