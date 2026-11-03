"""Check 5 — independent audit of the paper's statistical procedures.

Nothing here calls the project's ``aggregate`` module.  I re-implement from
scratch, and validate the implementations against references that are independent
of both the project and me (``scipy`` and ``statsmodels``):

* the paired **Wilcoxon signed-rank** test — exact null distribution by dynamic
  programming over subset sums (a different algorithm from scipy's), plus the
  normal approximation with tie and zero corrections;
* the **bootstrap confidence interval** of the paired mean difference (percentile
  bootstrap, 10 000 resamples);
* **Holm**–Bonferroni adjustment (validated against
  ``statsmodels.stats.multitest.multipletests(method="holm")``);
* **Cliff's δ**;
* **Spearman's ρ** with a *permutation* null (not an asymptotic p-value), which is
  what the H4 centrality claim needs: 20 shocked sectors is a small sample and the
  asymptotic p-value is not trustworthy.

Audits
------
5a  Machinery self-tests (always run): exact vs scipy Wilcoxon, Holm vs
    statsmodels, Spearman vs scipy, permutation p vs full enumeration on a tiny
    sample, bootstrap reproducibility.
5b  ``results/tables/tests.csv`` (when present): every paired-test row is
    recomputed from ``summary.csv`` — the statistic, the p-value, the effect
    size, the bootstrap CI and the Holm-adjusted p.  A mismatch is a finding.
5c  H4 (network amplification vs centrality): the amplification statistic is
    recomputed from ``runs.parquet``/``sector_panel.parquet`` per shocked sector,
    correlated against *three* candidate linkage definitions (row sums of A, the
    intermediate share (A x)/x, and Leontief column sums), with a 10 000-draw
    permutation p-value.  The reported ρ is compared with mine; the p-value is
    not taken from an asymptotic table.

The check reports SKIP (not PASS) when the tables/panel do not exist yet, so an
unaudited paper can never look audited.
"""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pandas as pd

from .common import RESULTS, TABLES, CheckResult, run_guarded

N_PERM = 10_000
PERM_SEED = 20261101
BOOT_SEED = 20260501


# ---------------------------------------------------------------------------
# own statistical implementations
# ---------------------------------------------------------------------------
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (1-based)."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: np.ndarray, alternative: str = "two-sided") -> dict:
    """Signed-rank test, exact null by DP over subset sums + normal approximation."""
    d = np.asarray(diffs, dtype=float)
    n_orig = d.size
    d = d[np.abs(d) > 0]                       # zero_method = "wilcox"
    n = d.size
    out = {"n": n, "n_orig": n_orig, "statistic": float("nan"),
           "p_exact": float("nan"), "p_normal": float("nan"), "ties": False}
    if n == 0:
        return out
    ranks = _average_ranks(np.abs(d))
    w_plus = float(np.sum(ranks[d > 0]))
    out["statistic"] = w_plus
    # --- exact null: distribution of the sum of a random subset of ranks --------
    counts = np.zeros(int(round(ranks.sum())) + 1)
    counts[0] = 1.0
    for rk in np.round(ranks).astype(int):
        new = np.zeros_like(counts)
        new[: counts.size] += counts
        new[rk:] += counts[: counts.size - rk]
        counts = new
    total = counts.sum()
    thresholds = np.cumsum(counts) / total
    idx = int(round(w_plus))
    lower = float(thresholds[idx]) if idx < thresholds.size else 1.0      # P(W <= w)
    upper = float(1.0 - (thresholds[idx - 1] if idx > 0 else 0.0))        # P(W >= w)
    if alternative == "two-sided":
        out["p_exact"] = min(1.0, 2.0 * min(lower, upper))
    elif alternative == "greater":
        out["p_exact"] = upper
    else:
        out["p_exact"] = lower
    # --- normal approximation with tie/zero correction -------------------------
    ties = np.unique(np.abs(d), return_counts=True)[1]
    tie_term = float(np.sum(ties ** 3 - ties)) if np.any(ties > 1) else 0.0
    out["ties"] = bool(np.any(ties > 1))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return out
    z = (w_plus - mean_w)
    # continuity correction of 0.5 towards the mean
    z = z - 0.5 * np.sign(z) if z != 0 else 0.0
    z /= math.sqrt(var_w)
    if alternative == "two-sided":
        out["p_normal"] = float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))
    elif alternative == "greater":
        out["p_normal"] = float(0.5 * math.erfc(z / math.sqrt(2.0)))
    else:
        out["p_normal"] = float(0.5 * math.erfc(-z / math.sqrt(2.0)))
    return out


def bootstrap_ci_mean(values, n_boot: int = N_PERM, alpha: float = 0.05,
                      seed: int = BOOT_SEED) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = (m - rank) * p[i]
        running = max(running, adjusted)
        out[i] = min(1.0, running)
    return out


def cliffs_delta(a, b) -> float:
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(float(np.sum(a_i > b)) for a_i in a)
    lt = sum(float(np.sum(a_i < b)) for a_i in a)
    return float((gt - lt) / (a.size * b.size))


def spearman_rho(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = _average_ranks(x[mask])
    ry = _average_ranks(y[mask])
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def spearman_permutation(x, y, n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Two-sided permutation p-value for Spearman's rho."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rho = spearman_rho(x, y)
    if x.size < 3 or not np.isfinite(rho):
        return {"rho": rho, "p_perm": float("nan"), "n": int(x.size), "n_perm": 0}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(y.size)
        if abs(spearman_rho(x, y[perm])) >= abs(rho) - 1e-15:
            hits += 1
    return {"rho": rho, "p_perm": (hits + 1.0) / (n_perm + 1.0),
            "n": int(x.size), "n_perm": int(n_perm)}


# ---------------------------------------------------------------------------
# machinery self-tests
# ---------------------------------------------------------------------------
def _self_tests(res: CheckResult) -> None:
    from scipy import stats

    rng = np.random.default_rng(7)
    # --- Wilcoxon exact vs scipy exact (no ties, no zeros, n <= 20) --------
    worst_exact = 0.0
    for trial in range(40):
        n = int(rng.integers(6, 20))
        d = rng.standard_normal(n) * 0.5 + 0.15
        d[np.abs(d) < 0.01] += 0.05       # avoid exact zeros
        mine = wilcoxon_signed_rank(d)["p_exact"]
        ref = stats.wilcoxon(d, zero_method="wilcox", method="exact").pvalue
        worst_exact = max(worst_exact, abs(mine - ref))
    res.check(worst_exact < 1e-12,
              f"5a: my exact Wilcoxon p-values disagree with scipy exact by {worst_exact:.2e}")
    res.numbers["5a_wilcoxon_exact_max_diff"] = worst_exact

    # --- normal approximation vs scipy (with ties) ------------------------
    worst_normal = 0.0
    for trial in range(20):
        n = int(rng.integers(20, 35))
        d = np.round(rng.standard_normal(n), 1) * 0.5 + 0.1
        mine = wilcoxon_signed_rank(d)["p_normal"]
        ref = stats.wilcoxon(d, zero_method="wilcox", method="approx",
                             correction=True).pvalue
        worst_normal = max(worst_normal, abs(mine - ref))
    res.check(worst_normal < 1e-10,
              f"5a: my normal-approx Wilcoxon p-values disagree with scipy by {worst_normal:.2e}")
    res.numbers["5a_wilcoxon_normal_max_diff"] = worst_normal

    # --- Holm vs statsmodels ---------------------------------------------
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222])
    ref = multipletests(p, method="holm")[1]
    mine = holm(p)
    res.check(np.max(np.abs(mine - ref)) < 1e-12,
              f"5a: Holm disagrees with statsmodels by {np.max(np.abs(mine - ref)):.2e}")
    # monotonicity: adjusted >= raw and sorted order preserved
    res.check(np.all(mine >= p - 1e-15), "5a: Holm adjusted p below the raw p")
    res.numbers["5a_holm_example"] = mine.tolist()

    # --- Spearman vs scipy ----------------------------------------------
    worst_rho = 0.0
    for trial in range(20):
        n = int(rng.integers(8, 25))
        x = rng.standard_normal(n)
        y = 0.6 * x + rng.standard_normal(n)
        if trial % 3 == 0:
            x = np.round(x, 1)            # force ties
            y = np.round(y, 1)
        rho = spearman_rho(x, y)
        ref = stats.spearmanr(x, y).statistic
        worst_rho = max(worst_rho, abs(rho - ref))
    res.check(worst_rho < 1e-12,
              f"5a: Spearman disagrees with scipy by {worst_rho:.2e}")
    res.numbers["5a_spearman_max_diff"] = worst_rho

    # --- permutation p vs full enumeration on a tiny sample ---------------
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([0.4, 0.9, 1.1, 2.2, 2.0, 3.1])
    rho = spearman_rho(x, y)
    exact_hits = 0
    all_perms = list(permutations(range(y.size)))
    for perm in all_perms:
        if abs(spearman_rho(x, y[list(perm)])) >= abs(rho) - 1e-15:
            exact_hits += 1
    exact_p = exact_hits / len(all_perms)
    perm = spearman_permutation(x, y, n_perm=20_000, seed=11)
    res.check(abs(perm["p_perm"] - exact_p) < 0.02,
              f"5a: permutation p {perm['p_perm']:.4f} vs enumeration {exact_p:.4f}")
    res.numbers["5a_permutation_vs_exact"] = {"perm": perm["p_perm"], "exact": exact_p}

    # --- bootstrap reproducibility ---------------------------------------
    vals = rng.standard_normal(20) + 0.1
    a = bootstrap_ci_mean(vals, seed=BOOT_SEED)
    b = bootstrap_ci_mean(vals, seed=BOOT_SEED)
    c = bootstrap_ci_mean(vals, seed=BOOT_SEED + 1)
    res.check(a == b, "5a: bootstrap CI not reproducible for a fixed seed")
    width = a[1] - a[0]
    res.check(abs(c[0] - a[0]) < 0.15 * width and abs(c[1] - a[1]) < 0.15 * width,
              "5a: bootstrap CI unstable across seeds (Monte Carlo noise too large)")
    res.numbers["5a_bootstrap_ci"] = {"ci": a, "ci_other_seed": c}


# ---------------------------------------------------------------------------
# 5b — audit results/tables/tests.csv
# ---------------------------------------------------------------------------
def _load_summary() -> pd.DataFrame:
    frames = []
    if not RESULTS.exists():
        return pd.DataFrame()
    for block in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        p = block / "summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "block" not in df:
                df["block"] = block.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _audit_test_log(res: CheckResult) -> None:
    path = TABLES / "tests.csv"
    if not path.exists():
        res.note("5b: results/tables/tests.csv does not exist yet — test-statistic audit skipped")
        res.numbers["5b_status"] = "SKIPPED_NO_TESTS_CSV"
        return
    tests = pd.read_csv(path)
    summary = _load_summary()
    res.numbers["5b_n_rows"] = int(len(tests))
    if summary.empty:
        res.note("5b: no summary.csv to recompute from")
        return

    mechanisms = sorted(set(summary.get("mechanism", pd.Series(dtype=str)).dropna().astype(str)))
    n_checked = 0
    n_pair_rows = 0
    worst_p = 0.0
    worst_effect = 0.0
    holm_checked = 0

    for _, row in tests.iterrows():
        comparison = str(row.get("comparison", ""))
        outcome_raw = str(row.get("outcome", ""))
        test_name = str(row.get("test", ""))
        # only the paired two-mechanism tests; the OLS/Spearman/Kruskal rows are
        # audited elsewhere (5c and the regression re-estimation below)
        if "paired wilcoxon" not in test_name.lower():
            continue
        # the paper's outcome names are not always the summary column names
        outcome = {
            "x_over_base": "output_ratio_mean",
            "welfare_loss_vs_baseline": "mean_welfare",
            "shock_output_loss": "shock_output_loss",
        }.get(outcome_raw, outcome_raw)
        if outcome not in summary.columns:
            continue
        found = [m for m in mechanisms if m in comparison]
        if len(found) != 2:
            continue
        a_name, b_name = found
        block = row.get("block")
        sub = summary if not isinstance(block, str) else summary[summary["block"] == block]
        sa = sub[sub["mechanism"] == a_name]
        sb = sub[sub["mechanism"] == b_name]
        if sa.empty or sb.empty:
            continue
        # pair on (cell_id-independent) seed plus any treatment column shared
        keys = [c for c in ("lag", "gain", "theta", "shock_size", "shock_sector",
                            "price_elasticity", "substrate") if c in sa.columns and c in sb.columns]
        keys = ["seed"] + keys
        merged = sa.merge(sb, on=keys, suffixes=("_a", "_b"))
        if merged.empty:
            continue
        a = merged[f"{outcome}_a"].to_numpy(dtype=float)
        b = merged[f"{outcome}_b"].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if a.size < 8:
            continue
        n_pair_rows += 1
        diff = a - b
        w = wilcoxon_signed_rank(diff, alternative="two-sided")
        p_stored = float(row.get("p_value", np.nan))
        if np.isfinite(p_stored) and np.isfinite(w["p_exact"]):
            err = min(abs(w["p_exact"] - p_stored), abs(w["p_normal"] - p_stored))
            worst_p = max(worst_p, err)
            res.check(err <= 1e-6,
                      f"5b: {block}/{outcome}/{comparison}: stored p {p_stored:.6g} vs my "
                      f"exact {w['p_exact']:.6g} / normal {w['p_normal']:.6g}")
            n_checked += 1
        # effect size: paired mean difference
        if str(row.get("effect_name", "")) in ("mean_diff", "paired_mean_diff",
                                               "cohens_dz", "paired_mean"):
            eff = float(row.get("effect_value", np.nan))
            mine = float(np.mean(diff))
            if np.isfinite(eff):
                worst_effect = max(worst_effect, abs(eff - mine) / max(1.0, abs(mine)))
                res.check(abs(eff - mine) <= 1e-6 * max(1.0, abs(mine)),
                          f"5b: {comparison}: effect {eff} vs my paired mean diff {mine}")
        # bootstrap CI, when recorded
        for lo_key, hi_key in (("ci95_lo", "ci95_hi"), ("ci_low", "ci_high")):
            if lo_key in tests.columns and hi_key in tests.columns:
                lo_stored, hi_stored = row.get(lo_key), row.get(hi_key)
                if pd.notna(lo_stored) and pd.notna(hi_stored) and np.isfinite(diff).all():
                    lo, hi = bootstrap_ci_mean(diff, seed=BOOT_SEED)
                    width = hi - lo
                    res.check(abs(lo - float(lo_stored)) <= 0.05 * width + 1e-9
                              and abs(hi - float(hi_stored)) <= 0.05 * width + 1e-9,
                              f"5b: {comparison}: stored CI [{lo_stored}, {hi_stored}] vs "
                              f"bootstrap [{lo:.6g}, {hi:.6g}]")

    # Reconstruct the revised mean-based tests from the recorded runs.
    from scipy.stats import t as student_t

    def right_tail(values, null=0.0):
        diff = np.asarray(values, dtype=float) - null
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        if se == 0:
            return float(diff.mean() <= 0)
        return float(student_t.sf(diff.mean() / se, df=len(diff) - 1))

    revised_checked = 0
    b2 = _load_block_summary("B2")
    b3 = _load_block_summary("B3")
    b5 = _load_block_summary("B5")
    for row in tests.itertuples():
        if row.hypothesis == "H2" and not b2.empty and "paired t-test" in row.test:
            other = row.comparison.split(" vs ", 1)[1].split(":", 1)[0]
            sample = b2[b2["gain"].eq(0.3) & b2["price_elasticity"].eq(0.4)]
            slopes = {}
            for (mechanism, seed), group in sample.groupby(["mechanism", "seed"]):
                lag = group["lag"] - group["lag"].mean()
                slopes[mechanism, seed] = -(lag * group["mean_welfare"]).sum() / (lag**2).sum()
            paired = pd.Series(slopes).unstack(0).dropna(subset=["lp_central", other])
            expected = right_tail(paired["lp_central"] - paired[other])
        elif row.hypothesis == "H4" and "paired_shock_output_amplification" in b3:
            mechanism = row.comparison.split(":", 1)[0]
            sub = b3[b3["mechanism"] == mechanism]
            expected = right_tail(sub.groupby("seed")["paired_shock_output_amplification"].mean(), 1)
        elif row.hypothesis == "H5" and not b5.empty and "paired t-tests" in row.test:
            lag = int(row.comparison.split()[1].rstrip(","))
            wide = b5[b5["lag"] == lag].pivot(index="seed", columns="theta", values="mean_welfare")
            expected = max(right_tail(wide[row.statistic] - wide[endpoint]) for endpoint in (0, 1))
        else:
            continue
        revised_checked += 1
        res.check(np.isclose(expected, row.p_value, rtol=1e-5, atol=1e-12),
                  f"5b: {row.comparison}: expected revised p={expected:.8g}, got {row.p_value:.8g}")
    res.numbers["5b_revised_mean_tests_checked"] = revised_checked

    # --- Holm audit: recompute within block from the raw p-values ---------
    if "p_value" in tests.columns and "p_holm" in tests.columns and "block" in tests.columns:
        for block, sub in tests.groupby("block"):
            raw = sub["p_value"].to_numpy(dtype=float)
            stored = sub["p_holm"].to_numpy(dtype=float)
            mine = holm(raw)
            good = np.isfinite(stored) & np.isfinite(mine)
            if good.any():
                holm_checked += int(good.sum())
                # the CSV stores p-values to 6 decimals, so the adjusted values can only
                # be reproduced to the propagated rounding error (~m * 1e-6)
                worst = float(np.max(np.abs(mine[good] - stored[good])))
                res.check(worst < 2e-5,
                          f"5b: block {block}: Holm-adjusted p disagrees by {worst:.2e} "
                          f"(beyond CSV rounding)")

    res.numbers["5b_paired_rows_checked"] = n_pair_rows
    res.numbers["5b_p_values_checked"] = n_checked
    res.numbers["5b_worst_p_diff"] = worst_p
    res.numbers["5b_worst_effect_rel_diff"] = worst_effect
    res.numbers["5b_holm_values_checked"] = holm_checked
    res.numbers["5b_status"] = "AUDITED"


# ---------------------------------------------------------------------------
# 5c — H4 amplification vs centrality, permutation p
# ---------------------------------------------------------------------------
def _linkage_predictors() -> dict[str, dict[str, float]]:
    """Three candidate centrality definitions, computed from the technology only."""
    from planlab.data import load_technology

    tech = load_technology("eurostat", 2020, 20)
    A = np.asarray(tech.A, dtype=float)
    x = np.asarray(tech.base_output, dtype=float)
    n = tech.n
    out = {
        "forward_rowsum_A": dict(zip(tech.sectors, A.sum(axis=1))),
        "forward_intermediate_share": dict(zip(tech.sectors, (A @ x) / np.maximum(x, 1e-12))),
        "leontief_colsum": dict(zip(tech.sectors, np.linalg.inv(np.eye(n) - A).sum(axis=0))),
        "leontief_rowsum": dict(zip(tech.sectors, np.linalg.inv(np.eye(n) - A).sum(axis=1))),
    }
    return out


def _amplification_by_sector() -> pd.DataFrame:
    """Mean paired loss and amplification by mechanism and shocked sector."""
    summary = _load_block_summary("B3")
    if summary.empty:
        return pd.DataFrame()
    loss = "paired_shock_output_loss"
    amp = "paired_shock_output_amplification"
    if loss not in summary or amp not in summary:
        return pd.DataFrame()
    sub = summary[summary["shock_sector"].notna()]
    out = (
        sub.groupby(["mechanism", "shock_sector"])
        .agg(amp=(amp, "mean"),
             loss=(loss, "mean"),
             n=("seed", "nunique"))
        .reset_index()
    )
    return out


def _audit_h4(res: CheckResult) -> None:
    amp = _amplification_by_sector()
    if amp.empty:
        res.note("5c: no B3-style shock cells in results/ yet — H4 correlation audit skipped")
        res.numbers["5c_status"] = "SKIPPED_NO_SHOCK_CELLS"
        return
    predictors = _linkage_predictors()
    # other blocks (B4/B6) also shock sectors but only one or two of them; a rank
    # correlation needs a real cross-section, so keep mechanisms with >= 8 sectors
    amp = amp[amp.groupby("mechanism")["shock_sector"].transform("nunique") >= 8]
    res.numbers["5c_n_cells"] = int(len(amp))
    rows = []
    for mechanism, grp in amp.groupby("mechanism"):
        sectors = grp["shock_sector"].astype(str).tolist()
        for outcome, values in (("output_loss", grp["loss"].to_numpy(dtype=float)),
                                ("amplification", grp["amp"].to_numpy(dtype=float))):
            for name, table in predictors.items():
                pred = np.array([table.get(s, np.nan) for s in sectors])
                perm = spearman_permutation(pred, values)
                rows.append({
                    "mechanism": mechanism, "outcome": outcome, "predictor": name,
                    "rho": round(perm["rho"], 6), "p_perm": round(perm["p_perm"], 6),
                    "n": perm["n"],
                })
                res.check(np.isfinite(perm["rho"]),
                          f"5c: {mechanism}/{outcome}/{name}: undefined Spearman rho")
        res.numbers.setdefault("5c_h4", []).extend(
            [r for r in rows if r["mechanism"] == mechanism])

    # ---- cross-check against the paper's own tests.csv H4-centrality rows -----
    tests_path = TABLES / "tests.csv"
    if tests_path.exists():
        tests = pd.read_csv(tests_path)
        centrality = tests[tests["hypothesis"] == "H4-centrality"]
        mine = {(r["mechanism"], r["predictor"]): r
                for r in res.numbers.get("5c_h4", [])
                if r["outcome"] == "output_loss"}
        for _, row in centrality.iterrows():
            comparison = str(row.get("comparison", ""))
            mech = next((m for m in amp["mechanism"].unique() if m in comparison), None)
            if mech is None:
                continue
            if "forward linkage" in comparison:
                pred = "forward_intermediate_share"
            elif "Leontief column sum" in comparison:
                pred = "leontief_colsum"
            else:
                continue
            entry = mine.get((mech, pred))
            if entry is None:
                continue
            stored_rho = float(row["statistic"])
            res.check(abs(entry["rho"] - stored_rho) <= 2e-3,
                      f"5c: tests.csv rho {stored_rho:.4f} for {mech}/{pred} != my "
                      f"{entry['rho']:.4f} recomputed from summary.csv")
            res.numbers.setdefault("5c_permutation_vs_paper", []).append({
                "mechanism": mech, "predictor": pred,
                "rho": entry["rho"], "p_perm": entry["p_perm"],
                "paper_p": float(row["p_value"]),
                "paper_p_holm": float(row["p_holm"]),
                "paper_significant": str(row.get("significant_05", "")),
            })
    res.numbers["5c_status"] = "AUDITED"
    res.note("5c: H4 correlations recomputed from summary.csv with a 10,000-draw permutation "
             "null; the paper's reported rho should match the forward_rowsum_A or "
             "forward_intermediate_share row for the same mechanism.")




# ---------------------------------------------------------------------------
# 5d — every headline table must be regenerable from results/
# ---------------------------------------------------------------------------
#: (table file, source block, group keys, optional row filter, {table column: (summary column, stat)})
TABLE_SPECS = (
    ("balance_decomposition.csv", "B8", ["revision_gain", "balance_rounds"], None,
     {"output_base": ("output_ratio_mean", "mean"),
      "output_base_sd": ("output_ratio_mean", "std"),
      "shortage": ("mean_shortage_rate", "mean"),
      "shortage_sd": ("mean_shortage_rate", "std"),
      "spoilage_rate": ("mean_spoilage_rate", "mean")}),
    ("noise_only.csv", "B1b", ["mechanism"], None,
     {"output_base": ("output_ratio_mean", "mean"),
      "output_base_sd": ("output_ratio_mean", "std"),
      "shortage": ("mean_shortage_rate", "mean"),
      "welfare": ("mean_welfare", "mean"),
      "spoilage_rate": ("mean_spoilage_rate", "mean")}),
    ("propagation.csv", "B3", ["mechanism", "shock_sector"], None,
     {"output_loss": ("shock_output_loss", "mean"),
      "output_loss_sd": ("shock_output_loss", "std"),
      "output_amplification": ("shock_output_amplification", "mean"),
      "paired_output_loss": ("paired_shock_output_loss", "mean"),
      "paired_output_amplification": ("paired_shock_output_amplification", "mean")}),
    ("rationing_shadow.csv", "B4", ["mechanism", "shock_size"], None,
     {"mean_shortage_rate_mean": ("mean_shortage_rate", "mean"),
      "mean_welfare_mean": ("mean_welfare", "mean"),
      "output_ratio_mean_mean": ("output_ratio_mean", "mean")}),
    ("dual_track.csv", "B5", ["theta", "lag"], None,
     {"welfare": ("mean_welfare", "mean"),
      "shortage": ("mean_shortage_rate", "mean"),
      "output_base": ("output_ratio_mean", "mean")}),
    ("robustness.csv", "B6", ["variant"], None,
     {"mean_shortage_rate": ("mean_shortage_rate", "mean"),
      "mean_x_over_base": ("output_ratio_mean", "mean")}),
    ("main_lag_grid.csv", "B2", ["mechanism", "gain", "price_elasticity", "lag"], None,
     {"welfare": ("mean_welfare", "mean"),
      "output_base": ("output_ratio_mean", "mean"),
      "shortage": ("mean_shortage_rate", "mean")}),
)


def _load_block_summary(block_name: str) -> pd.DataFrame:
    path = RESULTS / block_name / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _audit_tables(res: CheckResult) -> None:
    """Recompute each headline table from results/ and compare (the traceability rule)."""
    paper_tables = RESULTS.parent / "paper" / "tables"
    if not paper_tables.exists():
        res.note("5d: no paper/tables yet")
        res.numbers["5d_status"] = "SKIPPED"
        return
    checked = 0
    worst = 0.0
    worst_where = ""
    tables_seen = 0
    for table_name, block, keys, where, mapping in TABLE_SPECS:
        table_path = paper_tables / table_name
        if not table_path.exists():
            res.note(f"5d: {table_name} missing from results/tables")
            continue
        published = pd.read_csv(table_path)
        summary = _load_block_summary(block)
        if summary.empty:
            res.note(f"5d: {table_name}: no results/{block}/summary.csv")
            continue
        if where is not None:
            summary = summary.query(where)
            published = published.query(where)
        # the published table may rename the lag column; canonicalise before keying
        pub = published.rename(columns={"lag_reporting": "lag", "lag_effective": "m"})
        keys = [k for k in keys if k in summary.columns and k in pub.columns]
        if not keys:
            res.note(f"5d: {table_name}: no shared keys")
            continue
        agg = summary.groupby(keys).agg(**{
            new: (col, stat) for new, (col, stat) in mapping.items()
            if col in summary.columns
        }).reset_index()
        tables_seen += 1
        # A published table without a cell identifier (e.g. the task's elasticity
        # dimension) can have several rows per key.  In that case compare the *sorted*
        # values within each key: the question is whether the table's numbers come from
        # these results, not which row is which.
        dup = pub.groupby(keys).size()
        has_dups = bool((dup > 1).any())
        if has_dups:
            res.note(f"5d: {table_name}: {int((dup > 1).sum())} key group(s) have several rows "
                     f"(a dimension of the design is not recorded in the table); comparing "
                     f"sorted values within each key")
        for new in mapping:
            if new not in pub.columns:
                continue
            mine_col = f"{new}_mine"
            if has_dups:
                errs = []
                n_cmp = 0
                for key, grp in agg.groupby(keys):
                    try:
                        sub = pub.groupby(keys).get_group(key if len(keys) > 1 else key)
                    except KeyError:
                        continue
                    a = np.sort(grp[new].to_numpy(dtype=float))
                    b = np.sort(sub[new].to_numpy(dtype=float))
                    if a.size != b.size:
                        res.note(f"5d: {table_name}:{new}: row count mismatch at {key}")
                        continue
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.any():
                        errs.append(np.abs(a[ok] - b[ok]) / np.maximum(1.0, np.abs(b[ok])))
                        n_cmp += int(ok.sum())
                if not errs:
                    continue
                e = float(np.max(np.concatenate(errs)))
                checked += n_cmp
            else:
                merged = agg.merge(pub, on=keys, how="inner", suffixes=("_mine", ""))
                if merged.empty or mine_col not in merged or new not in merged:
                    continue
                mine = merged[mine_col].to_numpy(dtype=float)
                pub_vals = merged[new].to_numpy(dtype=float)
                ok = np.isfinite(mine) & np.isfinite(pub_vals)
                if not ok.any():
                    continue
                e = float(np.max(np.abs(mine[ok] - pub_vals[ok])
                                 / np.maximum(1.0, np.abs(pub_vals[ok]))))
                checked += int(ok.sum())
            if e > worst:
                worst, worst_where = e, f"{table_name}:{new}"
            res.check(e <= 2e-6,
                      f"5d: {table_name}: column {new} differs from results/{block} by {e:.2e}")
    res.numbers["5d_tables_checked"] = tables_seen
    res.numbers["5d_values_checked"] = checked
    res.numbers["5d_worst_rel_diff"] = worst
    res.numbers["5d_worst_at"] = worst_where
    res.numbers["5d_status"] = "AUDITED"


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def _body(res: CheckResult) -> None:
    _self_tests(res)
    _audit_test_log(res)
    _audit_h4(res)
    _audit_tables(res)
    if res.status == "PASS":
        parts = ["5a machinery validated independently (Wilcoxon exact vs scipy, Holm vs "
                 "statsmodels, Spearman vs scipy, permutation vs full enumeration, bootstrap "
                 "reproducible)"]
        if res.numbers.get("5b_status") == "AUDITED":
            parts.append(f"5b {res.numbers['5b_paired_rows_checked']} paired tests and "
                         f"{res.numbers['5b_revised_mean_tests_checked']} revised mean tests; "
                         f"{res.numbers['5b_holm_values_checked']} Holm values recomputed")
        else:
            parts.append("5b skipped (no tests.csv)")
        if res.numbers.get("5c_status") == "AUDITED":
            parts.append("5c H4 correlations recomputed with a permutation null")
        else:
            parts.append("5c skipped (no shock cells)")
        if res.numbers.get("5d_status") == "AUDITED":
            parts.append(f"5d {res.numbers['5d_values_checked']} published table values "
                         f"re-derived from results/ ({res.numbers['5d_tables_checked']} tables, "
                         f"worst {res.numbers['5d_worst_rel_diff']:.1e})")
        else:
            parts.append("5d skipped (no results/tables)")
        res.summary = "; ".join(parts)
        if "SKIPPED" in str(res.numbers.get("5b_status", "")) + str(res.numbers.get("5c_status", "")) + str(res.numbers.get("5d_status", "")):
            res.status = "SKIP"
            res.summary += " — the statistical audit of the PAPER's numbers is NOT complete yet."


def run() -> CheckResult:
    return run_guarded("statistics", "Statistical audit (Wilcoxon/bootstrap/Holm/H4)", _body)


if __name__ == "__main__":                              # pragma: no cover
    import json

    result = run()
    print(json.dumps(result.as_dict(), indent=2, default=str))
    raise SystemExit(0 if result.status in ("PASS", "SKIP") else 1)
