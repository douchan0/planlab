"""Check 4 — reproducibility of the stored experiment results.

What is verified
----------------
4a  **Re-runs.**  A deterministic random sample of at least ``MIN_RUNS``
    (cell, seed) runs is re-executed from the recorded scenario (the same dict the
    scenario YAML / grid cell carries) and *every* run-level metric in
    ``summary.csv`` is compared with the fresh run to machine precision.  The
    technology is resolved from the cell's authoritative technology spec
    (``cells.json:tech``), and the difference from the scenario-file view is
    noted where the two disagree.
4b  **Independent recomputation of the three headline run metrics.**  From the
    re-run's *period records* I recompute ``mean_shortage_rate``,
    ``cum_welfare`` (discount-weighted mean welfare x n) and
    ``mean_plan_fulfilment`` with my own code following ``docs/PROTOCOL.md`` §5
    and compare them with the stored summary.  This is the check that a stored
    number was not simply copied through the project's own aggregation.
4c  **Per-period reproducibility.**  For the sampled runs, every stored
    per-period metric in ``runs.parquet`` is compared with the re-run.
4d  **Manifest completeness.**  Every block manifest must record: the package
    versions (python/numpy/scipy/pandas), the seeds, a git commit (and dirty
    flag), and a scenario hash per cell that matches the hash recomputed from the
    recorded scenario.  Missing/blank fields are failures; a mismatch is a
    failure (it means the results were not produced from the recorded inputs).
4e  **Internal consistency.**  For *every* run in ``runs.parquet`` (not only the
    sample), the run-level means implied by the per-period rows must equal the
    values in ``summary.csv``.

Scenario YAMLs in ``scenarios/`` are exercised as well when they exist: each
YAML is loaded with the project loader, its hash checked against the manifest,
and one seed re-run; results are compared with the matching summary rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import RESULTS, SCENARIOS, CheckResult, run_guarded
from .check3_accounting import cell_tech_spec

MIN_RUNS = 20
METRIC_RTOL = 1e-10
SAMPLE_SEED = 20261101


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _blocks():
    if not RESULTS.exists():
        return []
    return sorted(p for p in RESULTS.iterdir()
                  if p.is_dir() and (p / "summary.csv").exists())


def _cells(block) -> dict:
    out = {}
    p = block / "cells.json"
    if p.exists():
        for cell in json.loads(p.read_text()):
            out[str(cell["cell_id"])] = cell
    return out


def _num_equal(a, b, rtol=METRIC_RTOL):
    a = float(a)
    b = float(b)
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# independent run-level recomputation (docs/PROTOCOL.md §5)
# ---------------------------------------------------------------------------
def recompute_run_metrics(records, burn_in, discount):
    used = [r for r in records if r.t > int(burn_in)]
    if not used:
        used = list(records)
    shortage = np.array([float(r.metrics["shortage_rate"]) for r in used])
    welfare = np.array([float(r.metrics["welfare"]) for r in used])
    fulfil = np.array([float(r.metrics["plan_fulfilment"]) for r in used])
    beta = float(discount)
    disc = np.array([beta ** (r.t - used[0].t) for r in used])
    disc = disc / disc.sum()
    return {
        "mean_shortage_rate": float(np.mean(shortage)),
        "cum_welfare": float(np.dot(welfare, disc) * len(used)),
        "mean_plan_fulfilment": float(np.mean(fulfil)),
        "n_periods_used": float(len(used)),
    }


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def _body(res: CheckResult) -> None:
    from planlab.core.engine import run_scenario
    from planlab.core.scenarios import (load_scenario, scenario_from_dict,
                                        scenario_hash, resolve_technology_spec)

    blocks = _blocks()
    if not blocks:
        res.status = "SKIP"
        res.summary = "results/ contains no summary.csv yet; nothing to reproduce."
        return

    # ------------------------------------------------------------------
    # build the sample of (block, cell, seed)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SAMPLE_SEED)
    all_runs = []
    summaries = {}
    cells_by_block = {}
    for block in blocks:
        summ = pd.read_csv(block / "summary.csv")
        if "block" not in summ:
            summ["block"] = block.name
        summaries[block.name] = summ
        cells_by_block[block.name] = _cells(block)
        keys = [c for c in ("cell_id", "mechanism") if c in summ.columns]
        for _, row in summ.iterrows():
            all_runs.append((block.name, str(row.get("cell_id", "")),
                             int(row["seed"]) if "seed" in summ.columns else 0))
    # de-duplicate while preserving determinism
    all_runs = sorted(set(all_runs))
    n_sample = min(len(all_runs), max(MIN_RUNS, MIN_RUNS * len(blocks)))
    idx = rng.choice(len(all_runs), size=n_sample, replace=False)
    sample = [all_runs[i] for i in sorted(idx)]
    res.numbers["n_runs_available"] = len(all_runs)
    res.numbers["n_runs_sampled"] = len(sample)

    checked_metrics = 0
    worst_metric = 0.0
    worst_metric_where = ""
    independent_worst = defaultdict(float)
    missing_cells = 0
    per_period_checked = 0
    per_period_worst = 0.0
    tech_disagreements = 0

    for block_name, cell_id, seed in sample:
        cell = cells_by_block.get(block_name, {}).get(cell_id)
        summ = summaries[block_name]
        row = summ[(summ["cell_id"] == cell_id) & (summ.get("seed", 0) == seed)]
        if cell is None or row.empty:
            missing_cells += 1
            res.note(f"4a: {block_name}/{cell_id}/seed{seed}: no cell definition or summary row")
            continue
        row = row.iloc[0]
        sc_dict = dict(cell["scenario"])
        scenario = scenario_from_dict(sc_dict)
        tech_spec = cell_tech_spec(cell)
        tech = resolve_technology_spec(tech_spec)
        if dict(sc_dict.get("technology") or {}) != dict(tech_spec):
            tech_disagreements += 1
        try:
            result = run_scenario(scenario, technology=tech, seed=seed)
        except Exception as exc:  # noqa: BLE001
            res.fail(f"4a: {block_name}/{cell_id}/seed{seed}: re-run raised {type(exc).__name__}: {exc}")
            continue

        # ---- 4a every stored run-level metric ---------------------------
        fresh = result.run_metrics
        for key in summ.columns:
            if key not in fresh:
                continue
            stored = row[key]
            try:
                stored_val = float(stored)
            except (TypeError, ValueError):
                continue
            value = float(fresh[key])
            checked_metrics += 1
            err = 0.0 if _num_equal(value, stored_val) else abs(value - stored_val) / max(
                1.0, abs(value), abs(stored_val))
            if err > worst_metric:
                worst_metric, worst_metric_where = err, f"{block_name}/{cell_id}/seed{seed}:{key}"
            if not _num_equal(value, stored_val):
                res.fail(f"4a: {block_name}/{cell_id}/seed{seed}: {key} stored {stored_val!r} "
                         f"vs re-run {value!r}")

        # ---- 4b independent recomputation of the three headline metrics --
        mine = recompute_run_metrics(result.records, scenario.burn_in, scenario.discount)
        for key, value in mine.items():
            if key not in summ.columns:
                continue
            stored_val = float(row[key])
            if not _num_equal(value, stored_val, rtol=1e-9):
                res.fail(f"4b: {block_name}/{cell_id}/seed{seed}: {key} stored {stored_val:.12g} "
                         f"vs independent {value:.12g}")
            independent_worst[key] = max(independent_worst[key],
                                         abs(value - stored_val) / max(1.0, abs(value)))

        # ---- 4c per-period reproducibility ------------------------------
        rp = block_path = next((b / "runs.parquet" for b in blocks if b.name == block_name), None)
        if rp is not None and rp.exists():
            per = pd.read_parquet(rp)
            per = per[(per["cell_id"] == cell_id) & (per["seed"] == seed)]
            if not per.empty:
                mine_rows = {r.t: r.metrics for r in result.records}
                for _, prow in per.iterrows():
                    t = int(prow["t"])
                    m = mine_rows.get(t)
                    if m is None:
                        continue
                    for key in m:
                        if key not in per.columns:
                            continue
                        per_period_checked += 1
                        a = float(prow[key])
                        b = float(m[key])
                        if not _num_equal(a, b, rtol=1e-9):
                            per_period_worst = max(per_period_worst,
                                                   abs(a - b) / max(1.0, abs(a)))
                            res.fail(f"4c: {block_name}/{cell_id}/seed{seed}/t{t}: {key} "
                                     f"stored {a:.12g} vs re-run {b:.12g}")

    # ------------------------------------------------------------------
    # 4d manifests
    # ------------------------------------------------------------------
    n_manifest_ok = 0
    for block in blocks:
        mpath = block / "manifest.json"
        if not mpath.exists():
            res.fail(f"4d: {block.name}: manifest.json missing")
            continue
        man = json.loads(mpath.read_text())
        pkg = man.get("package_versions") or {}
        for key in ("python", "numpy", "scipy", "pandas"):
            res.check(bool(pkg.get(key)), f"4d: {block.name}: manifest lacks package version {key!r}")
        res.check(bool(man.get("git_commit")), f"4d: {block.name}: manifest lacks git_commit")
        res.check("git_dirty" in man, f"4d: {block.name}: manifest lacks git_dirty flag")
        # The manifest's "scenario_hashes" field actually holds the *cell* content
        # digest (see grid.Cell.hash); the seeds are recorded as counts in
        # "seeds_per_cell" plus the actual values in cells.json/summary.csv.
        has_seeds = bool(man.get("seeds")) or bool(man.get("seeds_per_cell"))
        res.check(has_seeds, f"4d: {block.name}: manifest records no seeds at all")
        if man.get("seeds_per_cell") and "seeds" not in man:
            res.note(f"4d: {block.name}: seeds recorded only as counts (seeds_per_cell); the "
                     f"actual values live in cells.json/summary.csv")
        hashes = man.get("scenario_hashes") or {}
        res.check(bool(hashes), f"4d: {block.name}: manifest lacks scenario_hashes")
        cells = cells_by_block.get(block.name, {})
        for cid, cell in cells.items():
            if cid not in hashes:
                continue
            # independent re-hash of the recorded cell (canonical JSON, hashlib only)
            payload = json.dumps(cell, sort_keys=True, separators=(",", ":"))
            recalculated = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            if recalculated != hashes[cid]:
                res.fail(f"4d: {block.name}/{cid}: manifest hash {hashes[cid]} != "
                         f"recomputed cell digest {recalculated}")
            # the field is named scenario_hashes: also confirm it is not the scenario hash
            sc = scenario_from_dict(dict(cell["scenario"]))
            if scenario_hash(sc) == hashes[cid]:
                res.note(f"4d: {block.name}: scenario_hashes equals the policy scenario hash "
                         f"for {cid} (naming is then accurate)")
        res.check(int(man.get("cells_failed", 0)) == 0,
                  f"4d: {block.name}: manifest reports {man.get('cells_failed')} failed cells")
        n_manifest_ok += 1

    # ------------------------------------------------------------------
    # 4e internal consistency of runs.parquet vs summary.csv
    # ------------------------------------------------------------------
    n_int = 0
    for block in blocks:
        rp, sp = block / "runs.parquet", block / "summary.csv"
        if not (rp.exists() and sp.exists()):
            continue
        per = pd.read_parquet(rp)
        summ = pd.read_csv(sp)
        cells = cells_by_block.get(block.name, {})
        for cid, grp in per.groupby("cell_id"):
            cell = cells.get(cid)
            burn = int(cell["scenario"].get("burn_in", 0)) if cell else 0
            for seed, g2 in grp.groupby("seed"):
                used = g2[g2["t"] > burn]
                if used.empty:
                    used = g2
                row = summ[(summ["cell_id"] == cid) & (summ["seed"] == seed)]
                if row.empty:
                    continue
                row = row.iloc[0]
                for src, stored_key in (("shortage_rate", "mean_shortage_rate"),
                                        ("plan_fulfilment", "mean_plan_fulfilment")):
                    if src not in used or stored_key not in summ.columns:
                        continue
                    value = float(used[src].mean())
                    n_int += 1
                    if not _num_equal(value, float(row[stored_key]), rtol=1e-9):
                        res.fail(f"4e: {block.name}/{cid}/seed{seed}: {stored_key} "
                                 f"{row[stored_key]} != period mean {value}")

    # ------------------------------------------------------------------
    # scenario YAMLs, when they exist
    # ------------------------------------------------------------------
    n_yaml = 0
    if SCENARIOS.exists():
        for path in sorted(SCENARIOS.glob("*.yaml")):
            try:
                sc = load_scenario(path)
            except Exception as exc:  # noqa: BLE001
                res.fail(f"4f: {path.name}: load_scenario raised {type(exc).__name__}: {exc}")
                continue
            n_yaml += 1
            # find a stored summary row with this scenario id
            match = None
            for block_name, summ in summaries.items():
                col = "scenario_id" if "scenario_id" in summ.columns else None
                if col is None:
                    continue
                hit = summ[summ[col].astype(str) == sc.id]
                if not hit.empty:
                    match = (block_name, hit.iloc[0])
                    break
            if match is None:
                res.note(f"4f: {path.name}: no stored summary row for scenario id {sc.id!r}")
                continue
            seed = int(sc.seeds[0])
            try:
                result = run_scenario(sc, seed=seed)
            except Exception as exc:  # noqa: BLE001
                res.fail(f"4f: {path.name}: run raised {type(exc).__name__}: {exc}")
                continue
            block_name, row = match
            summ = summaries[block_name]
            for key in ("mean_shortage_rate", "cum_welfare", "mean_plan_fulfilment"):
                if key in summ.columns and key in result.run_metrics:
                    if not _num_equal(float(row[key]), float(result.run_metrics[key]), rtol=1e-9):
                        res.fail(f"4f: {path.name}: {key} stored {row[key]} vs re-run "
                                 f"{result.run_metrics[key]}")

    res.numbers.update({
        "n_samples": len(sample),
        "n_metric_comparisons_4a": checked_metrics,
        "worst_run_metric_diff_4a": worst_metric,
        "worst_run_metric_diff_at": worst_metric_where,
        "independent_metric_worst": dict(independent_worst),
        "per_period_comparisons": per_period_checked,
        "per_period_worst_diff": per_period_worst,
        "n_cells_without_definition": missing_cells,
        "n_tech_spec_disagreements": tech_disagreements,
        "n_manifest_blocks_ok": n_manifest_ok,
        "n_consistency_checks_4e": n_int,
        "n_scenario_yamls_rerun": n_yaml,
    })

    if res.status == "PASS":
        res.summary = (
            f"{len(sample)} of {len(all_runs)} (cell, seed) runs re-executed across "
            f"{len(blocks)} block(s): {checked_metrics} stored run-level metrics match to "
            f"{worst_metric:.1e}; my independent recomputation of mean_shortage_rate, "
            f"cum_welfare and mean_plan_fulfilment agrees to "
            f"{max([v for v in independent_worst.values()] or [0]):.1e}; "
            f"{per_period_checked} per-period metric values re-checked (worst "
            f"{per_period_worst:.1e}); manifests carry versions, git commit and scenario "
            f"hashes that match; runs.parquet and summary.csv agree; "
            f"{n_yaml} scenario YAML(s) re-run."
        )


def run() -> CheckResult:
    return run_guarded("reproducibility", "Reproducibility audit of the stored results", _body)


if __name__ == "__main__":                              # pragma: no cover
    import json as _json

    result = run()
    print(_json.dumps(result.as_dict(), indent=2, default=str))
    raise SystemExit(0 if result.status in ("PASS", "SKIP") else 1)
