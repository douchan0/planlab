"""Lead's independent reference run — insurance and cross-check for the experiment layer.

This script is deliberately separate from :mod:`planlab.experiments`: it is a second,
independent implementation of the pre-registered blocks, written by the Lead, so that
(a) the paper has generated numbers even if the experiment layer is late, and (b) the
experiment layer's numbers can be cross-checked against something that does not share
its code.

It produces ``reference_results/`` with one row per run, a per-cell aggregate with seed
dispersion, and a manifest.  Nothing here is used to *replace* the experiment layer;
where the two disagree, the disagreement is a finding.

    .venv/bin/python scripts/lead_reference.py            # full reference grid
    .venv/bin/python scripts/lead_reference.py --quick    # ~200 runs, smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reference_results"

# --------------------------------------------------------------------------- grid
T = 80
BURN = 30
SHOCK_T = 40
SHOCK_DUR = 8
SHOCK_SIZE = 0.5
STOCK = 0.10
MECH_ALL = ["lp_central", "material_balance", "rationing_priority", "tatonnement", "market", "hybrid_dual_track"]
GAINS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.60]
LAGS = [0, 1, 2, 3, 4, 6]
SECTORS = [f"c{i:02d}" for i in range(1, 21)]


#: Household- and planner-side parameters that must be IDENTICAL across mechanisms
#: for the comparison to be identified.  Only the institutional knob may vary.
MECHANISM_BASE_PARAMS = {
    "backlog_carry": 0.0,        # household-side feedback: never an institutional difference
    "stock_target_ratio": 0.10,  # planner-side, held common
    "max_target_ratio": 2.5,
    "min_target_ratio": 0.25,
}


def _base_params(mechanism: str, gain: float | None) -> dict:
    p = dict(MECHANISM_BASE_PARAMS)
    if mechanism == "lp_central":
        p["use_shadow_prices"] = 0.0          # see docs/AMENDMENTS.md; duals are diagnostics
    if gain is not None:
        if mechanism == "tatonnement":
            p["price_gain"] = gain
        elif mechanism == "material_balance":
            p["revision_gain"] = gain
        elif mechanism == "hybrid_dual_track":
            p["market_gain"] = gain
        elif mechanism == "rationing_priority":
            p["quota_speed"] = gain
    return p


def cells(quick: bool) -> list[dict]:
    """Declarative cell list: each cell is a scenario specification plus a seed list."""
    out: list[dict] = []
    seeds20 = list(range(20))
    seeds10 = list(range(10))

    # R1 — base-year reproduction (no shock, no noise, no lag), 1 seed
    for mech in MECH_ALL:
        out.append(dict(block="R1", cell=f"R1_{mech}", mechanism=mech,
                        params=_base_params(mech, None), lag=0, noise=0.0,
                        sigma=0.0, ar1=0.0, shock=None, seeds=[0], n_seeds=1, gain=None))

    # R2b — noise only, no shock: the information rent of an institution
    for mech in MECH_ALL:
        out.append(dict(block="R2b", cell=f"R2b_{mech}", mechanism=mech,
                        params=_base_params(mech, None), lag=0, noise=0.03,
                        sigma=0.02, ar1=0.3, shock=None, seeds=seeds20,
                        n_seeds=20, gain=None))

    # R2 — gain x lag (H1 on prices, H1' on targets)
    g_lags = [0, 2, 4] if quick else LAGS
    g_gains = [0.3, 1.0] if quick else GAINS
    g_seeds = list(range(4)) if quick else seeds10
    for mech in ("tatonnement", "material_balance"):
        for gain, lag in itertools.product(g_gains, g_lags):
            out.append(dict(block="R2", cell=f"R2_{mech}_g{gain:.2f}_l{lag}", mechanism=mech,
                            params=_base_params(mech, gain), lag=lag, noise=0.03,
                            sigma=0.02, ar1=0.3,
                            shock=dict(sector="c12", size=SHOCK_SIZE), seeds=g_seeds,
                            n_seeds=len(g_seeds), gain=gain))

    # R3 — sectoral shock propagation
    r3_sectors = SECTORS[:3] if quick else SECTORS
    r3_seeds = list(range(3)) if quick else seeds10
    for mech in ("material_balance", "rationing_priority", "tatonnement"):
        for sector in r3_sectors:
            out.append(dict(block="R3", cell=f"R3_{mech}_{sector}", mechanism=mech,
                            params=_base_params(mech, None), lag=0, noise=0.02,
                            sigma=0.02, ar1=0.3,
                            shock=dict(sector=sector, size=SHOCK_SIZE), seeds=r3_seeds,
                            n_seeds=len(r3_seeds), gain=None))

    # R4 — rationing vs shadow prices under shocks of different depth
    r4_seeds = list(range(8)) if quick else seeds20
    for mech in ("rationing_priority", "tatonnement"):
        for size in (0.90, 0.70, 0.50):
            out.append(dict(block="R4", cell=f"R4_{mech}_s{int((1 - size) * 100)}",
                            mechanism=mech, params=_base_params(mech, None), lag=0,
                            noise=0.02, sigma=0.02, ar1=0.3,
                            shock=dict(sector="c12", size=size), seeds=r4_seeds,
                            n_seeds=len(r4_seeds), gain=None))

    # R5 — dual track share theta
    thetas = [0.5] if quick else [round(i / 10, 1) for i in range(11)]
    r5_seeds = list(range(5)) if quick else seeds20
    for theta, lag in itertools.product(thetas, (0, 3)):
        p = _base_params("hybrid_dual_track", None)
        p["theta"] = theta
        out.append(dict(block="R5", cell=f"R5_theta{theta:.1f}_l{lag}",
                        mechanism="hybrid_dual_track", params=p, lag=lag, noise=0.02,
                        sigma=0.02, ar1=0.3, shock=dict(sector="c12", size=SHOCK_SIZE),
                        seeds=r5_seeds, n_seeds=len(r5_seeds), gain=theta))
    return out


# ------------------------------------------------------------------------- worker
_TECH_CACHE: dict = {}


def _tech():
    if "t" not in _TECH_CACHE:
        from planlab.data import load_technology

        _TECH_CACHE["t"] = load_technology("eurostat", 2020, 20)
    return _TECH_CACHE["t"]


def _regress_gain(records, tech) -> tuple[float, float]:
    """Local loop-gain estimators, computed SECTOR BY SECTOR then pooled by median.

    ``b_price`` : median over sectors of -d(unmet share_i)/d(log p_i)
    ``b_target``: median over sectors of -d(unmet share_i)/d(log target_i / base_i)

    Sector-by-sector rather than aggregate matters: the aggregate price level barely
    moves when one sector's price changes, so an aggregate regression attenuates the
    slope to numerical noise.  A *negative* estimate means the loop is positive
    (raising the instrument raises the shortage) -- the H1' case.
    """
    if len(records) < 12:
        return float("nan"), float("nan")
    n = len(tech.sectors)
    price = np.array([r.price for r in records])          # (T, n)
    unmet_share = np.array([
        r.unmet / np.maximum(r.demand, 1e-12) for r in records
    ])
    target_ratio = np.array([
        r.plan_targets / np.maximum(tech.base_output, 1e-12) for r in records
    ])

    def med_slope(x, y):
        out = []
        for j in range(n):
            xj = np.log(np.maximum(x[:, j], 1e-12))
            xj = xj - xj.mean()
            denom = float(np.dot(xj, xj))
            if denom <= 1e-14:
                continue
            out.append(-float(np.dot(xj, y[:, j] - y[:, j].mean()) / denom))
        return float(np.median(out)) if out else float("nan")

    return med_slope(price, unmet_share), med_slope(target_ratio, unmet_share)


def run_one(task: tuple[dict, int]) -> dict:
    cell, seed = task
    from planlab.core.engine import run_scenario
    from planlab.core.scenarios import scenario_from_dict

    tech = _tech()
    shocks = []
    if cell["shock"] is not None:
        shocks = [{
            "type": "capacity",
            "sector": cell["shock"]["sector"],
            "start": SHOCK_T,
            "duration": SHOCK_DUR,
            "size": cell["shock"]["size"],
        }]
    sc = scenario_from_dict({
        "id": cell["cell"],
        "mechanism": cell["mechanism"],
        "technology": {"source": "eurostat", "year": 2020, "n_sectors": 20},
        "periods": T,
        "burn_in": BURN,
        "seeds": [seed],
        "stock_ratio": STOCK,
        "mechanism_params": cell["params"],
        "info": {"lag": cell["lag"], "noise": cell["noise"]},
        "demand": {"ar1": cell["ar1"], "sigma": cell["sigma"], "price_elasticity": 0.6},
        "shocks": shocks,
    })
    try:
        res = run_scenario(sc, technology=tech, seed=seed)
    except Exception as exc:  # keep the grid running; record the failure
        return {"block": cell["block"], "cell": cell["cell"], "mechanism": cell["mechanism"],
                "seed": seed, "error": f"{type(exc).__name__}: {exc}"}

    m = dict(res.run_metrics)
    row = {
        "block": cell["block"],
        "cell": cell["cell"],
        "mechanism": cell["mechanism"],
        "gain": -1.0 if cell["gain"] is None else float(cell["gain"]),
        "lag": int(cell["lag"]),
        "k_eff": int(cell["lag"]) + 1,
        "seed": int(seed),
        "shock_sector": (cell["shock"] or {}).get("sector", ""),
        "shock_size": (cell["shock"] or {}).get("size", 1.0),
        "output_ratio": float(np.mean([float(np.sum(r.x)) for r in res.records[-1:]]))
        / float(np.sum(tech.base_output)),
        "error": "",
    }
    row.update({k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))})
    b_p, b_t = _regress_gain(res.records, tech)
    row["b_price"], row["b_target"] = b_p, b_t
    return row


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small smoke-test grid")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    import pandas as pd

    spec = cells(args.quick)
    tasks = [(c, s) for c in spec for s in c["seeds"]]
    print(f"reference grid: {len(spec)} cells, {len(tasks)} runs, {args.workers} workers")
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)} runs  {el:6.1f}s  ({el / i * 1000:.0f} ms/run)")
    df = pd.DataFrame(rows)
    errs = df[df["error"] != ""]
    if len(errs):
        print(f"WARNING: {len(errs)} runs failed")
        print(errs[["cell", "seed", "error"]].head(10).to_string(index=False))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)

    ok = df[df["error"] == ""]
    keys = [c for c in ok.columns if c not in
            {"block", "cell", "mechanism", "seed", "shock_sector", "error"}]
    numeric = [c for c in keys if pd.api.types.is_numeric_dtype(ok[c])]
    agg = ok.groupby(["block", "cell", "mechanism"], as_index=False)[numeric].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg.to_csv(out / "aggregate.csv", index=False)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - t0, 1),
        "n_cells": len(spec),
        "n_runs": len(tasks),
        "n_failed": int(len(errs)),
        "grid": {"T": T, "burn_in": BURN, "shock_start": SHOCK_T, "shock_duration": SHOCK_DUR,
                 "shock_size": SHOCK_SIZE, "stock_ratio": STOCK, "gains": GAINS, "lags": LAGS},
        "substrate": "data/processed/eurostat_2020_n20",
        "versions": {k: v for k, v in
                     (("python", platform.python_version()), ("numpy", np.__version__))},
        "git_head": _git_head(),
        "argv": sys.argv,
    }
    try:
        import scipy

        manifest["versions"]["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import pandas as _pd

        manifest["versions"]["pandas"] = _pd.__version__
    except Exception:
        pass
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/runs.csv ({len(df)} rows), aggregate.csv, manifest.json")
    print(f"wall: {manifest['wall_seconds']}s")
    return 0


def _git_head() -> str:
    head = ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text().strip().split(": ")[-1]
        return (ROOT / ".git" / ref).read_text().strip()[:12]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
