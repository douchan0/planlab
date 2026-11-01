"""Monte Carlo runner: cells in, per-period records and run-level statistics out.

The runner is the only part of the experiment layer that executes simulations, and
it is written to be *boring*: given the cells of :mod:`planlab.experiments.grid`
and the seeds they carry, it produces byte-identical numbers on every run, skips
work that is already on disk, and records enough provenance in
``results/<block>/manifest.json`` that a reader can tell what was actually run.

Layout written per block::

    results/<block>/
        cells/<cell_id>.json      # cell definition, summary rows, sector panel, hash
        cells/<cell_id>.parquet   # one row per (seed, period): all period metrics
        runs.parquet              # concatenation of the shards
        summary.csv               # one row per (cell, seed): run-level metrics
        aggregate.csv             # one row per cell: mean/sd/95% CI across seeds
        sector_panel.parquet      # one row per (cell, seed, sector)
        cells.json                # the full design, for reproducibility
        failures.json             # cells that raised (retried on the next run)
        manifest.json             # versions, hashes, wall time, mechanisms present

Resumability is per cell: a shard whose ``cell_hash`` still matches the current
grid is skipped.  A mechanism that is not registered yet is skipped with a
warning and reported in the manifest (``mechanisms_missing``); re-running after it
lands picks the cells up because their files then simply do not exist.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .grid import (
    BURN_IN,
    PERIODS,
    SHOCK_DURATION,
    SHOCK_SIZE,
    SHOCK_START,
    STANDARD_SECTOR,
    STOCK_RATIO,
    Cell,
    available_mechanisms_safe,
    build_cells,
    cell_from_dict,
    split_runnable,
)

__all__ = [
    "BLOCK_ENVIRONMENT",
    "BlockResult",
    "assemble_block",
    "run_block",
    "run_cells",
    "block_results_dir",
    "results_root",
    "is_complete",
    "code_fingerprint",
    "package_versions",
]

#: The standard experimental environment every block inherits.  Recorded in every
#: manifest so that a reader never has to infer it from the code.
BLOCK_ENVIRONMENT: Mapping[str, Any] = {
    "periods": PERIODS,
    "burn_in": BURN_IN,
    "shock_start": SHOCK_START,
    "shock_duration": SHOCK_DURATION,
    "shock_size": SHOCK_SIZE,
    "shock_sector": STANDARD_SECTOR,
    "stock_ratio": STOCK_RATIO,
    "note": (
        "docs/PROTOCOL.md §4 as amended by docs/AMENDMENTS.md 1-3; the horizon "
        "T=80 is recorded as a deviation in docs/DEVIATIONS.md"
    ),
}

#: metrics copied into ``summary.csv`` (all of them); listed for documentation
SUMMARY_METRIC_PREFIXES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def results_root(root: str | Path | None = None) -> Path:
    """The ``results/`` directory of the checkout (created on demand)."""
    if root is None:
        from planlab import paths

        base = paths.results_dir()
    else:
        base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    return base


def block_results_dir(block: str, root: str | Path | None = None) -> Path:
    """``results/<block>/`` with the ``cells`` subdirectory created."""
    path = results_root(root) / block
    (path / "cells").mkdir(parents=True, exist_ok=True)
    return path


def package_versions() -> dict[str, str]:
    """Versions of everything that can change a number."""
    import matplotlib
    import scipy

    import planlab

    return {
        "planlab": getattr(planlab, "__version__", "0.1.0"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyarrow": _safe_version("pyarrow"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _safe_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # pragma: no cover - optional dependency
        return "missing"


_CODE_FP: dict[str, str] = {}


def code_fingerprint(n_files: int = 400) -> str:
    """SHA-256 over the source of ``src/planlab`` and the experiment scripts.

    The checkout is not a git repository, so the manifest records a content
    fingerprint of the code instead of a commit id.
    """
    if "value" in _CODE_FP:
        return _CODE_FP["value"]
    import hashlib

    from planlab import paths

    root = paths.project_root()
    digest = hashlib.sha256()
    patterns = [root / "src" / "planlab"]
    found = 0
    for base in patterns:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if found >= n_files:
                break
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
            found += 1
    for extra in ("scripts/run_all.py", "scripts/reproduce.sh", "Makefile"):
        path = root / extra
        if path.exists():
            digest.update(extra.encode("utf-8"))
            digest.update(path.read_bytes())
    _CODE_FP["value"] = digest.hexdigest()[:16]
    return _CODE_FP["value"]


# ---------------------------------------------------------------------------
# technology resolution (per worker process)
# ---------------------------------------------------------------------------
_TECH_CACHE: dict[str, Any] = {}


def _tech_key(spec: Mapping[str, Any]) -> str:
    return json.dumps(dict(spec), sort_keys=True, separators=(",", ":"))


def load_cell_technology(spec: Mapping[str, Any]) -> Any:
    """Resolve a cell's technology spec, applying the A-perturbation if any.

    The spec is the scenario ``technology`` mapping plus two experiment-only keys:

    * ``a_perturb`` — log-scale standard deviation of a multiplicative perturbation
      of ``A`` (``docs/PROTOCOL.md`` §7 validity-threat mitigation);
    * ``perturb_seed`` — which draw of that perturbation.
    """
    key = _tech_key(spec)
    cached = _TECH_CACHE.get(key)
    if cached is not None:
        return cached
    from planlab.core.scenarios import resolve_technology_spec

    base_spec = {k: v for k, v in dict(spec).items() if k not in {"a_perturb", "perturb_seed"}}
    tech = resolve_technology_spec(base_spec)
    magnitude = float(spec.get("a_perturb", 0.0) or 0.0)
    if magnitude > 0:
        from .grid import perturb_technology

        tech = perturb_technology(tech, magnitude=magnitude, seed=int(spec.get("perturb_seed", 0)))
    _TECH_CACHE[key] = tech
    return tech


# ---------------------------------------------------------------------------
# probe injection
# ---------------------------------------------------------------------------
def _apply_probe(sim: Any, mech: Any, probe: Mapping[str, Any] | None, tech: Any) -> None:
    """Inject a probe instrument into a fresh mechanism/simulation.

    Probes exist only to *measure the local gain* ``b`` (``docs/notes_delay_theorem.md``
    §5.1); they are never part of a treatment cell.  They are applied through the
    public objects the engine exposes (``mechanism.targets`` and ``simulation.state``)
    and therefore cannot leak into a treatment run.

    * ``target_offset``: shift the carried plan target of one sector by ``delta``
      (the ``b_T`` measurement).  Requires a mechanism that reads ``self.targets``.
    * ``price_offset``: shift the initial price of one sector by ``delta`` (the
      ``b_p`` measurement).
    """
    if not probe:
        return
    kind = str(probe.get("kind", ""))
    sectors = list(getattr(tech, "sectors", ()))
    sector = probe.get("sector")
    delta = float(probe.get("delta", 0.0))
    if kind == "target_offset":
        if sector is None:
            targets = np.asarray(mech.targets, dtype=float) * (1.0 + delta)
        else:
            idx = sectors.index(str(sector)) if isinstance(sector, str) else int(sector)
            targets = np.asarray(mech.targets, dtype=float).copy()
            targets[idx] *= 1.0 + delta
        mech.targets = targets
    elif kind == "price_offset":
        state = sim.state
        price = np.asarray(state.price, dtype=float).copy()
        if sector is None:
            price = price * (1.0 + delta)
        else:
            idx = sectors.index(str(sector)) if isinstance(sector, str) else int(sector)
            price[idx] *= 1.0 + delta
        state.price = price
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown probe kind {kind!r}")


# ---------------------------------------------------------------------------
# instrumentation: does the planning step lower any target?
# ---------------------------------------------------------------------------
def _instrument_plan_lowering(mech: Any) -> dict[str, Any]:
    """Wrap ``form_plan`` and count the periods in which it lowers a target.

    The Lead's diagnostic for the material-balance decomposition: the balancing
    pass is a *downward-only* filter on the plan, so the relevant object is whether
    the targets that come out of ``form_plan`` are below the ones that went in.
    The wrapper only reads ``mech.targets`` before and after the call; it never
    changes behaviour.
    """
    stats: dict[str, Any] = {"lowered": 0, "raised": 0, "periods": 0}
    original = mech.form_plan

    def wrapped(obs: Any, state: Any) -> Any:
        before = np.asarray(getattr(mech, "targets", np.zeros(0)), dtype=float).copy()
        plan = original(obs, state)
        after = np.asarray(getattr(mech, "targets", np.zeros(0)), dtype=float)
        stats["periods"] += 1
        if before.size and before.size == after.size:
            if bool(np.any(after < before - 1e-12)):
                stats["lowered"] += 1
            if bool(np.any(after > before + 1e-12)):
                stats["raised"] += 1
        return plan

    mech.form_plan = wrapped  # type: ignore[method-assign]
    return stats


# ---------------------------------------------------------------------------
# per-run extraction
# ---------------------------------------------------------------------------
def _own_recovery(
    short: np.ndarray, t: np.ndarray, *, shock_start: int, baseline: float, fraction: float = 0.5
) -> float:
    """Periods after the peak until the shortage closes ``fraction`` of the gap.

    Independent re-computation of ``run_metrics['recovery_half_life']`` from the
    period series (``docs/DEVIATIONS.md`` §2): the core routine is the reference,
    this is the cross-check.  Returns ``NaN`` if the gap never closes inside the
    post-burn-in window.
    """
    window = np.asarray(short, dtype=float)[np.asarray(t) >= int(shock_start)]
    if window.size == 0:
        return float("nan")
    peak_idx = int(np.argmax(window))
    peak = float(window[peak_idx])
    if peak <= baseline:
        return 0.0
    target = baseline + (1.0 - fraction) * (peak - baseline)
    after = window[peak_idx:]
    idx = np.where(after <= target)[0]
    return float(idx[0]) if idx.size else float("nan")


def _shock_sector_index(shock: Any, tech: Any) -> int | None:
    """Resolve a shock's sector (int index or name) against the technology."""
    sector = getattr(shock, "sector", None)
    if sector is None:
        return None
    if isinstance(sector, (int, np.integer)):
        return int(sector)
    name = str(sector)
    try:
        return list(tech.sectors).index(name)
    except ValueError:
        return None


def _accounting_residual(record: Any) -> float:
    """Relative violation of the per-sector balance identity for one period.

    Delegates to :func:`planlab.core.engine.balance_residual`, the repository's
    single canonical definition (it includes the spoilage term).  The engine raises
    on a violation, so this is a *recorded* diagnostic for the calibration table,
    not a check that can silently fail.  The fallback exists only so that the
    experiment layer still imports against an older core.
    """
    try:
        from planlab.core.engine import balance_residual

        return float(balance_residual(record))
    except Exception:  # pragma: no cover - pre-spoilage core
        supply = np.asarray(record.supply, dtype=float)
        rhs = (
            np.asarray(record.Z, dtype=float).sum(axis=1)
            + np.asarray(record.C, dtype=float)
            + np.asarray(record.G, dtype=float)
            + np.asarray(record.invest, dtype=float)
            + np.asarray(record.inventory_end, dtype=float)
        )
        scale = max(1.0, float(np.max(np.abs(supply))))
        return float(np.max(np.abs(supply - rhs)) / scale)


def _period_rows(res: Any, cell: Cell, seed: int) -> list[dict[str, Any]]:
    """One row per period: every period metric plus the cell's treatment columns."""
    treatment = dict(cell.treatment)
    base_total = max(float(np.sum(np.asarray(res.technology.base_output, dtype=float))), 1e-12)
    rows: list[dict[str, Any]] = []
    for record in res.records:
        row: dict[str, Any] = {
            "block": cell.block,
            "cell_id": cell.cell_id,
            "mechanism": cell.mechanism,
            "seed": int(seed),
            "t": int(record.t),
        }
        row.update(treatment)
        row.update({k: float(v) for k, v in record.metrics.items()})
        row["accounting_residual_rel"] = _accounting_residual(record)
        row["x_ratio"] = float(np.sum(np.asarray(record.x, dtype=float))) / base_total
        rows.append(row)
    return rows


def _sector_rows(
    res: Any, cell: Cell, seed: int, tech: Any, scenario: Any
) -> list[dict[str, Any]]:
    """Run-level per-sector panel, including the incidence of the shortage.

    The engine serves intermediate inputs *before* final claims, so the interesting
    object is not the aggregate shortage but **who bears it** ("who bears the
    shortage is not who causes it").  For every good ``i`` this panel separates

    ``unmet_final``
        household demand that was not delivered (``demand_i - C_i``);
    ``unmet_other``
        government + investment claims not delivered, reconstructed from the
        engine's own claim formulas (``gov_share`` of base final demand and
        replacement investment), because the engine records only the grants;
    ``plan_gap_inter``
        intermediates the *plan* asked for but production did not use, measured
        against the plan: ``max(0, A_i (targets - x))``.  This is **not** rationing
        incidence — intermediates are served first and always receive ``A_i x`` —
        it is the cost the network pays in reduced production.  Reported separately
        as ``intermediate_service_rate`` so the two channels are never confused.

    Incidence shares are therefore computed over *rationed* claims only
    (households versus government/investment), and ``unmet_share_of_total`` is the
    share of the aggregate household shortfall borne by sector ``i``'s consumers.
    """
    records = [r for r in res.records if r.t > int(scenario.burn_in)]
    if not records:
        records = list(res.records)
    n = len(tech.sectors)
    A = np.asarray(tech.A, dtype=float)
    base = np.asarray(tech.base_output, dtype=float)
    base_total = float(base.sum())
    gov_base = float(scenario.gov_share) * np.asarray(tech.base_final, dtype=float)
    invest_total = float(scenario.depreciation) * float(np.dot(tech.kappa, base))
    invest_base = base / base_total * invest_total
    shock_sector = None
    for shock in scenario.shocks:
        if shock.type == "capacity":
            idx = _shock_sector_index(shock, tech)
            if idx is not None:
                shock_sector = tech.sectors[idx]

    unmet_final = np.zeros(n)
    unmet_other = np.zeros(n)
    plan_gap_inter = np.zeros(n)
    demand_sum = np.zeros(n)
    demand_peak_rate = np.zeros(n)
    other_claim_sum = np.zeros(n)
    x_sum = np.zeros(n)
    x_min = np.full(n, np.inf)
    cap_sum = np.zeros(n)
    inv_sum = np.zeros(n)
    inter_required = np.zeros(n)
    inter_served = np.zeros(n)
    for record in records:
        x = np.asarray(record.x, dtype=float)
        demand = np.asarray(record.demand, dtype=float)
        final = np.maximum(np.asarray(record.unmet, dtype=float), 0.0)
        targets = np.asarray(record.plan_targets, dtype=float)
        intermediate = np.asarray(A @ x, dtype=float)
        planned_intermediate = np.asarray(A @ targets, dtype=float)
        other_claim = gov_base + invest_base
        other_granted = np.asarray(record.G, dtype=float) + np.asarray(record.invest, dtype=float)
        unmet_final += final
        unmet_other += np.maximum(other_claim - other_granted, 0.0)
        plan_gap_inter += np.maximum(planned_intermediate - intermediate, 0.0)
        demand_sum += demand
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(demand > 0, final / np.maximum(demand, 1e-300), 0.0)
        demand_peak_rate = np.maximum(demand_peak_rate, rate)
        other_claim_sum += other_claim
        x_sum += x
        x_min = np.minimum(x_min, x)
        cap_sum += np.asarray(record.capacity, dtype=float)
        inv_sum += np.asarray(record.inventory_end, dtype=float)
        inter_required += planned_intermediate
        inter_served += intermediate

    periods = float(len(records))
    demand_mean = demand_sum / periods
    total_household = float(unmet_final.sum())
    rows: list[dict[str, Any]] = []
    for j in range(n):
        rationed = unmet_final[j] + unmet_other[j]
        rows.append(
            {
                "block": cell.block,
                "cell_id": cell.cell_id,
                "mechanism": cell.mechanism,
                "seed": int(seed),
                "sector": tech.sectors[j],
                "is_shocked": bool(shock_sector is not None and tech.sectors[j] == shock_sector),
                "demand_mean": float(demand_mean[j]),
                "demand_share": float(demand_mean[j] / max(demand_mean.sum(), 1e-12)),
                "unmet_final_mean": float(unmet_final[j] / periods),
                "unmet_other_mean": float(unmet_other[j] / periods),
                "plan_gap_inter_mean": float(plan_gap_inter[j] / periods),
                "household_unmet_rate_mean": (
                    float(unmet_final[j] / demand_sum[j]) if demand_sum[j] > 0 else 0.0
                ),
                "household_unmet_rate_peak": float(demand_peak_rate[j]),
                "gov_invest_unmet_rate_mean": (
                    float(unmet_other[j] / other_claim_sum[j]) if other_claim_sum[j] > 0 else 0.0
                ),
                "final_share_of_rationed": (
                    float(unmet_final[j] / rationed) if rationed > 0 else float("nan")
                ),
                "other_share_of_rationed": (
                    float(unmet_other[j] / rationed) if rationed > 0 else float("nan")
                ),
                "unmet_share_of_total": (
                    float(unmet_final[j] / total_household) if total_household > 0 else float("nan")
                ),
                "intermediate_service_rate": (
                    float(inter_served[j] / inter_required[j]) if inter_required[j] > 0 else 1.0
                ),
                "x_ratio_mean": float(x_sum[j] / periods / max(base[j], 1e-12)),
                "x_ratio_min": float(x_min[j] / max(base[j], 1e-12)),
                "capacity_ratio_mean": float(cap_sum[j] / periods / max(base[j], 1e-12)),
                "inventory_ratio_mean": float(inv_sum[j] / periods / max(base[j], 1e-12)),
            }
        )
    return rows


def _protocol_amplification(
    res: Any, tech: Any, scenario: Any, cell: Cell
) -> tuple[float, float]:
    """Protocol §5 amplification: ``sum_i shortfall_i / direct capacity loss``.

    Numerator: peak (and mean) aggregate household shortfall *in value units* over
    the shock window.  Denominator: the capacity destroyed at the trough, valued at
    base-year output, ``sum_j (1 - size_j) * base_output_j``.  This is the
    like-for-like ratio the amendments call ``shock_output_amplification``'s
    complement: one measures lost production, the other unserved final demand.
    """
    base = np.asarray(tech.base_output, dtype=float)
    loss = 0.0
    for shock in scenario.shocks:
        if shock.type == "capacity":
            idx = _shock_sector_index(shock, tech)
            share = 1.0 - min(float(shock.size), 1.0)
            loss += share * (float(base.sum()) if idx is None else float(base[idx]))
    if loss <= 0:
        return 0.0, 0.0
    window = [r for r in res.records if r.t >= SHOCK_START]
    if not window:
        return 0.0, 0.0
    unmet = np.array([float(np.sum(r.unmet)) for r in window])
    return float(unmet.max() / loss), float(unmet.mean() / loss)


def _summary_row(
    res: Any,
    cell: Cell,
    seed: int,
    tech: Any,
    scenario: Any,
    wall_s: float,
    instrumentation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run-level statistics: everything the engine reports plus the cross-checks."""
    short = np.array([float(r.metrics.get("shortage_rate", 0.0)) for r in res.records])
    t = np.array([int(r.t) for r in res.records])
    used = t > int(scenario.burn_in)
    metrics = dict(res.run_metrics)
    base_total = max(float(np.sum(np.asarray(tech.base_output, dtype=float))), 1e-12)
    amp_peak, amp_mean = _protocol_amplification(res, tech, scenario, cell)
    first_start = min((s.start for s in scenario.shocks), default=0)
    pre = short[(t < first_start) & used]
    baseline = float(np.mean(pre)) if pre.size else float(metrics.get("pre_shock_shortage", 0.0))
    rows: dict[str, Any] = {
        "block": cell.block,
        "cell_id": cell.cell_id,
        "mechanism": cell.mechanism,
        "seed": int(seed),
        "scenario_id": cell.scenario_id,
    }
    rows.update(dict(cell.treatment))
    rows.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))})
    used_short = short[used]
    rows.update(
        {
            "n_periods_used": float(int(used.sum())),
            "output_ratio_mean": float(
                np.mean(
                    [
                        float(np.sum(r.x)) / max(float(np.sum(np.asarray(tech.base_output, dtype=float))), 1e-12)
                        for r, keep in zip(res.records, used)
                        if keep
                    ]
                )
            )
            if used.any()
            else float("nan"),
            "mean_shortage_window": float(np.mean(used_short)) if used_short.size else float("nan"),
            "peak_shortage_window": float(np.max(used_short)) if used_short.size else float("nan"),
            "pre_shock_shortage_own": baseline,
            "recovery_half_life_own": _own_recovery(
                short, t, shock_start=first_start, baseline=baseline
            ),
            "amp_protocol_peak": amp_peak,
            "amp_protocol_mean": amp_mean,
            "mean_investment_ratio": float(
                np.mean(
                    [
                        float(np.sum(np.asarray(r.invest, dtype=float))) / base_total
                        for r, keep in zip(res.records, used)
                        if keep
                    ]
                )
            )
            if used.any()
            else float("nan"),
            "mean_labour_utilisation": float(
                np.mean(
                    [
                        float(np.dot(np.asarray(tech.labor, dtype=float), np.asarray(r.x, dtype=float)))
                        / max(
                            float(np.dot(np.asarray(tech.labor, dtype=float), np.asarray(tech.base_output, dtype=float)))
                            * float(scenario.labor_supply_factor),
                            1e-12,
                        )
                        for r, keep in zip(res.records, used)
                        if keep
                    ]
                )
            )
            if used.any()
            else float("nan"),
            "capital_ratio_final": float(
                np.sum(np.asarray(res.records[-1].capacity, dtype=float) * np.asarray(tech.kappa, dtype=float))
                / max(float(np.sum(np.asarray(tech.kappa, dtype=float) * np.asarray(tech.base_output, dtype=float))), 1e-12)
            )
            if res.records
            else float("nan"),
            "capital_ratio_mean": float(
                np.mean(
                    [
                        float(np.sum(np.asarray(r.capacity, dtype=float) * np.asarray(tech.kappa, dtype=float)))
                        / max(float(np.sum(np.asarray(tech.kappa, dtype=float) * np.asarray(tech.base_output, dtype=float))), 1e-12)
                        for r, keep in zip(res.records, used)
                        if keep
                    ]
                )
            )
            if used.any()
            else float("nan"),
            "accounting_residual_max": float(
                max((_accounting_residual(r) for r in res.records), default=0.0)
            ),
            "accounting_checks_passed": float(len(res.records)),
            "plan_lowering_periods": float(instrumentation.get("lowered", 0)) if instrumentation else float("nan"),
            "plan_raising_periods": float(instrumentation.get("raised", 0)) if instrumentation else float("nan"),
            "plan_instrumented_periods": float(instrumentation.get("periods", 0)) if instrumentation else float("nan"),
            "wall_s": float(wall_s),
            "status": "ok",
        }
    )
    # probe columns: the last quarter of the run is the probe's steady state
    tail = used & (t >= int(t.max()) - 19)
    if tail.any():
        rows["probe_shortage_tail"] = float(np.mean(short[tail]))
        demand = np.array([float(r.metrics.get("demand_total", 0.0)) for r in res.records])
        if demand[tail].sum() > 0:
            unmet = np.array([float(np.sum(r.unmet)) for r in res.records])
            rows["probe_unmet_tail"] = float(unmet[tail].mean())
    return rows


# ---------------------------------------------------------------------------
# audit sample (the verifier's raw-array check)
# ---------------------------------------------------------------------------
_AUDIT_ARRAYS: tuple[str, ...] = (
    "x",
    "Z",
    "C",
    "G",
    "invest",
    "inventory_end",
    "demand",
    "unmet",
    "capacity",
    "plan_targets",
    "supply",
    "price",
    "spoilage",
)


def _audit_periods(cell: Cell) -> list[int]:
    """Which periods of one run enter the audit sample (deterministic)."""
    periods = int(cell.scenario.get("periods", PERIODS))
    burn = int(cell.scenario.get("burn_in", BURN_IN))
    shocks = cell.scenario.get("shocks") or []
    ts: list[int] = []
    if shocks:
        start = int(min(s["start"] for s in shocks))
        end = int(max(int(s["start"]) + max(int(s["duration"]), 1) for s in shocks))
        ts = [burn + 1, max(burn + 1, start - 1), start + 1, min(periods, end + 2), periods]
    else:
        span = max(1, periods - burn)
        ts = [burn + 1, burn + span // 3, burn + 2 * span // 3, periods]
    return sorted({int(t) for t in ts if 1 <= int(t) <= periods})


def _audit_bucket(t: int, cell: Cell) -> str:
    shocks = cell.scenario.get("shocks") or []
    if not shocks:
        return "steady"
    start = int(min(s["start"] for s in shocks))
    end = int(max(int(s["start"]) + max(int(s["duration"]), 1) for s in shocks))
    burn = int(cell.scenario.get("burn_in", BURN_IN))
    if t < start:
        return "pre_shock" if t > burn else "burn_in"
    if t < end:
        return "shock"
    return "post_shock"


def audit_plan(cells: Sequence[Cell], target_records: int = 240) -> dict[str, dict[str, list[int]]]:
    """Stratified sample of (run, period) records to persist for the verifier.

    The sample is deterministic: runs are taken at a fixed stride through the
    block's (cell, seed) list and each selected run contributes a few periods
    chosen by :func:`_audit_periods` (pre-shock, in-shock, post-shock, first and
    last period).  That gives every block at least ``target_records`` records
    across its cells, periods and phases, without dumping the whole trajectory.
    """
    pairs: list[tuple[Cell, int]] = [(cell, int(seed)) for cell in cells for seed in cell.seeds]
    if not pairs:
        return {}
    per_run = max(1, len(_audit_periods(pairs[0][0])))
    n_runs = max(1, int(target_records) // per_run)
    step = max(1, len(pairs) // n_runs)
    chosen = pairs[::step][:n_runs]
    plan: dict[str, dict[str, list[int]]] = {}
    for cell, seed in chosen:
        plan.setdefault(cell.cell_id, {})[str(seed)] = _audit_periods(cell)
    return plan


def _append_audit(
    bucket: dict[str, list[Any]], cell: Cell, seed: int, record: Any, tech: Any
) -> None:
    """Append one (run, period) record to the audit sample.

    The substrate is stored *per record* (``A``, ``base_output``, ``base_final``,
    ``essentials``, ``consumption_weights``) so that the file is self-auditing even
    for a block that mixes substrates or A-perturbations (B6): the verifier can
    check ``Z == A diag(x)`` and the base-year identity without loading the data
    layer or trusting the runner.
    """
    bucket["seed"].append(int(seed))
    bucket["t"].append(int(record.t))
    bucket["bucket"].append(_audit_bucket(int(record.t), cell))
    bucket["cell_id"].append(cell.cell_id)
    bucket["mechanism"].append(cell.mechanism)
    bucket["scenario_id"].append(cell.scenario_id)
    bucket["block"].append(cell.block)
    bucket["n_sectors"].append(int(tech.n))
    for name in _AUDIT_ARRAYS:
        bucket[name].append(np.asarray(getattr(record, name), dtype=float).copy())
    bucket.setdefault("spoilage", [])
    bucket["A"].append(np.asarray(tech.A, dtype=float).copy())
    bucket["sectors"].append(np.asarray(tech.sectors, dtype=str))
    bucket["base_output"].append(np.asarray(tech.base_output, dtype=float).copy())
    bucket["base_final"].append(np.asarray(tech.base_final, dtype=float).copy())
    bucket["essentials"].append(np.asarray(tech.essentials, dtype=float).copy())
    bucket["consumption_weights"].append(
        np.asarray(tech.consumption_weights, dtype=float).copy()
    )


def _treatment_constants(cell: Cell) -> dict[str, Any]:
    """Scalar treatment constants stored once per audit file."""
    scenario = cell.scenario
    return {
        "periods": np.asarray(int(scenario.get("periods", PERIODS))),
        "burn_in": np.asarray(int(scenario.get("burn_in", BURN_IN))),
        "lag": np.asarray(int(scenario.get("info", {}).get("lag", 0))),
        "noise": np.asarray(float(scenario.get("info", {}).get("noise", 0.0))),
        "sigma_demand": np.asarray(float(scenario.get("demand", {}).get("sigma", 0.0))),
        "ar1": np.asarray(float(scenario.get("demand", {}).get("ar1", 0.0))),
        "price_elasticity": np.asarray(
            float(scenario.get("demand", {}).get("price_elasticity", 0.0))
        ),
        "stock_ratio": np.asarray(float(scenario.get("stock_ratio", STOCK_RATIO))),
        "a_perturb": np.asarray(float(cell.tech.get("a_perturb", 0.0) or 0.0)),
        "shock_size": np.asarray(
            float(max((s.get("size", 1.0) for s in scenario.get("shocks") or []), default=1.0))
        ),
        "shock_start": np.asarray(
            int(min((s.get("start", 0) for s in scenario.get("shocks") or []), default=0))
        ),
        "shock_duration": np.asarray(
            int(min((s.get("duration", 0) for s in scenario.get("shocks") or []), default=0))
        ),
    }


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------
def _execute_cell(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run every seed of one cell.  Module-level so that it pickles.

    Returns a plain dict (never raises): a failure is data, and the parent records
    it in ``failures.json`` so that the cell is retried on the next invocation.
    """
    cell = cell_from_dict(payload)
    started = time.perf_counter()
    out: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "block": cell.block,
        "mechanism": cell.mechanism,
        "cell_hash": cell.hash(),
        "status": "ok",
        "error": "",
        "period_rows": [],
        "summary_rows": [],
        "sector_rows": [],
        "audit": {name: [] for name in _AUDIT_ARRAYS}
        | {
            "seed": [], "t": [], "bucket": [], "cell_id": [], "mechanism": [],
            "scenario_id": [], "block": [], "n_sectors": [], "A": [], "sectors": [], "base_output": [],
            "base_final": [], "essentials": [], "consumption_weights": [],
        },
        "audit_constants": None,
        "code_fingerprint": code_fingerprint(),
        "wall_s": 0.0,
    }
    try:
        from planlab.core.engine import Simulation, run_scenario  # noqa: PLC0415
        from planlab.core.scenarios import scenario_from_dict  # noqa: PLC0415
        from planlab.mechanisms import get_mechanism  # noqa: PLC0415

        tech = load_cell_technology(cell.tech)
        scenario = scenario_from_dict(cell.scenario)
        for seed in cell.seeds:
            t0 = time.perf_counter()
            mech = get_mechanism(cell.mechanism, tech, scenario.mechanism_params)
            sim = Simulation(tech, mech, scenario, seed=int(seed))
            instrumentation = _instrument_plan_lowering(mech)
            if cell.probe:
                _apply_probe(sim, mech, cell.probe, tech)
            res = sim.run()
            wall = time.perf_counter() - t0
            out["period_rows"].extend(_period_rows(res, cell, int(seed)))
            out["summary_rows"].append(
                _summary_row(res, cell, int(seed), tech, scenario, wall, instrumentation)
            )
            out["sector_rows"].extend(_sector_rows(res, cell, int(seed), tech, scenario))
            audit_ts = (payload.get("audit") or {}).get(str(seed)) or []
            if audit_ts:
                wanted = {int(t) for t in audit_ts}
                for record in res.records:
                    if int(record.t) in wanted:
                        _append_audit(out["audit"], cell, int(seed), record, tech)
                if out["audit_constants"] is None:
                    out["audit_constants"] = _treatment_constants(cell)
    except Exception as exc:  # noqa: BLE001 - a failed cell is recorded, not raised
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_s"] = time.perf_counter() - started
    return out


# ---------------------------------------------------------------------------
# shard I/O
# ---------------------------------------------------------------------------
def _shard_paths(out_dir: Path, cell_id: str) -> tuple[Path, Path]:
    return out_dir / "cells" / f"{cell_id}.json", out_dir / "cells" / f"{cell_id}.parquet"


def is_complete(out_dir: str | Path, cell: Cell) -> bool:
    """True if a valid shard for ``cell`` (same hash and seeds) is on disk."""
    meta_path, data_path = _shard_paths(Path(out_dir), cell.cell_id)
    if not (meta_path.exists() and data_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - corrupt shard
        return False
    return (
        meta.get("status") == "ok"
        and meta.get("cell_hash") == cell.hash()
        and list(meta.get("seeds", [])) == list(cell.seeds)
    )


def _write_shard(out_dir: Path, cell: Cell, result: Mapping[str, Any]) -> None:
    """Persist one cell: JSON metadata (summary + sector rows) and a parquet shard."""
    meta_path, data_path = _shard_paths(out_dir, cell.cell_id)
    meta = {
        "cell_id": cell.cell_id,
        "block": cell.block,
        "mechanism": cell.mechanism,
        "cell_hash": cell.hash(),
        "code_fingerprint": result.get("code_fingerprint", ""),
        "seeds": list(cell.seeds),
        "status": result.get("status", "ok"),
        "error": result.get("error", ""),
        "wall_s": float(result.get("wall_s", 0.0)),
        "treatment": dict(cell.treatment),
        "scenario": cell.scenario_dict(),
        "tech": dict(cell.tech),
        "probe": None if cell.probe is None else dict(cell.probe),
        "summary_rows": list(result.get("summary_rows", [])),
        "sector_rows": list(result.get("sector_rows", [])),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_default), encoding="utf-8")
    period_rows = result.get("period_rows") or []
    if period_rows:
        pd.DataFrame(period_rows).to_parquet(data_path, index=False)
    elif data_path.exists():  # pragma: no cover - a cell that produced nothing
        data_path.unlink()
    audit = result.get("audit") or {}
    const = result.get("audit_constants")
    if const and audit.get("x"):
        payload = dict(const)
        payload.update(
            {
                "seed": np.asarray(audit["seed"], dtype=np.int64),
                "t": np.asarray(audit["t"], dtype=np.int64),
                "bucket": np.asarray(audit["bucket"], dtype=str),
                "block": np.asarray(audit["block"], dtype=str),
                "cell_id": np.asarray(audit["cell_id"], dtype=str),
                "mechanism": np.asarray(audit["mechanism"], dtype=str),
                "scenario_id": np.asarray(audit["scenario_id"], dtype=str),
            }
        )
        for name in _AUDIT_ARRAYS:
            payload[name] = np.asarray(audit[name], dtype=float)
        payload["A"] = np.asarray(audit["A"], dtype=float)          # (m, n, n)
        payload["sectors"] = np.asarray(audit["sectors"], dtype=str)  # (m, n)
        for name in ("base_output", "base_final", "essentials", "consumption_weights"):
            payload[name] = np.asarray(audit[name], dtype=float)
        payload["n_sectors"] = np.asarray(audit["n_sectors"], dtype=np.int64)
        np.savez_compressed(out_dir / "cells" / f"{cell.cell_id}.audit.npz", **payload)


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# block driver
# ---------------------------------------------------------------------------
@dataclass
class BlockResult:
    """What ``run_block`` reports back to ``scripts/run_all.py``."""

    block: str
    cells_planned: int
    cells_run: int
    cells_skipped: int
    cells_failed: int
    runs_completed: int
    wall_s: float
    mechanisms_present: list[str]
    mechanisms_missing: list[str]
    manifest_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "cells_planned": self.cells_planned,
            "cells_run": self.cells_run,
            "cells_skipped_existing": self.cells_skipped,
            "cells_failed": self.cells_failed,
            "runs_completed": self.runs_completed,
            "wall_s": round(self.wall_s, 2),
            "mechanisms_present": self.mechanisms_present,
            "mechanisms_missing": self.mechanisms_missing,
            "manifest": self.manifest_path,
        }


def _mp_context():
    """Process context for the worker pool.

    ``fork`` is preferred where available: it avoids re-importing the whole stack in
    every worker (a second of start-up on this checkout) and it keeps the runner
    usable from a ``python -`` session or a notebook.  Every run seeds its own RNG,
    so no state is inherited that could make results depend on the parent.
    """
    import multiprocessing as mp

    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX platforms
        return mp.get_context()


def _n_workers(workers: int | None) -> int:
    if workers is None:
        cpu = os.cpu_count() or 2
        return max(1, cpu - 1)
    return max(1, int(workers))


def run_cells(
    cells: Sequence[Cell],
    out_dir: str | Path,
    *,
    workers: int | None = None,
    force: bool = False,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Execute cells into ``out_dir`` and return (results, n_skipped, n_failed).

    This is the low-level entry point used by tests and by :func:`run_block`; it
    knows nothing about blocks beyond what the cells carry.
    """
    out = Path(out_dir)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    todo: list[Cell] = []
    skipped = 0
    for cell in cells:
        if not force and is_complete(out, cell):
            skipped += 1
            continue
        todo.append(cell)
    results: list[dict[str, Any]] = []
    if not todo:
        return results, skipped, 0
    plan = audit_plan(todo)
    n = _n_workers(workers)
    if n == 1 or len(todo) == 1:
        for cell in todo:
            if verbose:
                print(f"  [{cell.block}] {cell.cell_id} ({len(cell.seeds)} seeds)", flush=True)
            result = _execute_cell(_payload(cell, plan.get(cell.cell_id)))
            results.append(result)
            _write_shard(out, cell, result)
        return results, skipped, sum(1 for r in results if r["status"] != "ok")
    with ProcessPoolExecutor(max_workers=min(n, len(todo)), mp_context=_mp_context()) as pool:
        futures = {
            pool.submit(_execute_cell, _payload(cell, plan.get(cell.cell_id))): cell
            for cell in todo
        }
        done = 0
        for future in as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - worker crash
                result = {
                    "cell_id": cell.cell_id,
                    "block": cell.block,
                    "mechanism": cell.mechanism,
                    "cell_hash": cell.hash(),
                    "status": "error",
                    "error": f"worker crashed: {type(exc).__name__}: {exc}",
                    "period_rows": [],
                    "summary_rows": [],
                    "sector_rows": [],
                    "wall_s": 0.0,
                }
            results.append(result)
            _write_shard(out, cell, result)
            done += 1
            if verbose and (done % 10 == 0 or done == len(todo)):
                print(f"  [{cell.block}] {done}/{len(todo)} cells done", flush=True)
    failed = sum(1 for r in results if r["status"] != "ok")
    return results, skipped, failed


def _payload(cell: Cell, audit: Mapping[str, list[int]] | None = None) -> dict[str, Any]:
    return {
        "audit": dict(audit or {}),
        "block": cell.block,
        "cell_id": cell.cell_id,
        "mechanism": cell.mechanism,
        "scenario": dict(cell.scenario),
        "seeds": list(cell.seeds),
        "treatment": dict(cell.treatment),
        "tech": dict(cell.tech),
        "probe": None if cell.probe is None else dict(cell.probe),
        "notes": cell.notes,
    }


def _assemble_block(
    block: str,
    out: Path,
    cells: Sequence[Cell],
    *,
    mechanisms_present: Sequence[str],
    mechanisms_missing: Sequence[str],
    skipped_cells: Sequence[Cell],
    wall_s: float,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    """Concatenate shards into the block-level artefacts and write the manifest."""
    summaries: list[dict[str, Any]] = []
    sectors: list[dict[str, Any]] = []
    period_files: list[Path] = []
    failures: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    fingerprints: dict[str, int] = {}
    scenario_hashes: dict[str, str] = {}
    runs_completed = 0
    for cell in cells:
        meta_path, data_path = _shard_paths(out, cell.cell_id)
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - corrupt shard
            failures.append({"cell_id": cell.cell_id, "error": f"unreadable shard: {exc}"})
            continue
        if meta.get("status") != "ok":
            failures.append({"cell_id": cell.cell_id, "error": meta.get("error", "unknown")})
            continue
        hashes[cell.cell_id] = meta.get("cell_hash", "")
        try:
            from planlab.core.scenarios import scenario_from_dict, scenario_hash

            scenario_hashes[cell.cell_id] = scenario_hash(
                scenario_from_dict(cell.scenario_dict())
            )
        except Exception:  # pragma: no cover - invalid scenario never gets this far
            scenario_hashes[cell.cell_id] = ""
        fp = str(meta.get("code_fingerprint") or "")
        fingerprints[fp] = fingerprints.get(fp, 0) + 1
        summaries.extend(meta.get("summary_rows") or [])
        sectors.extend(meta.get("sector_rows") or [])
        runs_completed += len(meta.get("summary_rows") or [])
        if data_path.exists():
            period_files.append(data_path)

    treatment_keys = (
        sorted(set().union(*[set(c.treatment.keys()) for c in cells])) if cells else []
    )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(["cell_id", "seed"]).reset_index(drop=True)
        summary.to_csv(out / "summary.csv", index=False)
        aggregate = _aggregate_cells(summary, treatment_keys)
        aggregate.to_csv(out / "aggregate.csv", index=False)
    else:  # pragma: no cover - a block with no successful cell
        summary.to_csv(out / "summary.csv", index=False)
        pd.DataFrame(
            columns=["block", "cell_id", "mechanism", "n_seeds", "mean_shortage_rate_mean"]
        ).to_csv(out / "aggregate.csv", index=False)

    if period_files:
        frames = [pd.read_parquet(path) for path in period_files]
        runs = pd.concat(frames, ignore_index=True)
        runs = runs.sort_values(["cell_id", "seed", "t"]).reset_index(drop=True)
        runs.to_parquet(out / "runs.parquet", index=False)
    else:  # pragma: no cover
        pd.DataFrame().to_parquet(out / "runs.parquet", index=False)

    if sectors:
        panel = pd.DataFrame(sectors).sort_values(["cell_id", "seed", "sector"]).reset_index(drop=True)
        panel.to_parquet(out / "sector_panel.parquet", index=False)

    audit_info = _assemble_audit(out)

    (out / "cells.json").write_text(
        json.dumps([cell.full_dict() for cell in cells], indent=2, sort_keys=True, default=_default),
        encoding="utf-8",
    )

    identification = _identification_audit(cells)
    manifest = {
        "block": block,
        "mechanism_base_params": dict(identification["common_params"]),
        "mechanism_pinned_params": dict(identification["pinned_params"]),
        "identification_audit": identification,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package_versions": package_versions(),
        "code_fingerprint": code_fingerprint(),
        "code_fingerprints_seen": fingerprints,
        "mixed_code_versions": len([k for k in fingerprints if k]) > 1,
        **_git_state(),
        "environment": dict(BLOCK_ENVIRONMENT),
        "mechanisms_present": sorted(mechanisms_present),
        "mechanisms_missing": sorted(mechanisms_missing),
        "mechanisms_present_in_runs": sorted({c.mechanism for c in cells if hashes.get(c.cell_id)}),
        "cells_planned": len(cells),
        "cells_completed": len(hashes),
        "cells_skipped_missing_mechanism": _skipped_missing_count(skipped_cells, mechanisms_present),
        "cells_failed": len(failures),
        "runs_completed": int(runs_completed),
        "runs_planned": int(sum(len(c.seeds) for c in cells)),
        "seeds_per_cell": {cell.cell_id: len(cell.seeds) for cell in cells},
        "scenario_hashes": scenario_hashes,
        "cell_hashes": hashes,
        "scenario_ids": sorted({cell.scenario_id for cell in cells}),
        "grid_fingerprint": grid_fingerprint_cells(cells),
        "workers": int(workers),
        "wall_time_s": round(float(wall_s), 3),
        "sum_cell_wall_time_s": round(float(sum(v.get("wall_s", 0.0) for v in summaries)), 3),
        "forced_rerun": bool(force),
        "substrates": sorted({str(c.tech.get("source", "eurostat")) for c in cells}),
        "audit_records": audit_info.get("n_records", 0),
        "audit_runs": audit_info.get("n_runs", 0),
        "audit_file": audit_info.get("path"),
        "failures": failures,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_default), encoding="utf-8"
    )
    if failures:
        (out / "failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
        )
    return manifest


def _identification_audit(cells: Sequence[Cell]) -> dict[str, Any]:
    """Check that every cell received the common mechanism-parameter block.

    The mechanism comparison is only identified if the household-side and
    planning-side settings are *identical* across institutions (Lead, identification
    review).  This audits the cell definitions, not the mechanism code: a cell whose
    ``mechanism_params`` does not contain the common block is a design error and
    raises here rather than producing an unidentified ranking.
    """
    from .grid import COMMON_MECHANISM_PARAMS, MECHANISM_BASE_PARAMS

    problems: list[str] = []
    checked = 0
    for cell in cells:
        params = dict(cell.scenario.get("mechanism_params") or {})
        checked += 1
        for key, expected in COMMON_MECHANISM_PARAMS.items():
            if key not in params:
                problems.append(f"{cell.cell_id}: missing {key}")
                continue
            value = params[key]
            if isinstance(expected, float):
                if not np.isclose(float(value), expected):
                    problems.append(f"{cell.cell_id}: {key}={value} != {expected}")
            elif value != expected:
                problems.append(f"{cell.cell_id}: {key}={value} != {expected}")
        # Knobs a block deliberately sweeps (the gain axis, theta, balance rounds)
        # must be *present* but are allowed to differ from the pinned reference
        # operating point inside the cells that sweep them; everything else must
        # equal the pin exactly, or the mechanism is not the mechanism.
        swept = {"revision_gain", "price_gain", "theta", "balance_rounds"}
        for key, expected in MECHANISM_BASE_PARAMS.get(cell.mechanism, {}).items():
            if key not in params:
                problems.append(f"{cell.cell_id}: missing pinned {key}")
            elif key in swept:
                continue
            elif isinstance(expected, float):
                if not np.isclose(float(params[key]), expected):
                    problems.append(f"{cell.cell_id}: {key}={params[key]} != {expected}")
    if problems:
        raise AssertionError(
            "identification audit failed (mechanism parameters differ between cells): "
            + "; ".join(problems[:10])
        )
    return {
        "common_params": dict(COMMON_MECHANISM_PARAMS),
        "pinned_params": {k: dict(v) for k, v in MECHANISM_BASE_PARAMS.items()},
        "cells_checked": checked,
        "cells_compliant": checked,
        "ok": True,
        "institutional_knobs": ["revision_gain", "price_gain", "theta", "quota_speed"],
    }


def _assemble_audit(out: Path) -> dict[str, Any]:
    """Concatenate the per-cell audit shards into ``audit_records.npz``.

    The verifier gets raw per-period arrays for a stratified sample of (run,
    period) records, together with the substrate of *each* record, so accounting
    conservation, ``Z == A diag(x)``, ``C <= demand``, ``x <= capacity``, the
    priority-ladder order and the metric formulas can all be re-derived without
    re-running the simulation or trusting this code.

    A block may mix substrates (B1 runs the toy fixture and the real table), so
    array-valued records are padded to the largest sector count and ``n_sectors``
    is stored per record: only the first ``n_sectors[i]`` entries of record ``i``
    are meaningful.
    """
    files = sorted((out / "cells").glob("*.audit.npz"))
    if not files:
        return {"n_records": 0, "n_runs": 0, "path": None}
    columns: dict[str, list[np.ndarray]] = {}
    scalars: dict[str, np.ndarray] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                value = np.asarray(data[key])
                if value.ndim == 0:
                    scalars.setdefault(key, value)
                elif value.ndim == 1 and value.dtype.kind not in "f":
                    columns.setdefault(key, []).append(value)  # strings / ints per record
                elif value.ndim >= 2 and value.dtype.kind in "UiUS":
                    columns.setdefault(key, []).append(value)  # per-record sector labels
                elif value.dtype.kind in "UiUS":
                    scalars.setdefault(key, value)
                else:
                    columns.setdefault(key, []).append(value)
    payload: dict[str, Any] = {}
    for key, arrays in columns.items():
        if key == "A":
            payload[key] = _stack_padded(arrays)  # (N, n_max, n_max), NaN outside each record's n
        elif key == "sectors":
            width = max(a.shape[1] for a in arrays)
            padded = np.full((sum(a.shape[0] for a in arrays), width), "", dtype="<U16")
            row = 0
            for a in arrays:
                padded[row : row + a.shape[0], : a.shape[1]] = a
                row += a.shape[0]
            payload[key] = padded
        elif any(a.ndim > 1 for a in arrays):
            payload[key] = _stack_padded(arrays)
        else:
            payload[key] = np.concatenate([np.atleast_1d(a) for a in arrays], axis=0)
    payload.update(scalars)
    np.savez_compressed(out / "audit_records.npz", **payload)
    index = pd.DataFrame(
        {
            "cell_id": payload.get("cell_id", []),
            "mechanism": payload.get("mechanism", []),
            "seed": payload.get("seed", []),
            "t": payload.get("t", []),
            "bucket": payload.get("bucket", []),
        }
    )
    index.to_csv(out / "audit_index.csv", index=False)
    n_records = int(len(index))
    n_runs = (
        int(len({(str(c), int(s)) for c, s in zip(index["cell_id"], index["seed"])}))
        if n_records
        else 0
    )
    return {"n_records": n_records, "n_runs": n_runs, "path": str(out / "audit_records.npz")}


def _stack_padded(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate arrays along axis 0, padding trailing axes with NaN."""
    arrays = [np.atleast_2d(a) for a in arrays]
    ndim = max(a.ndim for a in arrays)
    target = [0] + [max(a.shape[i] for a in arrays) for i in range(1, ndim)]
    padded = []
    for arr in arrays:
        pads = [(0, 0)] + [
            (0, target[i] - arr.shape[i]) for i in range(1, arr.ndim)
        ]
        padded.append(np.pad(arr, pads, constant_values=np.nan))
    return np.concatenate(padded, axis=0)


def _skipped_missing_count(skipped_cells: Sequence[Cell], mechanisms_present: Sequence[str]) -> int:
    """Cells skipped because their mechanism is not registered."""
    present = set(mechanisms_present)
    return sum(1 for cell in skipped_cells if cell.mechanism not in present)


def grid_fingerprint_cells(cells: Sequence[Cell]) -> str:
    import hashlib

    items = sorted(cell.hash() for cell in cells)
    return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()[:16]


def _git_state() -> dict[str, Any]:
    """Commit id, dirty flag and diff stat of the checkout (all optional).

    The manifests must say which code made the numbers.  A dirty tree is recorded
    rather than hidden: a reader can then decide whether the diff matters.  When
    the checkout is not a git repository every field is ``None`` and the content
    fingerprint in the manifest is the only provenance.
    """
    state: dict[str, Any] = {"git_commit": None, "git_dirty": None, "git_diff_stat": None}
    try:
        import subprocess

        from planlab import paths

        root = str(paths.project_root())
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            timeout=5, check=False,
        )
        state["git_commit"] = head.stdout.strip() or None
        if state["git_commit"]:
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, capture_output=True,
                text=True, timeout=5, check=False,
            )
            state["git_dirty"] = bool(status.stdout.strip())
            diff = subprocess.run(
                ["git", "diff", "--stat", "HEAD"], cwd=root, capture_output=True,
                text=True, timeout=5, check=False,
            )
            lines = [line for line in diff.stdout.strip().splitlines() if line.strip()]
            state["git_diff_stat"] = lines[-1] if lines else ""
    except Exception:  # pragma: no cover - no git / no subprocess
        pass
    return state


def git_state() -> dict[str, Any]:
    """Public alias of :func:`_git_state` (used by the block manifest)."""
    return _git_state()


def _aggregate_cells(
    summary: pd.DataFrame, treatment_keys: Sequence[str] = ()
) -> pd.DataFrame:
    """Per-cell mean/sd/95% CI across seeds of the primary run-level metrics."""
    metrics = [
        "mean_shortage_rate",
        "max_shortage_rate",
        "mean_shortage_essentials",
        "cum_welfare",
        "mean_welfare",
        "mean_plan_fulfilment",
        "mean_plan_deviation",
        "mean_inventory_ratio",
        "mean_capital_utilisation",
        "mean_price_dispersion",
        "mean_price_volatility",
        "mean_iterations",
        "oscillation_amplitude",
        "oscillation_amplitude_sector_mean",
        "oscillation_period",
        "acf_lag1",
        "acf_lag2",
        "acf_lag3",
        "acf_lag4",
        "stability_index",
        "shock_peak_shortage",
        "shock_amplification",
        "shock_output_loss",
        "shock_output_amplification",
        "recovery_half_life",
        "recovery_half_life_own",
        "amp_protocol_peak",
        "amp_protocol_mean",
        "penalised_welfare",
        "mean_consumption",
        "mean_output",
        "output_ratio_mean",
        "accounting_residual_max",
        "mean_spoilage_rate",
        "mean_spoilage",
        "mean_investment_ratio",
        "capital_ratio_final",
        "capital_ratio_mean",
        "mean_labour_utilisation",
        "plan_lowering_periods",
        "plan_raising_periods",
        "accounting_checks_passed",
        "probe_shortage_tail",
        "probe_unmet_tail",
        "pre_shock_shortage",
        "n_periods_used",
        "wall_s",
    ]
    present = [m for m in metrics if m in summary.columns]
    group_cols = [c for c in (["block", "cell_id", "mechanism"] + list(treatment_keys)) if c in summary.columns]
    rows: list[dict[str, Any]] = []
    for keys, frame in summary.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(len(frame))
        for metric in present:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_sd"] = float("nan")
                row[f"{metric}_ci95_lo"] = float("nan")
                row[f"{metric}_ci95_hi"] = float("nan")
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
            half = 1.96 * sd / np.sqrt(values.size) if values.size > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_lo"] = mean - half
            row[f"{metric}_ci95_hi"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["block", "cell_id"]).reset_index(drop=True)


def assemble_block(
    block: str,
    *,
    results_dir: str | Path | None = None,
    cells: Sequence[Cell] | None = None,
    wall_s: float = 0.0,
    workers: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Rebuild a block's artefacts from the shards already on disk.

    Useful after adding shards by hand, or when only the aggregation changed: no
    simulation is performed.  ``run_block`` calls this internally.
    """
    planned = list(cells) if cells is not None else build_cells(block)
    present = available_mechanisms_safe()
    missing = sorted({c.mechanism for c in planned} - set(present))
    _runnable, skipped = split_runnable(planned, present)
    out = block_results_dir(block, results_dir)
    return _assemble_block(
        block,
        out,
        planned,
        mechanisms_present=present,
        mechanisms_missing=missing,
        skipped_cells=skipped,
        wall_s=wall_s,
        workers=workers,
        force=force,
    )


def run_block(
    block: str,
    *,
    workers: int | None = None,
    force: bool = False,
    results_dir: str | Path | None = None,
    verbose: bool = True,
    cells: Sequence[Cell] | None = None,
) -> BlockResult:
    """Run (or resume) one block and write its artefacts.

    Args:
        block: block id, one of ``planlab.experiments.grid.BLOCK_ORDER``.
        workers: process count; defaults to ``cpu_count() - 1``.
        force: re-run every cell even if a valid shard exists.
        results_dir: override for ``results/`` (tests use a temporary directory).
        verbose: print progress.
        cells: pre-built cells (tests); defaults to the frozen grid of ``block``.

    Returns:
        A :class:`BlockResult` summary.
    """
    planned = list(cells) if cells is not None else build_cells(block)
    present = available_mechanisms_safe()
    runnable, skipped = split_runnable(planned, present)
    missing = sorted({c.mechanism for c in planned} - set(present))
    if skipped and verbose:
        warnings.warn(
            f"[{block}] {len(skipped)} of {len(planned)} cells skipped: mechanisms not "
            f"registered yet ({', '.join(missing)}). Re-run after they land.",
            RuntimeWarning,
            stacklevel=2,
        )
    out = block_results_dir(block, results_dir)
    started = time.perf_counter()
    n_workers = _n_workers(workers)
    results, n_skipped, n_failed = run_cells(
        runnable, out, workers=n_workers, force=force, verbose=verbose
    )
    wall = time.perf_counter() - started
    manifest = _assemble_block(
        block,
        out,
        planned,
        mechanisms_present=present,
        mechanisms_missing=missing,
        skipped_cells=skipped,
        wall_s=wall,
        workers=n_workers,
        force=force,
    )
    if verbose:
        print(
            f"[{block}] cells {manifest['cells_completed']}/{manifest['cells_planned']}"
            f"  runs {manifest['runs_completed']}/{manifest['runs_planned']}"
            f"  failed {manifest['cells_failed']}"
            f"  skipped(missing mech) {manifest['cells_skipped_missing_mechanism']}"
            f"  {wall:.1f}s"
            + (f"  mechanisms present: {', '.join(present) or 'none'}" if present else "  NO MECHANISMS")
            + (f"  MISSING: {', '.join(missing)}" if missing else ""),
            flush=True,
        )
    return BlockResult(
        block=block,
        cells_planned=len(planned),
        cells_run=len(runnable),
        cells_skipped=n_skipped,
        cells_failed=n_failed,
        runs_completed=int(manifest["runs_completed"]),
        wall_s=wall,
        mechanisms_present=sorted(present),
        mechanisms_missing=missing,
        manifest_path=str(out / "manifest.json"),
    )
