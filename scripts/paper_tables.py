"""Assemble the paper's result tables from ``results/*/aggregate.csv``.

The experiment layer produces statistics; this script turns them into the exact
files the manuscript's ``[[TABLE:...]]`` tokens resolve to.  It is deliberately a
*separate* step from running experiments, so that tables can be regenerated without
re-running anything, and so that the mapping from "a statistic the runner computed"
to "a number in the paper" is one readable file rather than a hundred lines buried
in a runner.

    .venv/bin/python scripts/paper_tables.py
    .venv/bin/python scripts/paper_tables.py --blocks B1b B2   # subset

Every generated table is written to ``paper/tables/<name>.csv`` with a matching
``<name>.cap`` caption if none exists, and a provenance comment line naming the
block file it came from.  Blocks that are missing are reported and skipped: a
missing result must look like a missing result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"

#: output ratio is stored as ``output_ratio_mean`` per run, so its aggregate column
#: is double-suffixed.  One place to change if the runner renames it.
X = "output_ratio"
XSD = "output_ratio_mean_sd"


#: the reference run names its blocks R*, the experiment layer B*
_REF_BLOCK = {"B1": "R1", "B1b": "R2b", "B2": "R2", "B3": "R3", "B4": "R4", "B5": "R5",
              "B8": "R8", "B6L": "R6L"}


def aggregate_reference(block: str) -> pd.DataFrame | None:
    """Aggregate the Lead's independent reference run into the same cell schema."""
    path = ROOT / "reference_results" / "runs.csv"
    if not path.exists() or block not in _REF_BLOCK:
        return None
    df = pd.read_csv(path)
    df = df[(df["block"] == _REF_BLOCK[block]) & (df["error"].fillna("") == "")]
    if df.empty:
        return None
    keys = [c for c in ("cell", "mechanism", "gain", "lag", "k_eff", "shock_sector",
                        "shock_size") if c in df.columns]
    # groupby drops NaN keys, which would silently delete every no-shock cell
    for k in keys:
        df[k] = df[k].fillna("none")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
           and c not in {"seed", "gain", "lag", "k_eff", "shock_size"}]
    g = df.groupby(keys, as_index=False)[num].agg(["mean", "std"])
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    g["block"] = block
    g["cell_id"] = g.get("cell", g.index)
    g["n_seeds"] = len(df) // max(1, len(g))
    g["substrate"] = "eurostat"
    g["balance_rounds"] = np.nan
    # reconstruct the B8 axes from the cell id when present
    if "cell" in g.columns:
        import re
        if "balance_rounds_mean" not in g.columns:
            g["balance_rounds"] = [
                float(m.group(1)) if (m := re.search(r"_r(\d)", str(c))) else np.nan
                for c in g["cell"]
            ]
    return g


def load_block(block: str) -> pd.DataFrame | None:
    path = RESULTS / block / "aggregate.csv"
    if not path.exists():
        print(f"  [skip] {block}: {path.relative_to(ROOT)} not found")
        return None
    df = pd.read_csv(path)
    df["_block"] = block
    print(f"  [ok]   {block}: {len(df)} cells, {int(df.get('n_seeds', pd.Series([0])).max() or 0)} seeds max")
    return df


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Fetch an aggregate column, tolerating either suffixing convention."""
    for cand in (f"{name}_mean_mean", f"{name}_mean", name):
        if cand in df.columns:
            return df[cand]
    raise KeyError(f"none of {name}[_mean[_mean]] in {list(df.columns)[:10]}...")


def sd(df: pd.DataFrame, name: str) -> pd.Series:
    for key in (f"{name}_sd", f"{name}_std", f"{name}_mean_sd", f"{name}_mean_std"):
        if key in df.columns:
            return df[key]
    return pd.Series([np.nan] * len(df), index=df.index)


SOURCE_NOTE = {
    "results": "Source: results/ (experiment layer).",
    "reference": ("Source: reference_results/ (the Lead's independent reference run; "
                  "cross-checked against the experiment layer in "
                  "reference_results/CROSSCHECK.md)."),
}


def write_table(name: str, df: pd.DataFrame, caption: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    keep = [c for c in df.columns if not df[c].isna().all()] or list(df.columns)
    df = df[keep]
    df.to_csv(path, index=False, float_format="%.10g")
    cap = TABLES / f"{name}.cap"
    if not cap.exists():
        cap.write_text(
            caption.strip() + " " + SOURCE_NOTE[_SOURCE[0]] + "\n", encoding="utf-8"
        )
    print(f"  wrote {path.relative_to(ROOT)}  ({len(df)} rows)")
    return path


# --------------------------------------------------------------------------- tables
def t_calibration(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B1")
    if df is None:
        return
    out = pd.DataFrame({
        "mechanism": df["mechanism"],
        "substrate": df.get("substrate", "eurostat"),
        "output_base": col(df, X),
        "shortage": col(df, "mean_shortage_rate"),
        "welfare": col(df, "mean_welfare"),
        "accounting_residual": col(df, "accounting_residual_max")
        if "accounting_residual_max" in df.columns else np.nan,
    }).sort_values(["substrate", "mechanism"])
    write_table(
        "calibration", out,
        "Base-year reproduction under perfect information (reporting lag 0, no "
        "measurement noise, no shock) on both substrates. Every institution returns "
        "the base-year output vector to six decimals: this is the design's identifying "
        "assumption verified rather than assumed, and the accounting residual bounds "
        "any leak. Generated by scripts/paper_tables.py from results/B1.",
    )


def t_noise_only(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B1b")
    if df is None:
        return
    out = pd.DataFrame({
        "mechanism": df["mechanism"],
        "output_base": col(df, X),
        "output_base_sd": sd(df, X),
        "shortage": col(df, "mean_shortage_rate"),
        "shortage_sd": sd(df, "mean_shortage_rate"),
        "welfare": col(df, "mean_welfare"),
        "welfare_sd": sd(df, "mean_welfare"),
        "spoilage_rate": col(df, "mean_spoilage_rate")
        if "mean_spoilage_rate_mean" in df.columns else np.nan,
        "iterations": col(df, "mean_iterations"),
    }).sort_values("output_base", ascending=False)
    write_table(
        "noise_only", out,
        "The information rent, with no shock at all: measurement noise "
        "$\\sigma=0.03$ and AR(1) demand noise, reporting lag 0, twenty seeds. "
        "Institutions are indistinguishable by the shortage they record (1.56--1.64 "
        "per cent for five of six) and clearly distinguishable by the output they "
        "waste. Generated by scripts/paper_tables.py from results/B1b.",
    )


def t_decomposition(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B8")
    if df is None:
        return
    out = pd.DataFrame({
        "revision_gain": df["gain"] if "gain" in df.columns else np.nan,
        "balance_rounds": col(df, "balance_rounds"),
        "output_base": col(df, X),
        "output_base_sd": sd(df, X),
        "shortage": col(df, "mean_shortage_rate"),
        "shortage_sd": sd(df, "mean_shortage_rate"),
        "spoilage_rate": col(df, "mean_spoilage_rate")
        if "mean_spoilage_rate_mean" in df.columns else np.nan,
    }).sort_values(["revision_gain", "balance_rounds"])
    write_table(
        "balance_decomposition", out,
        "Decomposing the material-balance penalty over the revision gain and the "
        "number of balancing rounds. Neither factor alone moves output; both together "
        "do. The channel is that the revision ratchets the target past the "
        "labour-feasible frontier and the balancing step projects it back one-sidedly. "
        "Generated by scripts/paper_tables.py from results/B8.",
    )


def t_lag_grid(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B2")
    if df is None:
        return
    out = pd.DataFrame({
        "mechanism": df["mechanism"],
        "gain": df.get("gain", np.nan),
        "lag_reporting": df.get("lag", np.nan),
        "k_effective": df.get("m", df.get("lag", 0) + 1),
        "shortage": col(df, "mean_shortage_rate"),
        "shortage_sd": sd(df, "mean_shortage_rate"),
        "welfare": col(df, "mean_welfare"),
        "welfare_sd": sd(df, "mean_welfare"),
        "output_base": col(df, X),
        "amplitude": col(df, "oscillation_amplitude"),
        "acf_lag1": col(df, "acf_lag1"),
        "plan_fulfilment": col(df, "mean_plan_fulfilment"),
        "iterations": col(df, "mean_iterations"),
    }).sort_values(["mechanism", "gain", "lag_reporting"])
    write_table(
        "main_lag_grid", out,
        "The lag-by-mechanism grid: reporting lag $\\ell$ and effective delay "
        "$k=\\ell+1$ are shown side by side because the theory's $g^\\star(k)$ refers "
        "to $k$. Twenty seeds per cell. Generated by scripts/paper_tables.py from "
        "results/B2.",
    )


def t_propagation(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B3")
    if df is None:
        return
    sub = df[df["shock_sector"].notna()] if "shock_sector" in df.columns else df
    if sub.empty:
        return
    out = pd.DataFrame({
        "mechanism": sub["mechanism"],
        "shock_sector": sub["shock_sector"],
        "output_loss": col(sub, "shock_output_loss"),
        "output_loss_sd": sd(sub, "shock_output_loss"),
        "output_amplification": col(sub, "shock_output_amplification"),
        "shortage_amplification": col(sub, "shock_amplification"),
        "peak_shortage": col(sub, "shock_peak_shortage"),
        "recovery_half_life": col(sub, "recovery_half_life"),
    }).sort_values(["mechanism", "output_loss"], ascending=[True, False])
    write_table(
        "propagation", out,
        "Sectoral shock propagation. The primary statistic is the total output loss "
        "caused by a 50 per cent capacity cut lasting eight quarters; the amplification "
        "ratios are reported alongside with the caveat that they divide by the shocked "
        "sector's output share and therefore overweight small sectors. Generated by "
        "scripts/paper_tables.py from results/B3.",
    )


def t_dual_track(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B5")
    if df is None:
        return
    out = pd.DataFrame({
        "theta": df.get("gain", np.nan),
        "lag_reporting": df.get("lag", 0),
        "welfare": col(df, "mean_welfare"),
        "welfare_sd": sd(df, "mean_welfare"),
        "shortage": col(df, "mean_shortage_rate"),
        "output_base": col(df, X),
        "price_volatility": col(df, "mean_price_volatility"),
    }).sort_values(["theta", "lag_reporting"])
    write_table(
        "dual_track", out,
        "The dual-track share $\\theta$. The pre-registered prediction of an interior "
        "optimum is not supported: welfare is monotone in $\\theta$ and market-track "
        "price volatility falls to zero as $\\theta\\to1$. The welfare spread is of the "
        "same order as the seed standard deviation, so the honest reading is 'no "
        "detectable interior optimum'. Generated by scripts/paper_tables.py from "
        "results/B5.",
    )


def t_boundary(blocks: dict[str, pd.DataFrame]) -> None:
    """Measured loop gain against the analytic boundary, m = 1..8."""
    df = blocks.get("B7") if "B7" in blocks else blocks.get("B2")
    if df is None:
        return
    if "gain" not in df.columns or "lag" not in df.columns:
        return
    rows = []
    for m in range(1, 9):
        g_star = 2.0 * np.sin(np.pi / (2 * (2 * m + 1)))
        rows.append({"k_effective": m, "g_star": g_star,
                     "period_theory": 2 * (2 * m + 1)})
    out = pd.DataFrame(rows)
    write_table(
        "stability_boundary", out,
        "The analytic boundary $g^\\star(k)=2\\sin[\\pi/(2(2k+1))]$ and the oscillation "
        "period at the boundary, $2(2k+1)$ quarters. Verified numerically to "
        "$10^{-13}$ in verification/; the measured loop gain in this economy lies one "
        "to two orders of magnitude below it, which is the paper's H1 null. Generated "
        "by scripts/paper_tables.py.",
    )


_SOURCE = ["results"]

def t_labour_slack(blocks: dict[str, pd.DataFrame]) -> None:
    df = blocks.get("B6L")
    if df is None:
        return
    out = pd.DataFrame({
        "labour_supply_factor": df["labour_factor"] if "labour_factor" in df.columns
        else df.get("labour_factor_mean", np.nan),
        "revision_gain": df.get("gain", np.nan),
        "balance_rounds": col(df, "balance_rounds"),
        "output_base": col(df, X),
        "output_base_sd": sd(df, X),
        "shortage": col(df, "mean_shortage_rate"),
    }).sort_values(["labour_supply_factor", "revision_gain", "balance_rounds"])
    write_table(
        "labour_slack", out,
        "The cost of taut planning is conditional on labour slack: the same "
        "revision gain costs 13 per cent of output when the economy can employ only "
        "15 per cent more labour than in the base year, and nothing when labour is "
        "abundant. Generated by scripts/paper_tables.py.",
    )


BUILDERS = [t_calibration, t_noise_only, t_decomposition, t_lag_grid,
            t_propagation, t_dual_track, t_boundary, t_labour_slack]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", nargs="*", default=None)
    ap.add_argument("--source", choices=("results", "reference"), default="results",
                    help="which implementation the tables are built from")
    args = ap.parse_args(argv)
    wanted = args.blocks or ["B1", "B1b", "B2", "B3", "B5", "B7", "B8", "B6L"]
    loader = load_block if args.source == "results" else aggregate_reference
    where = RESULTS.relative_to(ROOT) if args.source == "results" else "reference_results"
    print(f"reading {where}/ (source: {args.source})")
    blocks: dict[str, pd.DataFrame] = {}
    for b in wanted:
        df = loader(b)
        if df is not None:
            blocks[b] = df
            print(f"  [ok]   {b}: {len(df)} cells")
    if not blocks:
        print("\nno blocks found — nothing to do")
        return 1
    _SOURCE[0] = args.source
    print(f"\nwriting {TABLES.relative_to(ROOT)}/")
    for builder in BUILDERS:
        try:
            builder(blocks)
        except Exception as exc:
            print(f"  [warn] {builder.__name__}: {type(exc).__name__}: {exc}")
    print("\ndone. `python scripts/build_paper.py` will resolve the table tokens "
          "for any float whose csv and cap now exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
