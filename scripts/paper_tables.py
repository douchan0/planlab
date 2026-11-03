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
    calm = blocks.get("B1")
    if calm is not None and "substrate" in calm and "mean_consumption_mean" in calm:
        base_consumption = calm.loc[calm["substrate"] == "eurostat", "mean_consumption_mean"].mean()
        out["consumption_base"] = col(df, "mean_consumption") / base_consumption
    labels = {
        "hybrid_dual_track": "双轨制", "tatonnement": "试探性定价", "market": "市场",
        "rationing_priority": "优先配给", "lp_central": "中央线性规划", "material_balance": "物资平衡",
    }
    localized = out.rename(columns={
        "mechanism": "机制", "output_base": "产出比", "shortage": "短缺率", "welfare": "福利点",
        "spoilage_rate": "溢损率", "consumption_base": "居民交付比",
    }).copy()
    localized["机制"] = localized["机制"].map(labels)
    columns = ["机制", "产出比", "居民交付比", "短缺率", "福利点", "溢损率"]
    localized.reindex(columns=columns).round(5).to_csv(TABLES / "noise_only.zh.csv", index=False)
    headings = {
        "mechanism": "Rule", "output_base": "Output", "consumption_base": "Household delivery",
        "shortage": "Shortage", "welfare": "Utility points", "spoilage_rate": "Spoilage",
    }
    out.reindex(columns=list(headings)).rename(columns=headings).round(5).to_csv(
        TABLES / "noise_only.en.csv", index=False,
    )
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
    out[["revision_gain", "balance_rounds", "output_base", "shortage", "spoilage_rate"]].rename(columns={
        "revision_gain": "修订增益", "balance_rounds": "平衡轮数", "output_base": "产出比",
        "shortage": "短缺率", "spoilage_rate": "溢损率",
    }).round(5).to_csv(TABLES / "balance_decomposition.zh.csv", index=False)
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
    # the price elasticity MUST be carried: tatonnement appears twice per (gain, lag)
    # at elasticities 0.4 and 0.8, and without this column those rows are
    # indistinguishable, so a reader cannot tell which one the text quotes.
    out = pd.DataFrame({
        "mechanism": df["mechanism"],
        "gain": df.get("gain", np.nan),
        "price_elasticity": df.get("price_elasticity", np.nan),
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
    }).sort_values(["mechanism", "price_elasticity", "gain", "lag_reporting"])
    write_table(
        "main_lag_grid", out,
        "The lag-by-mechanism grid: reporting lag $\\ell$ and effective delay "
        "$k=\\ell+1$ are shown side by side because the theory's $g^\\star(k)$ refers "
        "to $k$. Twenty seeds per cell. Generated by scripts/paper_tables.py from "
        "results/B2.",
    )

    if _SOURCE[0] == "results":
        reference = out[np.isclose(out["gain"], 0.30) & np.isclose(out["price_elasticity"], 0.4)]
        write_table("lag_grid_reference", reference, "Reference operating point from B2.")



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
    })
    if "paired_shock_output_loss_mean" in sub:
        out["paired_output_loss"] = col(sub, "paired_shock_output_loss")
        out["paired_output_amplification"] = col(sub, "paired_shock_output_amplification")
    out = out.sort_values(["mechanism", "output_loss"], ascending=[True, False])
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
    # `gain` in B5 is the mechanism's market_gain (constant 0.3); the dual-track
    # share is carried in its own column.  Using `gain` as theta silently produced a
    # table whose every row read 0.3, which a reader would have read as a flat result.
    if "theta" in df.columns:
        theta = df["theta"]
    elif "theta_param" in df.columns:
        theta = df["theta_param"]
    else:
        import re
        theta = df["cell_id"].astype(str).str.extract(r"theta([0-9.]+)")[0].astype(float)
    out = pd.DataFrame({
        "theta": theta,
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
    """Exact bound for the signed scalar equation, independent of the grid."""
    rows = []
    for m in range(9):
        g_star = 2.0 * np.sin(np.pi / (2 * (2 * m + 1)))
        rows.append({"k_effective": m, "g_star": g_star,
                     "period_theory": 2 * (2 * m + 1)})
    out = pd.DataFrame(rows)
    out.rename(columns={
        "k_effective": "有效时滞", "g_star": "临界增益", "period_theory": "临界周期",
    }).round(5).to_csv(TABLES / "stability_boundary.zh.csv", index=False)
    write_table(
        "stability_boundary", out,
        "Exact bound and period of the nonzero critical mode for the signed scalar equation.",
    )


_SOURCE = ["results"]

def t_additional_experiments(blocks: dict[str, pd.DataFrame]) -> None:
    if _SOURCE[0] != "results":
        return
    rationing = blocks.get("B4")
    if rationing is not None:
        columns = ["mechanism", "shock_size", "n_seeds", "mean_shortage_rate_mean",
                   "mean_shortage_rate_sd", "mean_welfare_mean", "mean_welfare_sd",
                   "output_ratio_mean_mean", "mean_spoilage_rate_mean"]
        out = rationing[columns].copy()
        out.insert(2, "capacity_loss", ((1.0 - out["shock_size"]) * 100).round().astype(int))
        write_table("rationing_shadow", out.sort_values(["capacity_loss", "mechanism"]),
                    "Priority rationing and administered price adjustment at three capacity losses.")
    if "B6" in blocks:
        robustness = pd.read_csv(RESULTS / "tables" / "t10_robustness.csv")
        write_table("robustness", robustness, "Sensitivity experiments from B6.")
        labour = robustness[robustness["variant"].str.startswith("labour")]
        write_table("labour_slack", labour[[
            "variant", "labour_supply_factor", "mean_x_over_base",
            "mean_shortage_rate", "mean_labour_utilisation",
        ]], "Matched labour-supply sensitivity experiments from B6.")


def t_adjudication() -> None:
    """Publish the verdicts from the analysis layer without a second test pipeline."""
    source = pd.read_csv(RESULTS / "tables" / "t00_hypotheses.csv")
    hypotheses = ["H1", "H1'", "H2", "H3", "H4", "H5"]
    evidence = source.set_index("hypothesis").loc[hypotheses].reset_index()
    tests = pd.read_csv(RESULTS / "tables" / "tests.csv")
    h3 = tests.loc[tests["hypothesis"] == "H3", "note"].iloc[0]
    evidence.loc[evidence["hypothesis"] == "H3", "verdict"] += f" ({h3})"
    evidence["n_significant_holm_05"] = evidence["n_significant_holm_05"].astype(int)
    evidence["n_tests"] = evidence["n_tests"].astype(int)
    evidence = evidence[["hypothesis", "verdict", "n_tests", "n_significant_holm_05"]]
    h3_min = float(source.loc[source["hypothesis"] == "H3", "key_statistic"].iloc[0])
    h3_below = int(tests.loc[tests["hypothesis"] == "H3", "effect_value"].iloc[0])
    translated = evidence.copy()
    translated["verdict"] = translated["verdict"].replace({
        "Threshold not identified": "实验无法识别标量阈值",
        "See test log": "修订与平衡的交互见分解表",
        "Ordering not established": "完整排序未达到校正标准",
        "LP steepest": "中央线性规划的下降最快",
        "Interior advantage not established": "未检出内点优势",
        "Interior advantage detected": "检测到内点优势",
    }, regex=False)
    h4 = translated["hypothesis"] == "H4"
    translated.loc[h4, "verdict"] = translated.loc[h4, "verdict"].str.replace(
        r"(\d+) amplification tests; (\d+) forward-linkage tests",
        r"放大检验通过 \1 项，前向关联检验通过 \2 项", regex=True,
    )
    translated.loc[translated["hypothesis"] == "H3", "verdict"] = (
        f"最低短缺率 {h3_min:.2%}，低于 0.5% 的部门有 {h3_below} 个"
    )
    translated.rename(columns={
        "hypothesis": "假设", "verdict": "裁决", "n_tests": "统计检验数",
        "n_significant_holm_05": "校正后显著数",
    }).to_csv(TABLES / "hypothesis_adjudication.zh.csv", index=False)
    write_table(
        "hypothesis_adjudication",
        evidence.rename(columns={
            "hypothesis": "Hypothesis", "verdict": "Result",
            "n_tests": "Tests", "n_significant_holm_05": "Holm < 0.05",
        }),
        "Decisions generated from the experiment test log.",
    )


BUILDERS = [t_calibration, t_noise_only, t_decomposition, t_lag_grid,
            t_propagation, t_dual_track, t_boundary, t_additional_experiments]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", nargs="*", default=None)
    ap.add_argument("--source", choices=("results", "reference"), default="results",
                    help="which implementation the tables are built from")
    args = ap.parse_args(argv)
    wanted = args.blocks or ["B1", "B1b", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
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
    if args.source == "results" and args.blocks is None:
        t_adjudication()
    for builder in BUILDERS:
        try:
            builder(blocks)
        except Exception as exc:
            print(f"  [error] {builder.__name__}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print("\ndone. `python scripts/build_paper.py` will resolve the table tokens "
          "for any float whose csv and cap now exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
