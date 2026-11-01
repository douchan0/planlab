"""Publication figures, generated from ``results/`` and nothing else.

Design rules
------------
* English labels, no CJK, no seaborn defaults: an explicit ``rcParams`` block
  (serif-free sans, thin spines, 8 pt ticks) is set once in :func:`_style`.
* Every figure is written as 300 dpi PNG **and** PDF, with the PNG metadata
  stripped so the file is byte-stable on this machine (a notebook can regenerate
  it and compare arrays).
* A figure is a pure function of the results files: no timestamps, no sampling,
  no hand-placed annotations with numbers in them.  Every number that appears in a
  title or an axis comes from ``results/``.
* ``results/figures_index.csv`` records, for each figure, the block it is built
  from, the two files and the function that made it.

Entry points
------------
``make_all(results_dir=None, out_dir=None)`` builds every figure.
``python -m planlab.experiments.figures --only fig_lag_grid`` builds one.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import aggregate as A

__all__ = ["FIGURES", "make_all", "make_lag_grid", "main"]

_PALETTE = {
    "lp_central": "#1f3b73",
    "material_balance": "#b2182b",
    "rationing_priority": "#e08214",
    "tatonnement": "#3b7dd8",
    "market": "#1b7837",
    "hybrid_dual_track": "#762a83",
}
_MARKERS = {
    "lp_central": "o",
    "material_balance": "s",
    "rationing_priority": "^",
    "tatonnement": "D",
    "market": "v",
    "hybrid_dual_track": "P",
}
_LABELS = {
    "lp_central": "central LP",
    "material_balance": "material balance",
    "rationing_priority": "priority rationing",
    "tatonnement": "tatonnement",
    "market": "market",
    "hybrid_dual_track": "dual track",
}


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.4,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "text.usetex": False,
            "svg.fonttype": "none",
        }
    )


def _save(fig: Any, out_dir: Path, name: str, written: list[dict[str, Any]], *, block: str, what: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    metadata = {"Software": None, "CreationDate": None, "ModDate": None}
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata=metadata)
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    import matplotlib.pyplot as plt

    plt.close(fig)
    written.append(
        {
            "figure": name,
            "block": block,
            "png": str(png),
            "pdf": str(pdf),
            "what": what,
            "generated_by": f"planlab.experiments.figures.{name}",
        }
    )


def _mechanism_order(values: Sequence[str]) -> list[str]:
    known = [m for m in A.MECH_ORDER if m in set(values)]
    return known + sorted(set(values) - set(known))


# ---------------------------------------------------------------------------
# 1. lag grid (H2)
# ---------------------------------------------------------------------------
def make_lag_grid(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Shortage and welfare loss against the reporting lag, per institution."""
    import matplotlib.pyplot as plt

    if not bundle.has("B2"):
        return
    frame = A._at_reference_gain(A._substrate(A.merged_summary(bundle, ["B2"])))
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for mechanism in _mechanism_order(frame["mechanism"].unique()):
        sub = frame[frame["mechanism"] == mechanism]
        stats = sub.groupby(["lag", "m"]).agg(
            shortage=("mean_shortage_rate", "mean"),
            shortage_sd=("mean_shortage_rate", "std"),
            loss=("welfare_loss_vs_baseline", "mean"),
            loss_sd=("welfare_loss_vs_baseline", "std"),
            n=("mean_shortage_rate", "size"),
        ).reset_index()
        stats = stats.sort_values("lag")
        x = stats["lag"].to_numpy(dtype=float)
        se = (stats["shortage_sd"] / np.sqrt(stats["n"].clip(lower=1))).to_numpy(dtype=float)
        axes[0].plot(x, stats["shortage"].to_numpy() * 100, marker=_MARKERS[mechanism],
                     color=_PALETTE[mechanism], label=_LABELS[mechanism])
        axes[0].fill_between(x, (stats["shortage"] - 1.96 * se) * 100,
                             (stats["shortage"] + 1.96 * se) * 100,
                             color=_PALETTE[mechanism], alpha=0.15, linewidth=0)
        se_l = (stats["loss_sd"] / np.sqrt(stats["n"].clip(lower=1))).to_numpy(dtype=float)
        axes[1].plot(x, stats["loss"].to_numpy() * 100, marker=_MARKERS[mechanism],
                     color=_PALETTE[mechanism])
        axes[1].fill_between(x, (stats["loss"] - 1.96 * se_l) * 100,
                             (stats["loss"] + 1.96 * se_l) * 100,
                             color=_PALETTE[mechanism], alpha=0.15, linewidth=0)
    axes[0].set_xlabel("reporting lag $\\ell$ (effective delay $k=\\ell+1$)")
    axes[0].set_ylabel("mean shortage rate (per cent)")
    axes[0].set_title("(a) Shortage against delay")
    axes[1].set_xlabel("reporting lag $\\ell$ (effective delay $k=\\ell+1$)")
    axes[1].set_ylabel("welfare loss vs baseline (per cent)")
    axes[1].set_title("(b) Welfare loss against delay")
    axes[0].legend(loc="lower right", ncol=2)
    _save(fig, out_dir, "fig_lag_grid", written, block="B2",
          what="shortage and welfare loss against the reporting lag, one line per institution (H2)")
    make_lag_grid.lag_stats = frame  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. gain sweep (H1 and H1')
# ---------------------------------------------------------------------------
def make_gain_sweep(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """The gain axis: plan revision (H1') and price adjustment (H1)."""
    import matplotlib.pyplot as plt

    if not bundle.has("B2"):
        return
    frame = A._substrate(A.merged_summary(bundle, ["B2"]))
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    mb = frame[frame["mechanism"] == "material_balance"]
    for lag, group in mb.groupby("lag"):
        stats = group.groupby("gain").agg(
            value=("mean_shortage_rate", "mean"), sd=("mean_shortage_rate", "std"), n=("seed", "size")
        ).reset_index()
        se = stats["sd"] / np.sqrt(stats["n"].clip(lower=1))
        axes[0].plot(stats["gain"], stats["value"] * 100, marker="o", label=f"$k={int(lag)+1}$")
        axes[0].fill_between(stats["gain"], (stats["value"] - 1.96 * se) * 100,
                             (stats["value"] + 1.96 * se) * 100, alpha=0.12, linewidth=0)
    axes[0].set_xlabel("plan-revision gain (material balance)")
    axes[0].set_ylabel("mean shortage rate (per cent)")
    axes[0].set_title("(a) Taut plans are self-defeating (H1$'$)")
    axes[0].legend(title="effective delay", ncol=2, loc="upper left")

    tat = frame[frame["mechanism"] == "tatonnement"]
    if not tat.empty:
        for lag, group in tat.groupby("lag"):
            stats = group.groupby("gain").agg(
                value=("oscillation_amplitude", "mean"), sd=("oscillation_amplitude", "std"),
                n=("seed", "size"),
            ).reset_index()
            axes[1].plot(stats["gain"], stats["value"], marker="s", label=f"$k={int(lag)+1}$")
        axes[1].set_xlabel("administered price gain (tatonnement)")
        axes[1].set_ylabel("price oscillation amplitude")
        axes[1].set_title("(b) The price loop stays quiet (H1)")
        axes[1].legend(title="effective delay", ncol=2, loc="upper left")
        axes[1].set_yscale("log")
    _save(fig, out_dir, "fig_gain_sweep", written, block="B2",
          what="shortage against the plan-revision gain and price amplitude against the price gain")


# ---------------------------------------------------------------------------
# 3. stability boundary (H1)
# ---------------------------------------------------------------------------
def make_stability_boundary(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Amplitude against gain, and the measured boundary against ``g*(m)/b_p``."""
    import matplotlib.pyplot as plt

    if not bundle.has("B2"):
        return
    frame = A._substrate(A.merged_summary(bundle, ["B2"]))
    tat = frame[frame["mechanism"] == "tatonnement"]
    if tat.empty:
        return
    cells, per_m, _per_gain = A.boundary_cells(tat)
    gains = A.gain_estimate(bundle)
    bp = A.bp_summary(gains)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    eps_used = float(tat["price_elasticity"].mode().iloc[0])
    sub = cells[np.isclose(cells["price_elasticity"], eps_used)]
    cmap = plt.get_cmap("viridis")
    ms = sorted(sub["m"].unique())
    for i, m in enumerate(ms):
        block = sub[sub["m"] == m].sort_values("gain")
        axes[0].plot(block["gain"], block["amp"], marker="o",
                     color=cmap(i / max(len(ms) - 1, 1)), label=f"$k={int(m)}$")
    axes[0].set_xlabel("price gain $\\eta$")
    axes[0].set_ylabel("price oscillation amplitude")
    axes[0].set_title(f"(a) Amplitude against gain ($\\varepsilon={eps_used:g}$)")
    axes[0].legend(title="effective delay", ncol=2)

    if not bp.empty:
        b_row = bp[np.isclose(bp["price_elasticity"], eps_used)]
        b = float(b_row["b_p_mean"].iloc[0]) if not b_row.empty else float("nan")
        if np.isfinite(b) and b > 0:
            mm = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
            theory = np.array([A.G_STAR(int(v)) / b for v in mm])
            axes[1].plot(theory, mm, "k-", linewidth=1.0, label="theory $g^*(m)/b_p$")
            if not per_m.empty:
                got = per_m[np.isclose(per_m["price_elasticity"], eps_used)].dropna(
                    subset=["gain_star"]
                )
                axes[1].plot(got["gain_star"], got["lag_effective"], "o", color="#b2182b",
                             label="measured $\\eta^*$")
            gain_max = float(tat["gain"].max())
            g_max = gain_max * b
            axes[1].axvline(gain_max, color="#777777", linestyle=":", linewidth=1.0)
            axes[1].annotate(
                f"largest gain tested\n($\\eta={gain_max:g}$, $g={g_max:.3g}$)",
                xy=(gain_max, 4.0), xytext=(-4, 0), textcoords="offset points",
                fontsize=7, ha="right", va="center", color="#555555",
            )
            axes[1].axvline(min(A.G_STAR(1) / b, gain_max), color="#bbbbbb", linewidth=0.0)
            axes[1].set_xlabel("critical price gain $\\eta^*$")
            axes[1].set_ylabel("effective delay $k=\\ell+1$")
            axes[1].set_title("(b) Where the boundary is")
            axes[1].legend(loc="lower right")
            axes[1].set_xlim(left=0)
    _save(fig, out_dir, "fig_stability_boundary", written, block="B2/B7",
          what="amplitude against gain and the analytic boundary g*(m)/b_p against measurement (H1)")


# ---------------------------------------------------------------------------
# 4. shock propagation (H4)
# ---------------------------------------------------------------------------
def make_shock_propagation(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Sector-by-sector response heatmap and amplification against linkage."""
    import matplotlib.pyplot as plt

    if "B3" not in bundle.sectors:
        return
    panel = bundle.sectors["B3"]
    panel = panel[panel["mechanism"] == "material_balance"]
    if panel.empty:
        return
    matrix = panel.pivot_table(
        index="cell_id", columns="sector", values="unmet_share_of_total", aggfunc="mean"
    )
    shocked_names = (
        panel[panel["is_shocked"]].groupby("cell_id")["sector"].first().reindex(matrix.index)
    )
    centrality = A.forward_linkage(bundle)
    order = sorted(
        matrix.index, key=lambda c: (centrality.get(str(shocked_names.get(c)), 0.0), str(c))
    )
    matrix = matrix.loc[order]
    row_labels = [f"{shocked_names.get(c, c)}" for c in matrix.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    im = axes[0].imshow(matrix.to_numpy(), aspect="auto", cmap="magma_r", origin="lower")
    axes[0].set_xticks(range(matrix.shape[1]))
    axes[0].set_xticklabels(matrix.columns, rotation=90, fontsize=6.5)
    axes[0].set_yticks(range(matrix.shape[0]))
    axes[0].set_yticklabels(row_labels, fontsize=6.5)
    axes[0].set_xlabel("sector bearing the shortage")
    axes[0].set_ylabel("shocked sector")
    axes[0].set_title("(a) Incidence of the household shortage")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02, label="share of shortage")

    summary = A.merged_summary(bundle, ["B3"])
    summary = A._substrate(summary)
    summary = summary[summary["mechanism"] == "material_balance"]
    stats = summary.groupby("shock_sector").agg(
        amp=("shock_output_amplification", "mean"),
        amp_sd=("shock_output_amplification", "std"),
        n=("shock_output_amplification", "size"),
    ).reset_index()
    stats["centrality"] = stats["shock_sector"].map(centrality)
    stats = stats.sort_values("centrality")
    y = np.arange(len(stats))
    axes[1].barh(y, stats["amp"], color="#b2182b", alpha=0.85)
    axes[1].errorbar(stats["amp"], y, xerr=1.96 * stats["amp_sd"] / np.sqrt(stats["n"].clip(lower=1)),
                     fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5)
    axes[1].axvline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(stats["shock_sector"], fontsize=6.5)
    axes[1].set_xlabel("output amplification $\\mathrm{Amp}^{\\mathrm{out}}$")
    axes[1].set_ylabel("shocked sector (increasing forward linkage $\\rightarrow$)")
    axes[1].set_title("(b) Network amplification (H4)")
    _save(fig, out_dir, "fig_shock_propagation", written, block="B3",
          what="sector x sector incidence of the shortage and output amplification by shocked sector")


# ---------------------------------------------------------------------------
# 5. priority ladder (B2L)
# ---------------------------------------------------------------------------
def make_ladder_comparison(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Production-first against consumption-first when a supply shock hits."""
    import matplotlib.pyplot as plt

    if not bundle.has("B2L"):
        return
    frame = A._substrate(A.merged_summary(bundle, ["B2L"]))
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    runs = bundle.runs.get("B2L")
    if runs is not None and not runs.empty:
        runs = runs[runs["substrate"] == "eurostat"] if "substrate" in runs else runs
        for lag in sorted(frame["lag"].unique())[:2]:
            for ladder, style in (("production_first", "-"), ("consumption_first", "--")):
                sub = runs[(runs["ladder"] == ladder) & (runs["lag"] == lag)]
                if sub.empty:
                    continue
                curve = sub.groupby("t")["shortage_rate"].mean()
                axes[0].plot(curve.index, curve.to_numpy() * 100, style,
                             color="#1b7837" if ladder == "production_first" else "#b2182b",
                             label=f"{ladder.replace('_', ' ')}, $\\ell={int(lag)}$")
        axes[0].set_xlabel("quarter")
        axes[0].set_ylabel("shortage rate (per cent)")
        axes[0].set_title("(a) Shortage trajectory")
        axes[0].legend(ncol=2)
    stats = frame.groupby(["ladder", "lag"]).agg(
        value=("mean_shortage_rate", "mean"), sd=("mean_shortage_rate", "std"), n=("seed", "size")
    ).reset_index()
    widths = 0.38
    lags = sorted(stats["lag"].unique())
    for i, ladder in enumerate(("production_first", "consumption_first")):
        sub = stats[stats["ladder"] == ladder].set_index("lag").reindex(lags)
        axes[1].bar(np.arange(len(lags)) + (i - 0.5) * widths, sub["value"] * 100,
                    width=widths, label=ladder.replace("_", " "),
                    color="#1b7837" if ladder == "production_first" else "#b2182b", alpha=0.85)
        axes[1].errorbar(np.arange(len(lags)) + (i - 0.5) * widths, sub["value"] * 100,
                         yerr=1.96 * sub["sd"] / np.sqrt(sub["n"].clip(lower=1)) * 100,
                         fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5)
    axes[1].set_xticks(np.arange(len(lags)))
    axes[1].set_xticklabels([f"$k={int(v)+1}$" for v in lags])
    axes[1].set_ylabel("mean shortage rate (per cent)")
    axes[1].set_title("(b) Mean shortage by ladder")
    axes[1].legend()
    _save(fig, out_dir, "fig_ladder_comparison", written, block="B2L",
          what="priority ladder as a treatment: production-first vs consumption-first under a shock")


# ---------------------------------------------------------------------------
# 6. dual track (H5)
# ---------------------------------------------------------------------------
def make_dual_track_frontier(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Welfare and price volatility against the plan share theta."""
    import matplotlib.pyplot as plt

    if not bundle.has("B5"):
        return
    frame = A._substrate(A.merged_summary(bundle, ["B5"]))
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for lag, group in frame.groupby("lag"):
        stats = group.groupby("theta").agg(
            welfare=("mean_welfare", "mean"), wsd=("mean_welfare", "std"),
            loss=("welfare_loss_vs_baseline", "mean"), lsd=("welfare_loss_vs_baseline", "std"),
            vol=("mean_price_volatility", "mean"), vsd=("mean_price_volatility", "std"),
            n=("seed", "size"),
        ).reset_index()
        colour = "#1f3b73" if lag == 0 else "#e08214"
        axes[0].plot(stats["theta"], stats["loss"] * 100, marker="o", color=colour,
                     label=f"$k={int(lag)+1}$")
        axes[0].fill_between(stats["theta"],
                             (stats["loss"] - 1.96 * stats["lsd"] / np.sqrt(stats["n"])) * 100,
                             (stats["loss"] + 1.96 * stats["lsd"] / np.sqrt(stats["n"])) * 100,
                             color=colour, alpha=0.15, linewidth=0)
        axes[1].plot(stats["theta"], stats["vol"], marker="s", color=colour,
                     label=f"$k={int(lag)+1}$")
    for ax in axes:
        ax.set_xlabel("plan share $\\theta$")
    axes[0].set_ylabel("welfare loss vs baseline (per cent)")
    axes[0].set_title("(a) Welfare against the plan share")
    axes[1].set_ylabel("price volatility")
    axes[1].set_title("(b) Price volatility against the plan share")
    axes[0].legend()
    _save(fig, out_dir, "fig_dual_track_frontier", written, block="B5",
          what="dual-track frontier: welfare and price volatility against the plan share theta (H5)")


# ---------------------------------------------------------------------------
# 7. welfare-shortage trade-off across all cells
# ---------------------------------------------------------------------------
def make_welfare_shortage_tradeoff(
    bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]
) -> None:
    """Every cell of the design on one welfare/shortage plane."""
    import matplotlib.pyplot as plt

    blocks = [b for b in ("B2", "B3", "B4", "B5", "B6") if bundle.has(b)]
    if not blocks:
        return
    summary = A._substrate(A.merged_summary(bundle, blocks))
    if summary.empty:
        return
    stats = summary.groupby(["block", "mechanism"]).agg(
        shortage=("mean_shortage_rate", "mean"),
        loss=("welfare_loss_vs_baseline", "mean"),
        n=("seed", "size"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for mechanism in _mechanism_order(stats["mechanism"].unique()):
        sub = stats[stats["mechanism"] == mechanism]
        ax.scatter(sub["shortage"] * 100, sub["loss"] * 100, s=26,
                   color=_PALETTE[mechanism], marker=_MARKERS[mechanism],
                   label=_LABELS[mechanism], alpha=0.9, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("mean shortage rate (per cent)")
    ax.set_ylabel("welfare loss vs baseline (per cent)")
    ax.set_title("Institutions on the shortage/welfare plane")
    ax.legend(ncol=2)
    _save(fig, out_dir, "fig_welfare_shortage_tradeoff", written, block="B2-B6",
          what="every cell of the design on the shortage/welfare plane, coloured by institution")


# ---------------------------------------------------------------------------
# 8. planning effort (balance rounds) and the information rent (B1b)
# ---------------------------------------------------------------------------
def make_budget_line(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Planning effort (internal balance rounds) against outcome, and the information rent."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    if bundle.has("B6"):
        frame = A._substrate(A.merged_summary(bundle, ["B6"]))
        effort = frame[frame["variant"].astype(str).str.startswith("balance_rounds")]
        if not effort.empty:
            effort = effort.copy()
            effort["rounds"] = effort["variant"].astype(str).str.extract(r"(\d+)").astype(float)
            stats = effort.groupby("rounds").agg(
                shortage=("mean_shortage_rate", "mean"), ssd=("mean_shortage_rate", "std"),
                loss=("welfare_loss_vs_baseline", "mean"), lsd=("welfare_loss_vs_baseline", "std"),
                n=("seed", "size"),
            ).reset_index()
            axes[0].plot(stats["rounds"], stats["shortage"] * 100, marker="o", color="#b2182b",
                         label="shortage")
            axes[0].set_xlabel("internal balance rounds per quarter (planning effort)")
            axes[0].set_ylabel("mean shortage rate (per cent)", color="#b2182b")
            twin = axes[0].twinx()
            twin.plot(stats["rounds"], stats["loss"] * 100, marker="s", color="#1f3b73",
                      linestyle="--", label="welfare loss")
            twin.set_ylabel("welfare loss (per cent)", color="#1f3b73")
            twin.spines["right"].set_visible(True)
            twin.grid(False)
            axes[0].set_title("(a) Planning effort is not free")
            # reference and the two rounds variants are connected by a straight
            # interpolation, which is honest: the grid has only these levels.
    if bundle.has("B1b"):
        frame = A._substrate(A.merged_summary(bundle, ["B1b"]))
        stats = frame.groupby("mechanism").agg(
            x=("output_ratio_mean", "mean"), xsd=("output_ratio_mean", "std"),
            shortage=("mean_shortage_rate", "mean"), n=("seed", "size"),
        ).reset_index()
        stats = stats.set_index("mechanism").loc[
            [m for m in _mechanism_order(stats["mechanism"].unique()) if m in stats["mechanism"].values]
        ].reset_index()
        y = np.arange(len(stats))
        axes[1].barh(y, stats["x"], color=[_PALETTE[m] for m in stats["mechanism"]], alpha=0.9)
        axes[1].errorbar(stats["x"], y, xerr=1.96 * stats["xsd"] / np.sqrt(stats["n"].clip(lower=1)),
                         fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([_LABELS[m] for m in stats["mechanism"]])
        axes[1].set_xlim(min(0.9, float(stats["x"].min()) - 0.02), 1.0)
        axes[1].set_xlabel("output relative to the base year, $x/\\bar x$")
        axes[1].set_title("(b) The information rent of an institution")
    _save(fig, out_dir, "fig_budget_line", written, block="B6/B1b",
          what="planning effort against outcome, and the information rent with no shock at all")


# ---------------------------------------------------------------------------
# 9. incidence (B3)
# ---------------------------------------------------------------------------
def make_incidence(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """Who bears the shortage: households, government or intermediate users."""
    import matplotlib.pyplot as plt

    if "B3" not in bundle.sectors:
        return
    panel = bundle.sectors["B3"]
    panel = panel[panel["mechanism"] == "material_balance"]
    if panel.empty:
        return
    shocked = panel[panel["is_shocked"]].groupby("sector").agg(
        household=("final_share_of_rationed", "mean"),
        other=("other_share_of_rationed", "mean"),
        unmet_rate=("household_unmet_rate_peak", "mean"),
        service=("intermediate_service_rate", "mean"),
    ).reset_index().sort_values("household", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    y = np.arange(len(shocked))
    axes[0].barh(y, shocked["household"], color="#b2182b", alpha=0.9, label="households")
    axes[0].barh(y, shocked["other"], left=shocked["household"], color="#4d4d4d", alpha=0.9,
                 label="government and investment")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(shocked["sector"], fontsize=6.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("share of the rationed shortfall of the shocked good")
    axes[0].set_title("(a) Who bears it")
    axes[0].legend(loc="lower right")

    axes[1].scatter(shocked["unmet_rate"], shocked["service"], s=24, color="#1f3b73",
                    edgecolors="white", linewidths=0.4)
    for _, row in shocked.iterrows():
        axes[1].annotate(str(row["sector"]), (row["unmet_rate"], row["service"]), fontsize=5.5,
                         xytext=(2, 2), textcoords="offset points")
    axes[1].set_xlabel("peak household unmet rate on the shocked good")
    axes[1].set_ylabel("intermediate service rate (plan gap)")
    axes[1].set_title("(b) Households versus production")
    _save(fig, out_dir, "fig_incidence", written, block="B3",
          what="incidence of the shortfall: households versus government/investment, and the intermediate plan gap")


# ---------------------------------------------------------------------------
# 9b. material-balance decomposition (B8)
# ---------------------------------------------------------------------------
def make_mb_decomposition(
    bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]
) -> None:
    """Two heatmaps: output and shortage over revision gain x balancing rounds."""
    import matplotlib.pyplot as plt

    if not bundle.has("B8"):
        return
    frame = A.merged_summary(bundle, ["B8"])
    if frame.empty:
        return
    x = frame.pivot_table(
        index="balance_rounds", columns="revision_gain", values="output_ratio_mean", aggfunc="mean"
    )
    s = frame.pivot_table(
        index="balance_rounds", columns="revision_gain", values="mean_shortage_rate", aggfunc="mean"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, matrix, label, cmap, fmt in (
        (axes[0], x, "output relative to base year", "viridis", "{:.3f}"),
        (axes[1], s * 100, "mean shortage rate (per cent)", "magma_r", "{:.2f}"),
    ):
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([f"{v:g}" for v in matrix.columns])
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels([f"{v:g}" for v in matrix.index])
        ax.set_xlabel("plan-revision gain")
        ax.set_ylabel("internal balance rounds")
        ax.set_title(label)
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.to_numpy()[i, j]
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=6.5,
                        color="white" if (cmap == "viridis" and value < 0.95) or
                        (cmap != "viridis" and value > 3.0) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _save(fig, out_dir, "fig_mb_decomposition", written, block="B8",
          what="material balance decomposed: output and shortage over revision gain x balancing rounds")


# ---------------------------------------------------------------------------
# 10. noise only (B1b)
# ---------------------------------------------------------------------------
def make_noise_only(bundle: A.ResultsBundle, out_dir: Path, written: list[dict[str, Any]]) -> None:
    """The information rent: output and shortage with no shock and no lag."""
    import matplotlib.pyplot as plt

    if not bundle.has("B1b"):
        return
    frame = A._substrate(A.merged_summary(bundle, ["B1b"]))
    stats = frame.groupby("mechanism").agg(
        x=("output_ratio_mean", "mean"), xsd=("output_ratio_mean", "std"),
        shortage=("mean_shortage_rate", "mean"), ssd=("mean_shortage_rate", "std"),
        n=("seed", "size"),
    ).reset_index()
    stats = stats.set_index("mechanism")
    stats = stats.loc[[m for m in A.MECH_ORDER if m in stats.index]].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    y = np.arange(len(stats))
    axes[0].barh(y, stats["shortage"] * 100, color=[_PALETTE[m] for m in stats["mechanism"]], alpha=0.9)
    axes[0].errorbar(stats["shortage"] * 100, y,
                     xerr=1.96 * stats["ssd"] / np.sqrt(stats["n"].clip(lower=1)) * 100,
                     fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_LABELS[m] for m in stats["mechanism"]])
    axes[0].set_xlabel("mean shortage rate (per cent)")
    axes[0].set_title("(a) Shortage without a shock")
    axes[1].barh(y, (1.0 - stats["x"]) * 100, color=[_PALETTE[m] for m in stats["mechanism"]], alpha=0.9)
    axes[1].errorbar((1.0 - stats["x"]) * 100, y,
                     xerr=1.96 * stats["xsd"] / np.sqrt(stats["n"].clip(lower=1)) * 100,
                     fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("output loss relative to the base year (per cent)")
    axes[1].set_title("(b) The information rent")
    _save(fig, out_dir, "fig_noise_only", written, block="B1b",
          what="no shock, no lag: shortage and output loss caused only by stale and noisy information")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
FIGURES: Mapping[str, Callable[[A.ResultsBundle, Path, list], None]] = {
    "fig_lag_grid": make_lag_grid,
    "fig_gain_sweep": make_gain_sweep,
    "fig_stability_boundary": make_stability_boundary,
    "fig_shock_propagation": make_shock_propagation,
    "fig_ladder_comparison": make_ladder_comparison,
    "fig_dual_track_frontier": make_dual_track_frontier,
    "fig_welfare_shortage_tradeoff": make_welfare_shortage_tradeoff,
    "fig_budget_line": make_budget_line,
    "fig_mb_decomposition": make_mb_decomposition,
    "fig_incidence": make_incidence,
    "fig_noise_only": make_noise_only,
}


def make_all(
    results_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    only: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build every figure that the available results support.

    Returns:
        ``{"figures": [...], "index": path, "missing": [...]}``.
    """
    _style()
    bundle = A.load_results(results_dir, with_runs=True)
    if out_dir is None:
        from planlab import paths

        out = paths.figures_dir()
    else:
        out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(only) if only is not None else list(FIGURES)
    written: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in wanted:
        func = FIGURES.get(name)
        if func is None:
            missing.append(f"{name}: unknown")
            continue
        try:
            func(bundle, out, written)
        except Exception as exc:  # a missing block must not kill the rest
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    # results/figures_index.csv is the index the paper's builder reads
    index_path = A._results_root(results_dir) / "figures_index.csv"
    pd.DataFrame(written).to_csv(index_path, index=False)
    (out / "figures_index.csv").write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
    if verbose:
        print(f"figures: {len(written)} written to {out}")
        print(f"index  : {index_path}")
        for entry in missing:
            print(f"  skipped {entry}")
    return {"figures": written, "index": str(index_path), "missing": missing}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the paper's figures from results/.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="figure names to build")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    make_all(args.results_dir, args.out_dir, only=args.only, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
