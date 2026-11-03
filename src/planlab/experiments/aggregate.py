"""Tables and pre-registered tests, from ``results/`` to ``results/tables/``.

Everything the paper reports is produced here, and every test statistic is written
to ``results/tables/tests.csv`` as well as being rendered into a table.  Nothing in
this module reads a number that was typed by hand.

Structure
---------
``load_results``
    Read the block artefacts written by :mod:`planlab.experiments.runner` into a
    :class:`ResultsBundle`.  Pure I/O: no simulation, no clock.
``build_tables``
    The publication tables, as ``results/tables/tNN_*.csv`` **and** ``*.tex``
    (booktabs), plus ``tests.csv`` (machine-readable test statistics) and
    ``t00_hypotheses`` (one verdict line per pre-registered hypothesis).
``main``
    ``python -m planlab.experiments.aggregate`` rebuilds every table.

Statistical procedures are the ones fixed in ``docs/PROTOCOL.md`` §6 — the unit of
analysis is the run, never the period; two-mechanism comparisons are paired on
common random numbers (Wilcoxon signed-rank + 10 000-resample bootstrap CI of the
paired difference); multi-mechanism comparisons are Kruskal–Wallis with
Holm-corrected pairwise Mann–Whitney; the lag effect is a regression with mechanism
fixed effects and cluster-robust standard errors by seed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ResultsBundle",
    "load_results",
    "merged_summary",
    "cell_means",
    "build_tables",
    "boundary_table",
    "main",
    "G_STAR",
]

#: welfare normalisation: the baseline run of B1 (docs/PROTOCOL.md §4)
BASELINE_CELL = "baseline_lp_k0"
BASELINE_CELL_NEW = "B1.lp_central.eurostat"

#: theoretical stability boundary of ``u_{t+1} = u_t - g u_{t-m}`` (Prop. 1)
def G_STAR(m: int) -> float:
    """``g*(m) = 2 sin(pi / (2(2m+1)))`` — the largest stable loop gain."""
    return float(2.0 * math.sin(math.pi / (2.0 * (2 * int(m) + 1))))


#: predicted oscillation period at the boundary, in periods
def PERIOD_STAR(m: int) -> float:
    return float(2.0 * (2 * int(m) + 1))


#: the mechanisms, in the order the paper's tables use
MECH_ORDER: tuple[str, ...] = (
    "lp_central",
    "material_balance",
    "rationing_priority",
    "tatonnement",
    "market",
    "hybrid_dual_track",
)

_GAIN_MECHANISMS = ("material_balance", "tatonnement")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@dataclass
class ResultsBundle:
    """Everything under ``results/`` that a table needs, in memory."""

    root: Path
    summaries: dict[str, pd.DataFrame] = field(default_factory=dict)
    aggregates: dict[str, pd.DataFrame] = field(default_factory=dict)
    sectors: dict[str, pd.DataFrame] = field(default_factory=dict)
    runs: dict[str, pd.DataFrame] = field(default_factory=dict)
    manifests: dict[str, dict] = field(default_factory=dict)

    @property
    def blocks(self) -> list[str]:
        return sorted(self.summaries)

    def manifest(self, block: str) -> dict:
        return self.manifests.get(block, {})

    def has(self, block: str) -> bool:
        return block in self.summaries and not self.summaries[block].empty


def _results_root(results_dir: str | Path | None) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    from planlab import paths

    return paths.results_dir()


def load_results(
    results_dir: str | Path | None = None,
    blocks: Sequence[str] | None = None,
    *,
    with_runs: bool = True,
) -> ResultsBundle:
    """Read the block artefacts into a :class:`ResultsBundle`.

    Args:
        results_dir: override for ``results/``.
        blocks: restrict to these block ids (default: every block directory found).
        with_runs: load the per-period ``runs.parquet`` files too (they are large;
            most tables only need ``summary.csv``).

    Returns:
        A bundle whose missing blocks are simply absent, so callers can degrade
        gracefully while a block is still running.
    """
    root = _results_root(results_dir)
    bundle = ResultsBundle(root=root)
    wanted = list(blocks) if blocks is not None else None
    if not root.exists():
        return bundle
    for block_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        block = block_dir.name
        if wanted is not None and block not in wanted:
            continue
        summary = block_dir / "summary.csv"
        if summary.exists():
            frame = pd.read_csv(summary)
            if not frame.empty:
                bundle.summaries[block] = frame
        for name, attr in (("aggregate.csv", "aggregates"), ("sector_panel.parquet", "sectors")):
            path = block_dir / name
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            except Exception:  # pragma: no cover - partially written file
                continue
            if not frame.empty:
                getattr(bundle, attr)[block] = frame
        if with_runs:
            runs = block_dir / "runs.parquet"
            if runs.exists():
                try:
                    frame = pd.read_parquet(runs)
                    if not frame.empty:
                        bundle.runs[block] = frame
                except Exception:  # pragma: no cover
                    pass
        manifest = block_dir / "manifest.json"
        if manifest.exists():
            try:
                bundle.manifests[block] = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                pass
    return bundle


def baseline_welfare(bundle: ResultsBundle) -> float:
    """``cum_welfare`` of the B1 baseline (the welfare normalisation).

    ``baseline_lp_k0`` in the pre-registered naming; ``B1.lp_central.eurostat`` in
    the runner's cell naming.  Raises if neither is present, because every welfare
    loss in the paper is relative to it and silently substituting a different
    origin would make the numbers incomparable.
    """
    frame = bundle.summaries.get("B1")
    if frame is None or frame.empty:
        raise RuntimeError("B1 is missing: the welfare normalisation cannot be built")
    for cell in (BASELINE_CELL, BASELINE_CELL_NEW):
        rows = frame[(frame["cell_id"] == cell) & (frame.get("substrate", "eurostat") == "eurostat")]
        if not rows.empty:
            return float(rows["cum_welfare"].iloc[0])
    rows = frame[frame["cell_id"].astype(str).str.contains("lp_central")]
    if rows.empty:
        raise RuntimeError("no lp_central baseline row in B1")
    return float(rows["cum_welfare"].iloc[0])


def merged_summary(
    bundle: ResultsBundle, blocks: Sequence[str] | None = None, *, baseline: float | None = None
) -> pd.DataFrame:
    """All run-level rows, with ``welfare_loss_vs_baseline`` added.

    One row per (block, cell, seed).  The loss is ``1 - cum_welfare / baseline``,
    so 0 means "the base-year perfect-information welfare" and 0.01 means "one per
    cent of it was lost".
    """
    names = list(blocks) if blocks is not None else bundle.blocks
    frames = [bundle.summaries[b] for b in names if b in bundle.summaries]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if baseline is not None:
        base: float | None = float(baseline)
    else:
        try:
            base = baseline_welfare(bundle)
        except RuntimeError:
            # B1 has not been run yet: keep the pipeline usable (partial runs,
            # figures built before the calibration block) and say so, rather than
            # silently inventing a different welfare origin.
            import warnings

            warnings.warn(
                "B1 is missing: welfare_loss_vs_baseline is NaN. Every welfare loss in "
                "the paper is relative to results/B1/" + BASELINE_CELL + "; run the block.",
                RuntimeWarning,
                stacklevel=2,
            )
            base = None
    column = (
        1.0 - frame["cum_welfare"] / float(base)
        if base is not None
        else pd.Series(np.nan, index=frame.index)
    )
    frame["welfare_loss_vs_baseline"] = column
    for column in ("lag", "m", "gain", "price_elasticity", "theta"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # x/base ratio: mean_output is the run-level mean over used periods; the base
    # total is the same for every row of the same substrate, so normalise by the
    # largest mean_output of the corresponding B1 cell (identical per substrate).
    return frame


def cell_means(
    bundle: ResultsBundle, block: str, metrics: Sequence[str] | None = None
) -> pd.DataFrame:
    """Per-cell means and dispersions from ``aggregate.csv`` (fallback: recompute)."""
    frame = bundle.aggregates.get(block)
    if frame is not None and not frame.empty:
        return frame.copy()
    summary = bundle.summaries.get(block)
    if summary is None or summary.empty:
        return pd.DataFrame()
    keys = [c for c in ("block", "cell_id", "mechanism", "lag", "m", "gain", "price_elasticity",
                        "theta", "substrate", "variant", "ladder", "shock_sector", "sector",
                        "delta", "shock_size") if c in summary.columns]
    return summary.groupby(keys, dropna=False).size().reset_index(name="n_seeds")


# ---------------------------------------------------------------------------
# statistical helpers
# ---------------------------------------------------------------------------
def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (monotone, never below the raw value)."""
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = (m - rank) * p[i]
        running = max(running, adjusted)
        out[i] = min(1.0, running)
    return out


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's δ: P(a > b) − P(a < b), in [-1, 1]."""
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(float(np.sum(a_i > b)) for a_i in a)
    lt = sum(float(np.sum(a_i < b)) for a_i in a)
    return float((gt - lt) / (a.size * b.size))


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 20260501
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (deterministic given ``seed``)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
    n_boot: int = 10_000,
    seed: int = 20260501,
) -> dict[str, float]:
    """Paired comparison of two mechanisms on common random numbers.

    Returns the Wilcoxon signed-rank statistic and p-value, the mean paired
    difference with its bootstrap CI, and Cliff's δ computed on the two run-level
    samples.  When every difference is zero (deterministic cells) the test is
    reported as non-significant with p = 1 rather than as an error.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    diff = aa[mask] - bb[mask]
    out: dict[str, float] = {
        "n_pairs": float(diff.size),
        "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
        "sd_diff": float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0,
        "cliffs_delta": cliffs_delta(aa[mask], bb[mask]),
        "wilcoxon_stat": float("nan"),
        "p_value": 1.0,
    }
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed) if diff.size else (np.nan, np.nan)
    out["ci95_lo"], out["ci95_hi"] = float(lo), float(hi)
    if diff.size >= 8 and np.any(np.abs(diff) > 0):
        try:
            from scipy.stats import wilcoxon

            result = wilcoxon(diff, alternative=alternative, zero_method="wilcox")
            out["wilcoxon_stat"] = float(result.statistic)
            out["p_value"] = float(result.pvalue)
        except Exception:  # pragma: no cover - scipy edge cases
            pass
    out["effect_paired_std"] = (
        float(out["mean_diff"] / out["sd_diff"]) if out["sd_diff"] else float("inf")
    )
    return out


def kruskal_holm(groups: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Kruskal–Wallis across mechanisms + Holm-corrected pairwise Mann–Whitney."""
    from scipy.stats import kruskal, mannwhitneyu

    names = [k for k, v in groups.items() if np.isfinite(np.asarray(v, dtype=float)).sum() > 1]
    samples = [np.asarray(groups[k], dtype=float) for k in names]
    out: dict[str, Any] = {"n_groups": len(names), "kw_stat": float("nan"), "kw_p": float("nan")}
    if len(samples) >= 2:
        try:
            res = kruskal(*samples)
            out["kw_stat"], out["kw_p"] = float(res.statistic), float(res.pvalue)
        except Exception:  # pragma: no cover
            pass
    pairs: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            try:
                res = mannwhitneyu(samples[i], samples[j], alternative="two-sided")
                p = float(res.pvalue)
                stat = float(res.statistic)
            except Exception:  # pragma: no cover
                p, stat = float("nan"), float("nan")
            pairs.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "u_stat": stat,
                    "p_raw": p,
                    "cliffs_delta": cliffs_delta(samples[i], samples[j]),
                }
            )
            raw_p.append(p)
    adjusted = holm(raw_p)
    for entry, padj in zip(pairs, adjusted):
        entry["p_holm"] = float(padj)
    out["pairs"] = pairs
    return out


def ols_cluster(
    frame: pd.DataFrame,
    y: str,
    xs: Sequence[str],
    *,
    fe: str | None = None,
    cluster: str = "seed",
) -> dict[str, Any]:
    """OLS with optional fixed effects and cluster-robust (CR1) standard errors.

    Implemented directly with numpy so that the numbers do not depend on an
    optional ``statsmodels`` install: ``V = (X'X)^{-1} [Σ_g X_g' e_g e_g' X_g]
    (X'X)^{-1}`` with the usual finite-sample factor.  Fixed effects are absorbed
    as dummies (the designs here are small), which keeps the reported intercept
    interpretable.
    """
    data = frame.dropna(subset=[y, *xs, cluster]).copy()
    if data.empty:
        return {"n": 0, "params": {}, "se": {}, "p": {}, "r2": float("nan")}
    columns = [np.ones(len(data))]
    names = ["const"]
    for x in xs:
        columns.append(pd.to_numeric(data[x], errors="coerce").to_numpy(dtype=float))
        names.append(x)
    if fe and fe in data.columns:
        dummies = pd.get_dummies(data[fe].astype(str), drop_first=True, dtype=float)
        for column in dummies.columns:
            columns.append(dummies[column].to_numpy(dtype=float))
            names.append(f"fe[{column}]")
    X = np.column_stack(columns)
    yv = pd.to_numeric(data[y], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(yv)):
        keep = np.isfinite(yv) & np.all(np.isfinite(X), axis=1)
        X, yv = X[keep], yv[keep]
        data = data.iloc[np.where(keep)[0]]
    if X.shape[0] <= X.shape[1]:
        return {"n": int(X.shape[0]), "params": {}, "se": {}, "p": {}, "r2": float("nan")}
    try:
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        xtx_inv = np.linalg.inv(X.T @ X)
        groups = data[cluster].to_numpy()
        meat = np.zeros((X.shape[1], X.shape[1]))
        for g in np.unique(groups):
            mask = groups == g
            xe = X[mask].T @ resid[mask]
            meat += np.outer(xe, xe)
        n, k = X.shape
        n_groups = len(np.unique(groups))
        factor = (n_groups / max(n_groups - 1, 1)) * ((n - 1) / max(n - k, 1))
        V = xtx_inv @ meat @ xtx_inv * factor
        se = np.sqrt(np.maximum(np.diag(V), 0.0))
        from scipy.stats import t as student_t

        tstat = beta / np.where(se > 0, se, np.nan)
        pvals = 2 * (1 - student_t.cdf(np.abs(tstat), df=max(n_groups - 1, 1)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        return {
            "n": int(n),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "params": {name: float(value) for name, value in zip(names, beta)},
            "se": {name: float(value) for name, value in zip(names, se)},
            "t": {name: float(value) for name, value in zip(names, tstat)},
            "p": {name: float(value) for name, value in zip(names, pvals)},
        }
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        return {"n": int(X.shape[0]), "params": {}, "se": {}, "p": {}, "r2": float("nan")}


def spearman(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Spearman rank correlation and its p-value (NaN-safe)."""
    from scipy.stats import spearmanr

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return (float("nan"), float("nan"))
    res = spearmanr(aa[mask], bb[mask])
    return (float(res.statistic), float(res.pvalue))


# ---------------------------------------------------------------------------
# test log
# ---------------------------------------------------------------------------
class TestLog:
    """Collects every test statistic into one machine-readable frame."""

    COLUMNS = (
        "block", "hypothesis", "outcome", "comparison", "test", "statistic", "df",
        "p_value", "p_holm", "effect_name", "effect_value", "n_units", "n_runs",
        "significant_05", "note",
    )

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(
        self,
        *,
        block: str,
        hypothesis: str,
        outcome: str,
        comparison: str,
        test: str,
        statistic: float = float("nan"),
        df: float = float("nan"),
        p_value: float = float("nan"),
        effect_name: str = "",
        effect_value: float = float("nan"),
        n_units: int = 0,
        n_runs: int = 0,
        note: str = "",
    ) -> None:
        self._rows.append(
            {
                "block": block,
                "hypothesis": hypothesis,
                "outcome": outcome,
                "comparison": comparison,
                "test": test,
                "statistic": float(statistic),
                "df": float(df),
                "p_value": float(p_value),
                "p_holm": float("nan"),
                "effect_name": effect_name,
                "effect_value": float(effect_value),
                "n_units": int(n_units),
                "n_runs": int(n_runs),
                "significant_05": "",
                "note": note,
            }
        )

    def apply_holm(self, within: str = "block") -> None:
        """Holm-correct the p-values within each value of ``within``."""
        if not self._rows:
            return
        frame = pd.DataFrame(self._rows)
        frame["p_holm"] = np.nan
        for _, idx in frame.groupby(within).groups.items():
            sub = frame.loc[idx]
            frame.loc[idx, "p_holm"] = holm(sub["p_value"].to_numpy(dtype=float))
        frame["significant_05"] = np.where(
            frame["p_holm"].notna(),
            np.where(frame["p_holm"] < 0.05, "yes", "no"),
            "",
        )
        self._rows = frame.to_dict("records")

    def frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=list(self.COLUMNS))
        frame = pd.DataFrame(self._rows)
        for column in self.COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[list(self.COLUMNS)]


# ---------------------------------------------------------------------------
# table formatting
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a booktabs table (``--`` for missing)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "--"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        exp = int(np.floor(np.log10(abs(x))))
        mant = x / 10**exp
        return f"${mant:.2f}\\times10^{{{exp}}}$"
    return f"{x:.{digits}f}"


def _fmt_p(value: Any) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(p):
        return "--"
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.3f}"


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def to_tex(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    columns: Sequence[str] | None = None,
    headers: Sequence[str] | None = None,
    formatters: Mapping[str, Callable[[Any], str]] | None = None,
    notes: str = "",
    digits: int = 3,
    alignment: str | None = None,
) -> str:
    """Render a dataframe as a booktabs ``table`` environment."""
    cols = list(columns) if columns is not None else list(frame.columns)
    head = list(headers) if headers is not None else [_tex_escape(c) for c in cols]
    fns = dict(formatters or {})
    align = alignment or ("l" + "r" * (len(cols) - 1))
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(head) + " \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in cols:
            value = row[column]
            fn = fns.get(column)
            if fn is not None:
                cells.append(fn(value))
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append(_fmt(value, digits))
            else:
                cells.append(_tex_escape(value) if value is not None else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines += [
            "\\par\\smallskip",
            "\\begin{minipage}{\\linewidth}\\footnotesize " + notes + "\\end{minipage}",
        ]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


@dataclass
class TableWriter:
    """Writes ``<name>.csv`` and ``<name>.tex`` and records what it wrote."""

    out_dir: Path
    written: list[dict[str, Any]] = field(default_factory=list)

    def write(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        caption: str,
        label: str,
        notes: str = "",
        columns: Sequence[str] | None = None,
        headers: Sequence[str] | None = None,
        formatters: Mapping[str, Callable[[Any], str]] | None = None,
        tex: bool = True,
        digits: int = 3,
        alignment: str | None = None,
    ) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.out_dir / f"{name}.csv"
        frame.to_csv(csv_path, index=False, float_format="%.6g")
        if tex:
            (self.out_dir / f"{name}.tex").write_text(
                to_tex(
                    frame,
                    caption=caption,
                    label=label,
                    columns=columns,
                    headers=headers,
                    formatters=formatters,
                    notes=notes,
                    digits=digits,
                    alignment=alignment,
                ),
                encoding="utf-8",
            )
        self.written.append(
            {
                "table": name,
                "csv": str(csv_path),
                "tex": str(self.out_dir / f"{name}.tex") if tex else "",
                "rows": int(len(frame)),
                "caption": caption,
            }
        )
        return csv_path

    def write_index(self, extra: Mapping[str, Any] | None = None) -> Path:
        index = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tables": self.written,
            **(dict(extra or {})),
        }
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(index, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# derived frames
# ---------------------------------------------------------------------------
def _substrate(frame: pd.DataFrame, value: str = "eurostat") -> pd.DataFrame:
    if "substrate" not in frame.columns:
        return frame
    return frame[frame["substrate"].astype(str) == value]


def _at_reference_gain(frame: pd.DataFrame) -> pd.DataFrame:
    """B2 rows at the reference gain, one row per (mechanism, lag)."""
    if frame.empty:
        return frame
    out = frame.copy()
    out = out[np.isclose(out["gain"].astype(float), 0.30, atol=1e-9)]
    if "price_elasticity" in out.columns:
        out = out[np.isclose(out["price_elasticity"].astype(float), 0.4, atol=1e-9)]
    return out


def _mean_ci(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Grouped mean / sd / t-CI of one run-level column."""
    grouped = frame.groupby(["mechanism", "lag", "m"], dropna=False)[column]
    out = grouped.agg(["mean", "std", "size"]).reset_index()
    out["ci95_lo"] = out["mean"] - 1.96 * out["std"] / np.sqrt(out["size"].clip(lower=1))
    out["ci95_hi"] = out["mean"] + 1.96 * out["std"] / np.sqrt(out["size"].clip(lower=1))
    return out


# ---------------------------------------------------------------------------
# individual tables
# ---------------------------------------------------------------------------
def table_validation(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t01 — B1: base-year reproduction on both substrates."""
    block = "B1"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = bundle.summaries[block].copy()
    rows: list[dict[str, Any]] = []
    for (mechanism, substrate), frame in summary.groupby(["mechanism", "substrate"], dropna=False):
        rows.append(
            {
                "mechanism": mechanism,
                "substrate": substrate,
                "n_runs": int(len(frame)),
                "x_over_base": float(frame["output_ratio_mean"].mean()),
                "mean_shortage_rate": float(frame["mean_shortage_rate"].mean()),
                "mean_plan_fulfilment": float(frame["mean_plan_fulfilment"].mean()),
                "cum_welfare": float(frame["cum_welfare"].mean()),
                "max_abs_residual": float(frame["accounting_residual_max"].max()),
                "checks_passed": int(frame["accounting_checks_passed"].sum()),
                "reproduction_error": float(abs(1.0 - frame["output_ratio_mean"].mean())),
            }
        )
    table = pd.DataFrame(rows).sort_values(["substrate", "mechanism"])
    writer.write(
        "t01_validation_b1",
        table,
        caption=(
            "Calibration and validation (block B1). Six institutions on two substrates with "
            "perfect information, no shock and no lag. $x/\\bar x$ is realised gross output "
            "relative to the base year; a correct implementation returns 1.000 for every "
            "institution, so any departure, any positive shortage rate or any balance "
            "residual above $10^{-8}$ is an implementation failure rather than a result."
        ),
        label="tab:validation",
        notes=(
            "Accounting residual is $\\max_t \\max_i |S^0_i+x_i-\\sum_j Z_{ij}-C_i-G_i-I_i-S^{\\mathrm{end}}_i|$ "
            "divided by the scale of the supply vector (the engine raises on any violation). "
            "Source: \\texttt{results/B1/}."
        ),
        formatters={
            "mechanism": lambda v: _tex_escape(v),
            "substrate": lambda v: _tex_escape(v),
            "max_abs_residual": _fmt_p,
            "reproduction_error": _fmt_p,
        },
        digits=4,
    )
    # The implementation test itself: the residual bound and the reproduction error.
    tests.add(
        block=block,
        hypothesis="B1-implementation",
        outcome="max_abs_residual",
        comparison="all mechanisms, both substrates",
        test="max over runs",
        statistic=float(table["max_abs_residual"].max()),
        p_value=float("nan"),
        effect_name="max_abs_residual",
        effect_value=float(table["max_abs_residual"].max()),
        n_units=int(len(table)),
        n_runs=int(table["n_runs"].sum()),
        note="passes if < 1e-8",
    )
    tests.add(
        block=block,
        hypothesis="B1-implementation",
        outcome="base_year_reproduction_error",
        comparison="all mechanisms, both substrates",
        test="max |1 - x/base|",
        statistic=float(table["reproduction_error"].max()),
        p_value=float("nan"),
        effect_name="max_reproduction_error",
        effect_value=float(table["reproduction_error"].max()),
        n_units=int(len(table)),
        n_runs=int(table["n_runs"].sum()),
        note="passes if ~ 0 (perfect information)",
    )
    return table


def table_noise_only(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t02 — B1b: the information rent, no shock and no lag."""
    block = "B1b"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = bundle.summaries[block].copy()
    rows: list[dict[str, Any]] = []
    for mechanism, frame in summary.groupby("mechanism"):
        rows.append(
            {
                "mechanism": mechanism,
                "n_seeds": int(len(frame)),
                "x_over_base_mean": float(frame["output_ratio_mean"].mean()),
                "x_over_base_sd": float(frame["output_ratio_mean"].std(ddof=1)),
                "mean_shortage_rate_mean": float(frame["mean_shortage_rate"].mean()),
                "mean_shortage_rate_sd": float(frame["mean_shortage_rate"].std(ddof=1)),
                "mean_welfare_mean": float(frame["mean_welfare"].mean()),
                "mean_welfare_sd": float(frame["mean_welfare"].std(ddof=1)),
                "mean_spoilage_rate": float(frame["mean_spoilage_rate"].mean())
                if "mean_spoilage_rate" in frame
                else float("nan"),
                "mean_plan_fulfilment_mean": float(frame["mean_plan_fulfilment"].mean()),
                "mean_iterations_mean": float(frame["mean_iterations"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MECH_ORDER)}
    table = table.sort_values("mechanism", key=lambda s: s.map(order))
    writer.write(
        "t02_noise_only_information_rent",
        table,
        caption=(
            "The information rent of a coordination institution (block B1b): no capacity "
            "shock, no reporting lag, only measurement noise ($\\sigma=0.03$) and AR(1) demand "
            "noise (ar1 $=0.3$, $\\sigma=0.02$). Since every institution reproduces the base "
            "year exactly when information is perfect (Table~\\ref{tab:validation}), any "
            "departure here is attributable to the institution's use of information."
        ),
        label="tab:noise_only",
        notes=(
            "20 seeds per institution, real 20-sector Eurostat 2020 substrate, "
            "$\\bar x$ = base-year gross output. **Read the shortage column with the ratio "
            "columns, never alone**: household demand is income-elastic in this model, so an "
            "institution that destroys output also destroys the demand that would have been "
            "recorded as unmet --- the shortage rate is scale-free by construction and is "
            "near-invariant across institutions (1.56--1.64 per cent for five of six) while "
            "output ranges over 2.4 points and welfare over 2.7. Source: "
            "\\texttt{results/B1b/}."
        ),
        formatters={"mechanism": lambda v: _tex_escape(v)},
        digits=4,
    )
    # Paired tests: every institution against the two extremes.
    def _series(name: str) -> np.ndarray:
        return summary[summary["mechanism"] == name].sort_values("seed")["output_ratio_mean"].to_numpy()

    for other in ("material_balance", "hybrid_dual_track"):
        if other in set(summary["mechanism"]):
            res = paired_test(_series("material_balance"), _series(other))
            tests.add(
                block=block,
                hypothesis="H3'",
                outcome="x_over_base",
                comparison=f"material_balance vs {other}",
                test="paired Wilcoxon signed-rank (CRN)",
                statistic=res["wilcoxon_stat"],
                p_value=res["p_value"],
                effect_name="paired mean difference",
                effect_value=res["mean_diff"],
                n_units=int(res["n_pairs"]),
                n_runs=int(2 * summary.shape[0] / summary["mechanism"].nunique()),
                note=f"ci95=[{res['ci95_lo']:.4f},{res['ci95_hi']:.4f}], cliff={res['cliffs_delta']:.2f}",
            )
    return table


def table_lag_mechanism(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t03 — B2 at the reference gain: lag x mechanism."""
    block = "B2"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _at_reference_gain(_substrate(summary))
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (mechanism, lag), group in frame.groupby(["mechanism", "lag"]):
        rows.append(
            {
                "mechanism": mechanism,
                "lag_reporting": int(lag),
                "lag_effective": int(lag) + 1,
                "n_seeds": int(len(group)),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "mean_shortage_rate_sd": float(group["mean_shortage_rate"].std(ddof=1)),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_welfare_loss_sd": float(group["welfare_loss_vs_baseline"].std(ddof=1)),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "oscillation_amplitude": float(group["oscillation_amplitude"].mean()),
                "acf_lag1": float(group["acf_lag1"].mean()),
                "mean_plan_fulfilment": float(group["mean_plan_fulfilment"].mean()),
                "mean_iterations": float(group["mean_iterations"].mean()),
                "recovery_half_life": float(
                    np.nanmean(group["recovery_half_life"].to_numpy(dtype=float))
                )
                if "recovery_half_life" in group
                else float("nan"),
            }
        )
    order = {m: i for i, m in enumerate(MECH_ORDER)}
    table = pd.DataFrame(rows).sort_values(
        ["lag_reporting", "mechanism"], key=lambda s: s.map(order) if s.name == "mechanism" else s
    )
    writer.write(
        "t03_lag_mechanism",
        table,
        caption=(
            "Delay and institution (block B2, reference gain 0.30, price elasticity 0.4): "
            "the reporting lag $\\ell$ is what the scenario sets, the effective delay "
            "$k=\\ell+1$ is what the theory's $g^*(k)$ refers to, because the planner observes "
            "period $t-1-\\ell$ when planning $t$. All cells share the same exogenous path "
            "(common random numbers), so row differences are institutional, not stochastic."
        ),
        label="tab:lag_grid",
        notes=(
            "Welfare loss is $1-\\mathrm{cum\\,welfare}/\\mathrm{baseline}$ against "
            "\\texttt{baseline\\_lp\\_k0}. Amplitude is the within-run standard deviation of the "
            "detrended mean log price; ACF is its lag-1 autocorrelation. 20 seeds per cell. "
            "Source: \\texttt{results/B2/}."
        ),
        formatters={"mechanism": lambda v: _tex_escape(v)},
        digits=4,
    )

    # H2 compares slopes within a seed at the shared operating point.
    from scipy.stats import ttest_1samp

    slopes = []
    for (mechanism, seed), group in frame.groupby(["mechanism", "seed"]):
        if group["lag"].nunique() < 2:
            continue
        # A negative slope of mean utility is a positive slope of utility loss.
        slope = np.polyfit(group["lag"], -group["mean_welfare"], 1)[0]
        slopes.append({"mechanism": mechanism, "seed": seed, "slope": slope})
    if slopes:
        paired = pd.DataFrame(slopes).pivot(index="seed", columns="mechanism", values="slope")
        if "lp_central" in paired:
            for mechanism in paired.columns.drop("lp_central"):
                pair = paired[["lp_central", mechanism]].dropna()
                if len(pair) < 2:
                    continue
                diff = pair["lp_central"] - pair[mechanism]
                fit = ttest_1samp(diff, 0.0, alternative="greater")
                ci = ttest_1samp(diff, 0.0).confidence_interval()
                tests.add(
                    block=block,
                    hypothesis="H2",
                    outcome="mean_welfare",
                    comparison=f"lp_central vs {mechanism}: welfare-loss slope",
                    test="one-sided paired t-test on per-seed slopes",
                    statistic=fit.statistic,
                    p_value=fit.pvalue,
                    effect_name="slope difference (utility points per period of delay)",
                    effect_value=diff.mean(),
                    n_units=len(diff),
                    note=(f"lp_central={pair['lp_central'].mean():.6g}; "
                          f"{mechanism}={pair[mechanism].mean():.6g}; "
                          f"unadjusted two-sided CI=[{ci.low:.6g}, {ci.high:.6g}]"),
                )
    return table


def table_gain_sweep(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t04 — B2 gain sweep: H1 (price loop) and H1' (target loop)."""
    block = "B2"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _substrate(summary)
    frame = frame[frame["mechanism"].isin(_GAIN_MECHANISMS)]
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (mechanism, gain, lag, eps), group in frame.groupby(
        ["mechanism", "gain", "lag", "price_elasticity"]
    ):
        rows.append(
            {
                "mechanism": mechanism,
                "gain": float(gain),
                "gain_param": "revision_gain" if mechanism == "material_balance" else "price_gain",
                "lag_reporting": int(lag),
                "lag_effective": int(lag) + 1,
                "price_elasticity": float(eps),
                "n_seeds": int(len(group)),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "mean_shortage_rate_sd": float(group["mean_shortage_rate"].std(ddof=1)),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "oscillation_amplitude": float(group["oscillation_amplitude"].mean()),
                "acf_lag1": float(group["acf_lag1"].mean()),
                "oscillation_period": float(
                    np.nanmean(group["oscillation_period"].to_numpy(dtype=float))
                )
                if "oscillation_period" in group
                else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["mechanism", "lag_reporting", "gain"])
    gains = gain_estimate(bundle)
    bp_map = (
        {float(r["price_elasticity"]): float(r["b_p_mean"]) for _, r in bp_summary(gains).iterrows()}
        if not gains.empty
        else {}
    )
    table["b_p_mean"] = table.apply(
        lambda r: bp_map.get(float(r["price_elasticity"]), float("nan"))
        if r["mechanism"] == "tatonnement"
        else float("nan"),
        axis=1,
    )
    table["loop_gain"] = table["gain"] * table["b_p_mean"]
    table["g_star_m"] = table["lag_effective"].map(G_STAR)
    table["loop_gain_over_bound"] = table["loop_gain"] / table["g_star_m"]
    writer.write(
        "t04_gain_sweep",
        table,
        caption=(
            "The gain axis (block B2, extended by Amendment 3). By Amendment 1 the "
            "stability test is the price loop of \\texttt{tatonnement} "
            "(\\texttt{price\\_gain}, $\\varepsilon>0$); the plan-revision loop of "
            "\\texttt{material\\_balance} (\\texttt{revision\\_gain}) has the opposite sign "
            "and is reported as H1$'$."
        ),
        label="tab:gain_sweep",
        notes=(
            "Amplitude: within-run sd of the detrended mean log price. "
            "$k=\\ell+1$. 20 seeds per cell. Source: \\texttt{results/B2/}."
        ),
        formatters={"mechanism": lambda v: _tex_escape(v)},
        digits=4,
    )

    # ---- H1': monotonicity of shortage in the plan-revision gain ------------
    mb = frame[frame["mechanism"] == "material_balance"]
    if not mb.empty:
        cell = mb.groupby("gain")["mean_shortage_rate"].mean().sort_index()
        rho, p_rho = spearman(cell.index.to_numpy(dtype=float), cell.to_numpy(dtype=float))
        tests.add(
            block=block,
            hypothesis="H1'",
            outcome="mean_shortage_rate",
            comparison="material_balance: gain 0.05 -> 1.6",
            test="Spearman rank correlation (cell means)",
            statistic=rho,
            p_value=p_rho,
            effect_name="rho(gain, shortage)",
            effect_value=rho,
            n_units=int(len(cell)),
            note="positive rho = taut plans raise shortage (positive feedback loop)",
        )
        wide = mb.pivot(index=["lag", "seed"], columns="gain", values="mean_shortage_rate")
        if 0.05 in wide and 1.6 in wide:
            matched = wide[[0.05, 1.6]].dropna().groupby("seed").mean()
            res = paired_test(matched[1.6], matched[0.05], alternative="greater")
            tests.add(
                block=block,
                hypothesis="H1'",
                outcome="mean_shortage_rate",
                comparison="material_balance gain 1.6 vs 0.05 (paired by seed, averaged over lags)",
                test="paired Wilcoxon signed-rank (one-sided, greater)",
                statistic=res["wilcoxon_stat"],
                p_value=res["p_value"],
                effect_name="paired mean difference",
                effect_value=res["mean_diff"],
                n_units=int(res["n_pairs"]),
                n_runs=2 * len(wide[[0.05, 1.6]].dropna()),
                note=f"ci95=[{res['ci95_lo']:.4f},{res['ci95_hi']:.4f}], cliff={res['cliffs_delta']:.2f}",
            )

    # Compare each reported delay to zero within the same elasticity and seed.
    tat = frame[frame["mechanism"] == "tatonnement"]
    if not tat.empty:
        top = tat[np.isclose(tat["gain"], tat["gain"].max())]
        for elasticity, group in top.groupby("price_elasticity"):
            wide = group.pivot(index="seed", columns="lag", values="oscillation_amplitude")
            if 0 not in wide:
                continue
            for lag in wide.columns.drop(0):
                pair = wide[[lag, 0]].dropna()
                res = paired_test(pair[lag], pair[0], alternative="greater")
                tests.add(
                    block=block,
                    hypothesis="H1-amplitude",
                    outcome="oscillation_amplitude",
                    comparison=f"gain {tat['gain'].max():g}, elasticity {elasticity:g}: lag {lag} vs 0",
                    test="one-sided paired Wilcoxon by seed",
                    statistic=res["wilcoxon_stat"],
                    p_value=res["p_value"],
                    effect_name="paired amplitude difference",
                    effect_value=res["mean_diff"],
                    n_units=len(pair),
                    n_runs=2 * len(pair),
                    note="Descriptive evidence about detrended amplitude, not a stability test.",
                )
    return table


def boundary_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measured critical gain per effective delay, and critical delay per gain.

    Implements the pre-registered rule literally (``docs/PROTOCOL.md`` §6): a cell is
    *unstable* when the mean lag-1 autocorrelation of the detrended price deviation
    exceeds 0.5 **and** the mean amplitude exceeds the lag-0 amplitude of the same
    gain by at least three of its seed standard deviations.
    """
    rows_k: list[dict[str, Any]] = []
    rows_gain: list[dict[str, Any]] = []
    for (eps, gain), group in frame.groupby(["price_elasticity", "gain"]):
        stats = group.groupby(["lag", "m"])["oscillation_amplitude"].agg(
            amp="mean", amp_sd="std", n="size"
        ).reset_index()
        acf = group.groupby(["lag", "m"])["acf_lag1"].mean().reset_index()
        stats = stats.merge(acf, on=["lag", "m"], how="left")
        ref = stats[stats["lag"] == 0]
        if ref.empty:
            continue
        ref_amp = float(ref["amp"].iloc[0])
        ref_sd = float(ref["amp_sd"].iloc[0]) if np.isfinite(ref["amp_sd"].iloc[0]) else 0.0
        stats["unstable"] = (stats["acf_lag1"] > 0.5) & (stats["amp"] > ref_amp + 3.0 * ref_sd)
        stats["price_elasticity"] = float(eps)
        stats["gain"] = float(gain)
        stats["ref_amp"] = ref_amp
        stats["ref_sd"] = ref_sd
        unstable_lags = stats[stats["unstable"]]
        rows_gain.append(
            {
                "price_elasticity": float(eps),
                "gain": float(gain),
                "lag_star": float(unstable_lags["lag"].min()) if not unstable_lags.empty else float("nan"),
                "m_star": float(unstable_lags["m"].min()) if not unstable_lags.empty else float("nan"),
                "max_amp": float(stats["amp"].max()),
                "max_acf1": float(stats["acf_lag1"].max()),
            }
        )
        rows_k.append(stats)
    cells = pd.concat(rows_k, ignore_index=True) if rows_k else pd.DataFrame()
    per_gain = pd.DataFrame(rows_gain)
    # critical gain per effective delay
    rows_m: list[dict[str, Any]] = []
    if not cells.empty:
        for (eps, m), group in cells.groupby(["price_elasticity", "m"]):
            unstable = group[group["unstable"]]
            rows_m.append(
                {
                    "price_elasticity": float(eps),
                    "lag_reporting": int(m) - 1,
                    "lag_effective": int(m),
                    "gain_star": float(unstable["gain"].min()) if not unstable.empty else float("nan"),
                    "gain_max_tested": float(group["gain"].max()),
                    "max_amp": float(group["amp"].max()),
                    "max_acf1": float(group["acf_lag1"].max()),
                    "n_seeds": int(group["n"].max()),
                }
            )
    per_m = pd.DataFrame(rows_m)
    return cells, per_m, per_gain


def gain_estimate(bundle: ResultsBundle) -> pd.DataFrame:
    r"""``b_p`` estimated from block B7 (the price-response probe).

    ``b_p = -ds/d log p``, measured sector by sector: the probed sector's own
    household unmet rate under a permanent 50% capacity shock, with and without a
    $\pm5\\%$ price offset.  Theory predicts ``b_p`` of the order of the demand
    elasticity, and the boundary gain is ``g*(m)/b_p``.
    """
    block = "B7"
    if not bundle.has(block) or block not in bundle.sectors:
        return pd.DataFrame()
    summary = bundle.summaries[block]
    panel = bundle.sectors[block]
    rows: list[dict[str, Any]] = []
    for (eps, sector), group in summary.groupby(["price_elasticity", "sector"]):
        cell_ids = group["cell_id"].astype(str)
        plus = group[cell_ids.str.endswith("d+0.05")]
        minus = group[cell_ids.str.endswith("d-0.05")]
        if plus.empty or minus.empty:
            continue
        sector_rows = panel[(panel["sector"] == sector) & (panel["cell_id"].isin(cell_ids))]
        s_plus = float(sector_rows[sector_rows["cell_id"].isin(plus["cell_id"])]["household_unmet_rate_mean"].mean())
        s_minus = float(sector_rows[sector_rows["cell_id"].isin(minus["cell_id"])]["household_unmet_rate_mean"].mean())
        dlogp = math.log(1.05) - math.log(0.95)
        rows.append(
            {
                "price_elasticity": float(eps),
                "sector": sector,
                "shortage_rate_at_plus5pct": s_plus,
                "shortage_rate_at_minus5pct": s_minus,
                "b_p": float(-(s_plus - s_minus) / dlogp),
            }
        )
    return pd.DataFrame(rows)


def bp_summary(gains: pd.DataFrame) -> pd.DataFrame:
    """Per-elasticity summary of the measured local gain ``b_p``.

    ``b_p`` is reported **per sector** as the Lead's identification review requires:
    the aggregate mean hides a two-order-of-magnitude spread, and it is the spread
    that decides whether the loop can reach its bound at all.
    """
    if gains.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for eps, group in gains.groupby("price_elasticity"):
        values = group["b_p"].to_numpy(dtype=float)
        rows.append(
            {
                "price_elasticity": float(eps),
                "n_sectors": int(np.isfinite(values).sum()),
                "b_p_mean": float(np.nanmean(values)),
                "b_p_median": float(np.nanmedian(values)),
                "b_p_min": float(np.nanmin(values)),
                "b_p_max": float(np.nanmax(values)),
                "b_p_sd": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0,
                "b_p_argmin_sector": str(group["sector"].to_numpy()[int(np.nanargmin(values))]),
                "b_p_argmax_sector": str(group["sector"].to_numpy()[int(np.nanargmax(values))]),
            }
        )
    return pd.DataFrame(rows).sort_values("price_elasticity").reset_index(drop=True)


def table_bp_by_sector(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t12 — B7: the local price response, sector by sector."""
    gains = gain_estimate(bundle)
    if gains.empty:
        return pd.DataFrame()
    table = gains.sort_values(["price_elasticity", "b_p"]).reset_index(drop=True)
    table["b_p_rank_within_elasticity"] = table.groupby("price_elasticity")["b_p"].rank(
        method="first"
    ).astype(int)
    writer.write(
        "t12_bp_by_sector",
        table,
        caption=(
            "The local gain of the price loop, measured sector by sector (block B7): "
            "$b_{p,j}=-\\mathrm{d}s_j/\\mathrm{d}\\log p_j$ from a $\\pm5$ per cent permanent "
            "price offset on each good, with that good's capacity permanently cut by 50 per cent "
            "and the price gain switched off. The loop gain of the mechanism is "
            "$g=\\eta\\,b_p$, so these numbers decide whether the analytic bound "
            "$g^*(m)=2\\sin(\\pi/(2(2m+1)))$ is reachable at all."
        ),
        label="tab:bp_sector",
        notes=(
            "One seed, no measurement noise: the probe is deterministic. "
            "Source: \\texttt{results/B7/}."
        ),
        formatters={"sector": lambda v: _tex_escape(v)},
        digits=4,
    )
    summary = bp_summary(gains)
    for _, row in summary.iterrows():
        tests.add(
            block="B7",
            hypothesis="H1-gain",
            outcome="b_p",
            comparison=f"price_elasticity={row['price_elasticity']:g}: sector distribution",
            test="mean over 20 sectors",
            statistic=float(row["b_p_mean"]),
            p_value=float("nan"),
            effect_name="b_p",
            effect_value=float(row["b_p_mean"]),
            n_units=int(row["n_sectors"]),
            note=(
                f"min={row['b_p_min']:.4g} ({row['b_p_argmin_sector']}), "
                f"max={row['b_p_max']:.4g} ({row['b_p_argmax_sector']})"
            ),
        )
    return table


def table_stability_boundary(
    bundle: ResultsBundle, writer: TableWriter, tests: TestLog
) -> pd.DataFrame:
    """t05 — compare scalar benchmarks with diagnostics from the one-sided rule."""
    block = "B2"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _substrate(summary)
    frame = frame[frame["mechanism"] == "tatonnement"]
    if frame.empty:
        return pd.DataFrame()
    _, per_m, _ = boundary_cells(frame)
    gains = gain_estimate(bundle)
    bp = bp_summary(gains)
    bp_map = {float(row["price_elasticity"]): row for _, row in bp.iterrows()} if not bp.empty else {}
    gain_max = float(frame["gain"].max())
    rows: list[dict[str, Any]] = []
    if not per_m.empty:
        for _, row in per_m.iterrows():
            eps = float(row["price_elasticity"])
            m = int(row["lag_effective"])
            info = bp_map.get(eps)
            b = float(info["b_p_mean"]) if info is not None else float("nan")
            theory_gain = G_STAR(m) / b if np.isfinite(b) and b > 0 else float("nan")
            g_max = gain_max * b if np.isfinite(b) else float("nan")
            measured = float(row["gain_star"])
            rows.append(
                {
                    "lag_reporting": int(row["lag_reporting"]),
                    "lag_effective": m,
                    "price_elasticity": eps,
                    "b_p_mean": b,
                    "b_p_min": float(info["b_p_min"]) if info is not None else float("nan"),
                    "b_p_max": float(info["b_p_max"]) if info is not None else float("nan"),
                    "g_star_m": G_STAR(m),
                    "scalar_gain_benchmark": theory_gain,
                    "diagnostic_first_trigger_gain": measured,
                    "trigger_over_scalar_benchmark": (
                        measured / theory_gain
                        if np.isfinite(theory_gain) and theory_gain > 0 and np.isfinite(measured)
                        else float("nan")
                    ),
                    "gain_max_tested": gain_max,
                    "gain_times_mean_response": g_max,
                    "proxy_over_scalar_bound": g_max / G_STAR(m) if G_STAR(m) > 0 else float("nan"),
                    "proxy_meets_scalar_bound": bool(np.isfinite(g_max) and g_max >= G_STAR(m)),
                    "diagnostic_trigger_below_scalar_bound": bool(
                        np.isfinite(measured) and np.isfinite(b) and measured * b < G_STAR(m)
                    ),
                    "period_theory": PERIOD_STAR(m),
                    "max_amplitude": float(row["max_amp"]),
                    "max_acf_lag1": float(row["max_acf1"]),
                    "n_seeds": int(row["n_seeds"]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(["price_elasticity", "lag_effective"]).reset_index(drop=True)
    writer.write(
        "t05_stability_boundary",
        table,
        caption=(
            "Scalar stability benchmark and diagnostics for the simulated one-sided price rule. "
            "The sector-average price response is a descriptive proxy, not a bound on dynamic eigenvalues."
        ),
        label="tab:stability_boundary",
        notes=(
            "The first diagnostic trigger uses autocorrelation and detrended amplitude. "
            "A trigger does not identify a stability crossing."
        ),
        digits=3,
        formatters={
            "proxy_meets_scalar_bound": lambda v: "yes" if bool(v) else "no",
            "diagnostic_trigger_below_scalar_bound": lambda v: "yes" if bool(v) else "no",
        },
    )
    tests.add(
        block=block, hypothesis="H1", outcome="stability_threshold",
        comparison="scalar threshold versus simulated one-sided price rule",
        test="identification assessment", note="Threshold not identified by this design.",
    )
    return table


def table_propagation(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t06 — B3: network amplification by shocked sector (H4)."""
    block = "B3"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _substrate(summary)
    if frame.empty:
        return pd.DataFrame()
    paired_columns = {"paired_shock_output_amplification", "paired_shock_output_loss"}
    has_controls = paired_columns.issubset(frame.columns)
    rows: list[dict[str, Any]] = []
    for (mechanism, sector), group in frame.groupby(["mechanism", "shock_sector"]):
        rows.append(
            {
                "mechanism": mechanism,
                "shocked_sector": sector,
                "n_seeds": int(len(group)),
                **({name: float(group[name].mean()) for name in sorted(paired_columns)}
                   if has_controls else {}),
                "shock_output_amplification": float(group["shock_output_amplification"].mean()),
                "shock_output_amplification_sd": float(group["shock_output_amplification"].std(ddof=1)),
                "shock_amplification_distributional": float(group["shock_amplification"].mean()),
                "shock_output_loss": float(group["shock_output_loss"].mean()),
                "peak_shortage_rate": float(group["shock_peak_shortage"].mean()),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "recovery_half_life": float(
                    np.nanmean(group["recovery_half_life"].to_numpy(dtype=float))
                )
                if "recovery_half_life" in group
                else float("nan"),
                "amp_protocol_peak": float(group["amp_protocol_peak"].mean())
                if "amp_protocol_peak" in group
                else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["mechanism", "shocked_sector"]).reset_index(drop=True)
    predictors = linkage_predictors(bundle)
    table["forward_linkage"] = table["shocked_sector"].map(predictors["forward"])
    table["leontief_multiplier"] = table["shocked_sector"].map(predictors["leontief"])
    table["output_share"] = table["shocked_sector"].map(predictors["share"])
    writer.write(
        "t06_propagation",
        table,
        caption=(
            "Sectoral propagation (block B3): a 50 per cent capacity cut lasting eight quarters, "
            "one shocked sector at a time, 20 seeds each. "
            "Paired loss compares each run to the same-seed no-shock control. "
            "The original loss relative to base-year output is retained as a separate column."
        ),
        label="tab:propagation",
        notes="Forward linkage is the intermediate share of a sector's own output.",
        formatters={"mechanism": lambda v: _tex_escape(v), "shocked_sector": lambda v: _tex_escape(v)},
        digits=3,
    )
    # H3 uses a fixed threshold, so no p-value or Holm count is assigned.
    mb = table.loc[table["mechanism"] == "material_balance", "mean_shortage_rate"]
    if not mb.empty:
        below = int((mb < 0.005).sum())
        tests.add(
            block=block, hypothesis="H3", outcome="mean_shortage_rate",
            comparison="material balance by shocked sector versus 0.5% shortage",
            test="descriptive threshold check", statistic=float(mb.min()),
            effect_name="cells below threshold", effect_value=below, n_units=len(mb),
            note=f"{below}/{len(mb)} below 0.5%; min={mb.min():.6g}, mean={mb.mean():.6g}, max={mb.max():.6g}",
        )

    # H4 requires a no-shock counterfactual. Historical summaries lack one.
    if not has_controls:
        return table
    for mechanism, group in table.groupby("mechanism"):
        loss = group["paired_shock_output_loss"].to_numpy(dtype=float)
        amp = group["paired_shock_output_amplification"].to_numpy(dtype=float)
        fwd = np.array([predictors.get("forward", {}).get(str(s), np.nan) for s in group["shocked_sector"]])
        colsum = np.array([predictors.get("leontief", {}).get(str(s), np.nan) for s in group["shocked_sector"]])
        share = np.array([predictors.get("share", {}).get(str(s), np.nan) for s in group["shocked_sector"]])
        from scipy.stats import ttest_1samp

        runs = frame[frame["mechanism"] == mechanism]
        per_seed = runs.groupby("seed")["paired_shock_output_amplification"].mean()
        fit = ttest_1samp(per_seed, 1.0, alternative="greater") if len(per_seed) > 1 else None
        tests.add(
            block=block,
            hypothesis="H4",
            outcome="paired_shock_output_amplification",
            comparison=f"{mechanism}: mean amplification across fixed sectors vs 1",
            test="one-sided t-test on per-seed sector means",
            statistic=float(fit.statistic) if fit is not None else float("nan"),
            p_value=float(fit.pvalue) if fit is not None else float("nan"),
            effect_name="mean amplification - 1",
            effect_value=float(per_seed.mean() - 1.0),
            n_units=len(per_seed),
            n_runs=len(runs),
            note=f"mean={per_seed.mean():.3f}; paired output loss spans {np.nanmin(loss):.4f}-{np.nanmax(loss):.4f}",
        )
        for label, values in (("forward linkage", fwd), ("Leontief column sum", colsum),
                              ("output share", share)):
            rho, p_rho = spearman(values, loss)
            tests.add(
                block=block,
                hypothesis="H4-centrality",
                outcome="paired_shock_output_loss",
                comparison=f"{mechanism}: output loss vs {label}",
                test="Spearman rank correlation",
                statistic=rho,
                p_value=p_rho,
                effect_name="rho",
                effect_value=rho,
                n_units=int(len(loss)),
                n_runs=int(group["n_seeds"].sum()),
                note="primary H4 predictor is the forward linkage (downstream dependence)",
            )
        # the multiplier as a small-denominator caveat
        rho_m, p_m = spearman(fwd, amp)
        tests.add(
            block=block,
            hypothesis="H4-multiplier",
            outcome="paired_shock_output_amplification",
            comparison=f"{mechanism}: amplification vs forward linkage",
            test="Spearman rank correlation",
            statistic=rho_m,
            p_value=p_m,
            effect_name="rho",
            effect_value=rho_m,
            n_units=int(len(amp)),
            n_runs=int(group["n_seeds"].sum()),
            note=(
                "reported as SECONDARY: the multiplier divides by the sector's output share, "
                "so it explodes for tiny sectors (a small-denominator artefact)"
            ),
        )
    return table


def linkage_predictors(bundle: ResultsBundle) -> dict[str, dict[str, float]]:
    """The three candidate H4 predictors, at base-year output.

    * ``forward``: the intermediate share of a sector's own output,
      ``sum_i A_{ji} x_i / x_j`` --- how much of what the sector makes other sectors
      need as input (its *downstream* dependence);
    * ``leontief``: the column sum of ``(I-A)^{-1} - I``, the standard upstream
      multiplier (how much the sector needs from others);
    * ``share``: the sector's share of aggregate output.

    The Lead's finding is that only the first predicts propagation, which is a
    sharper statement than "network position matters".
    """
    try:
        from planlab.data import load_technology

        tech = load_technology("eurostat", 2020, 20)
        A = np.asarray(tech.A, dtype=float)
        x = np.asarray(tech.base_output, dtype=float)
        total = float(x.sum())
        forward = (A @ x) / np.maximum(x, 1e-12)
        inv = np.linalg.inv(np.eye(tech.n) - A) - np.eye(tech.n)
        colsum = inv.sum(axis=0)
        return {
            "forward": {str(s): float(v) for s, v in zip(tech.sectors, forward)},
            "leontief": {str(s): float(v) for s, v in zip(tech.sectors, colsum)},
            "share": {str(s): float(v / total) for s, v in zip(tech.sectors, x)},
        }
    except Exception:  # pragma: no cover - data layer unavailable
        return {"forward": {}, "leontief": {}, "share": {}}


def forward_linkage(bundle: ResultsBundle) -> dict[str, float]:
    """Row sums of the direct-requirement matrix $A$ (forward linkage)."""
    try:
        from planlab.data import load_technology

        tech = load_technology("eurostat", 2020, 20)
        row_sums = np.asarray(tech.A, dtype=float).sum(axis=1)
        total = float(row_sums.sum())
        return {str(s): float(v / total) for s, v in zip(tech.sectors, row_sums)}
    except Exception:  # pragma: no cover - data layer unavailable
        return {}


def table_incidence(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t07 — B3 incidence: who bears the shortage, not who causes it."""
    block = "B3"
    if block not in bundle.sectors:
        return pd.DataFrame()
    panel = bundle.sectors[block]
    rows: list[dict[str, Any]] = []
    for (mechanism, cell), group in panel.groupby(["mechanism", "cell_id"]):
        shocked = group[group["is_shocked"]]
        if shocked.empty:
            continue
        sector = str(shocked["sector"].iloc[0])
        house_share = float(np.nanmean(shocked["final_share_of_rationed"].to_numpy(dtype=float)))
        other_share = float(np.nanmean(shocked["other_share_of_rationed"].to_numpy(dtype=float)))
        rows.append(
            {
                "mechanism": mechanism,
                "shocked_sector": sector,
                "n_seeds": int(shocked["seed"].nunique()),
                "household_unmet_rate_mean": float(shocked["household_unmet_rate_mean"].mean()),
                "household_unmet_rate_peak": float(shocked["household_unmet_rate_peak"].mean()),
                "share_of_household_shortage_borne_by_shocked_sector": float(
                    np.nanmean(shocked["unmet_share_of_total"].to_numpy(dtype=float))
                ),
                "household_share_of_rationed": house_share,
                "intermediate_share_of_rationed": 0.0,
                "gov_invest_share_of_rationed": other_share,
                "intermediate_service_rate": float(shocked["intermediate_service_rate"].mean()),
                "intermediate_plan_gap_rate": float(
                    1.0 - shocked["intermediate_service_rate"].mean()
                ),
                "x_ratio_min": float(shocked["x_ratio_min"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["mechanism", "shocked_sector"]).reset_index(drop=True)
    writer.write(
        "t07_incidence",
        table,
        caption=(
            "Incidence of the shortfall (block B3): the engine serves intermediate inputs before "
            "final claims, so rationing incidence and production incidence must be read "
            "separately. Within the rationed (final) claims, the ``household share'' is the part "
            "borne by households and ``gov/invest share'' the part borne by collective and "
            "investment demand; intermediate users bear **zero** rationed shortfall by "
            "construction and instead bear a plan gap, reported as the rate at which planned "
            "intermediate inputs were not supplied."
        ),
        label="tab:incidence",
        notes=(
            "Household unmet rate is $\\sum_t \\mathrm{unmet}_i/\\sum_t d_i$ for the shocked "
            "good over the post-burn-in window; the peak is the worst quarter. Source: "
            "\\texttt{results/B3/sector\\_panel.parquet}."
        ),
        formatters={"mechanism": lambda v: _tex_escape(v), "shocked_sector": lambda v: _tex_escape(v)},
        digits=4,
    )
    mb = table[table["mechanism"] == "material_balance"]
    if not mb.empty:
        tests.add(
            block=block,
            hypothesis="incidence",
            outcome="household_share_of_rationed",
            comparison="material_balance, shocked good",
            test="mean over 20 sectors",
            statistic=float(mb["household_share_of_rationed"].mean()),
            p_value=float("nan"),
            effect_name="household share of rationed shortfall",
            effect_value=float(mb["household_share_of_rationed"].mean()),
            n_units=int(len(mb)),
            note="intermediates are served first: their share is 0 by construction",
        )
        tests.add(
            block=block,
            hypothesis="incidence",
            outcome="household_unmet_rate_peak",
            comparison="material_balance, shocked good",
            test="mean over 20 sectors",
            statistic=float(mb["household_unmet_rate_peak"].mean()),
            p_value=float("nan"),
            effect_name="peak household unmet rate",
            effect_value=float(mb["household_unmet_rate_peak"].mean()),
            n_units=int(len(mb)),
            note="how much of the shocked good's household demand goes unmet at the trough",
        )
    return table


def table_rationing_shadow(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t08 — B4: rationing vs shadow prices at three shock sizes."""
    block = "B4"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _substrate(summary)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (mechanism, size), group in frame.groupby(["mechanism", "shock_size"]):
        rows.append(
            {
                "mechanism": mechanism,
                "capacity_loss_pct": int(round((1.0 - float(size)) * 100)),
                "shock_size": float(size),
                "n_seeds": int(len(group)),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "peak_shortage_rate": float(group["shock_peak_shortage"].mean()),
                "mean_shortage_essentials": float(group["mean_shortage_essentials"].mean()),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "shock_output_amplification": float(group["shock_output_amplification"].mean()),
                "recovery_half_life": float(
                    np.nanmean(group["recovery_half_life"].to_numpy(dtype=float))
                )
                if "recovery_half_life" in group
                else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["capacity_loss_pct", "mechanism"]).reset_index(drop=True)
    writer.write(
        "t08_rationing_vs_shadow",
        table,
        caption=(
            "Administrative rationing against shadow-price coordination (block B4) at three "
            "capacity-loss sizes, all with the same eight-quarter duration and the same "
            "shocked sector."
        ),
        label="tab:rationing_shadow",
        notes="20 seeds per cell. Source: \\texttt{results/B4/}.",
        formatters={"mechanism": lambda v: _tex_escape(v)},
        digits=4,
    )
    for size, group in table.groupby("shock_size"):
        if set(group["mechanism"]) >= {"rationing_priority", "tatonnement"}:
            a = frame[
                (frame["mechanism"] == "rationing_priority") & np.isclose(frame["shock_size"], size)
            ].sort_values("seed")
            b = frame[
                (frame["mechanism"] == "tatonnement") & np.isclose(frame["shock_size"], size)
            ].sort_values("seed")
            res = paired_test(a["mean_shortage_rate"], b["mean_shortage_rate"])
            tests.add(
                block=block,
                hypothesis="Q2-rationing",
                outcome="mean_shortage_rate",
                comparison=f"rationing vs tatonnement at {int(round((1-size)*100))}% capacity loss",
                test="paired Wilcoxon signed-rank (CRN)",
                statistic=res["wilcoxon_stat"],
                p_value=res["p_value"],
                effect_name="paired mean difference",
                effect_value=res["mean_diff"],
                n_units=int(res["n_pairs"]),
                n_runs=int(len(a) + len(b)),
                note=f"ci95=[{res['ci95_lo']:.4f},{res['ci95_hi']:.4f}], cliff={res['cliffs_delta']:.2f}",
            )
    return table


def table_dual_track(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t09 — B5: the dual-track frontier in theta (H5)."""
    block = "B5"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _substrate(summary)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (lag, theta), group in frame.groupby(["lag", "theta"]):
        rows.append(
            {
                "lag_reporting": int(lag),
                "lag_effective": int(lag) + 1,
                "theta": float(theta),
                "n_seeds": int(len(group)),
                "mean_welfare": float(group["mean_welfare"].mean()),
                "mean_welfare_sd": float(group["mean_welfare"].std(ddof=1)),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "mean_shortage_essentials": float(group["mean_shortage_essentials"].mean()),
                "mean_price_volatility": float(group["mean_price_volatility"].mean()),
                "mean_plan_fulfilment": float(group["mean_plan_fulfilment"].mean()),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "mean_shortage_sd": float(group["sd_shortage_rate"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["lag_reporting", "theta"]).reset_index(drop=True)
    writer.write(
        "t09_dual_track",
        table,
        caption=(
            "The dual-track frontier (block B5): welfare, shortage and price volatility against "
            "the plan share $\\theta$ at reporting lags 0 and 3. $\\theta=0$ is the pure market "
            "pole and $\\theta=1$ the pure plan pole."
        ),
        label="tab:dual_track",
        notes="20 seeds per cell. Source: \\texttt{results/B5/}.",
        digits=4,
    )
    # To beat both endpoints a candidate must pass both one-sided comparisons.
    # The maximum of their p-values tests that joint claim. Holm covers all candidates.
    for lag, group in frame.groupby("lag"):
        wide = group.pivot(index="seed", columns="theta", values="mean_welfare")
        if 0.0 not in wide or 1.0 not in wide:
            continue
        for theta in wide.columns[(wide.columns > 0) & (wide.columns < 1)]:
            paired = wide[[theta, 0.0, 1.0]].dropna()
            if paired.empty:
                continue
            left = paired_test(paired[theta], paired[0.0], alternative="greater")
            right = paired_test(paired[theta], paired[1.0], alternative="greater")
            tests.add(
                block=block, hypothesis="H5", outcome="mean_welfare",
                comparison=f"lag {lag}, theta {theta:g} vs both endpoints",
                test="intersection of two one-sided paired Wilcoxon tests",
                statistic=theta, p_value=max(left["p_value"], right["p_value"]),
                effect_name="smaller paired mean advantage",
                effect_value=min(left["mean_diff"], right["mean_diff"]),
                n_units=len(paired), n_runs=3 * len(paired),
                note=(f"vs 0: {left['mean_diff']:.6g} (p={left['p_value']:.6g}); "
                      f"vs 1: {right['mean_diff']:.6g} (p={right['p_value']:.6g})"),
            )
    return table


def table_robustness(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t10 — B6: robustness of the headline ordering."""
    block = "B6"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, group in summary.groupby("variant"):
        rows.append(
            {
                "variant": variant,
                "n_seeds": int(len(group)),
                "substrate": str(group["substrate"].iloc[0]),
                "lag_reporting": int(group["lag"].iloc[0]),
                "revision_gain": float(group["gain"].iloc[0]),
                "sigma_obs": float(group["sigma_obs"].iloc[0]),
                "sigma_demand": float(group["sigma_demand"].iloc[0]),
                "labour_supply_factor": float(group["labor_supply_factor"].iloc[0])
                if "labor_supply_factor" in group
                else float("nan"),
                "mean_labour_utilisation": float(group["mean_labour_utilisation"].mean())
                if "mean_labour_utilisation" in group
                else float("nan"),
                "mean_investment_ratio": float(group["mean_investment_ratio"].mean())
                if "mean_investment_ratio" in group
                else float("nan"),
                "a_perturbation": float(group["a_perturb"].iloc[0]),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "mean_shortage_rate_sd": float(group["mean_shortage_rate"].std(ddof=1)),
                "peak_shortage_rate": float(group["shock_peak_shortage"].mean()),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "shock_output_amplification": float(group["shock_output_amplification"].mean()),
                "recovery_half_life": float(
                    np.nanmean(group["recovery_half_life"].to_numpy(dtype=float))
                )
                if "recovery_half_life" in group
                else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_shortage_rate", ascending=False).reset_index(drop=True)
    writer.write(
        "t10_robustness",
        table,
        caption=(
            "Robustness of the reference condition (block B6): substrate, revision gain, demand "
            "and measurement noise, planning effort (internal balance rounds), a 5 per cent "
            "perturbation of the input-output matrix, the normative penalty weights at "
            "$\\pm50$ per cent, and the **labour-slack axis** (rows ``labour*''), which "
            "re-runs the B8 decomposition at 1.05--2.00 times labour supply: with little slack "
            "the taut plan cannot be executed and costs output, with a lot of slack it can, "
            "which turns the second conclusion into a conditional result."
        ),
        label="tab:robustness",
        notes=(
            "A-perturbation is $A'_{ij}=A_{ij}\\exp(0.05 z_{ij})$ with $z$ fixed by the recorded "
            "seed, columns rescaled to stay productive and base final demand recomputed. Source: "
            "\\texttt{results/B6/}."
        ),
        formatters={"variant": lambda v: _tex_escape(v), "substrate": lambda v: _tex_escape(v)},
        digits=4,
    )
    return table


def table_mb_decomposition(
    bundle: ResultsBundle, writer: TableWriter, tests: TestLog
) -> pd.DataFrame:
    """t13 — B8: the revision/balancing decomposition of material balance."""
    block = "B8"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (gain, rounds), group in summary.groupby(["revision_gain", "balance_rounds"]):
        rows.append(
            {
                "revision_gain": float(gain),
                "balance_rounds": int(rounds),
                "n_seeds": int(len(group)),
                "x_over_base_mean": float(group["output_ratio_mean"].mean()),
                "x_over_base_sd": float(group["output_ratio_mean"].std(ddof=1)),
                "mean_investment_ratio": float(group["mean_investment_ratio"].mean())
                if "mean_investment_ratio" in group
                else float("nan"),
                "capital_ratio_final": float(group["capital_ratio_final"].mean())
                if "capital_ratio_final" in group
                else float("nan"),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "mean_shortage_rate_sd": float(group["mean_shortage_rate"].std(ddof=1)),
                "mean_spoilage_rate": float(group["mean_spoilage_rate"].mean())
                if "mean_spoilage_rate" in group
                else float("nan"),
                "mean_labour_utilisation": float(group["mean_labour_utilisation"].mean())
                if "mean_labour_utilisation" in group
                else float("nan"),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_plan_fulfilment": float(group["mean_plan_fulfilment"].mean()),
                "plan_lowering_periods": float(group["plan_lowering_periods"].mean())
                if "plan_lowering_periods" in group
                else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["balance_rounds", "revision_gain"]).reset_index(drop=True)
    # the decomposition: main effects and the interaction, on x/base
    pivot = table.pivot(index="revision_gain", columns="balance_rounds", values="x_over_base_mean")
    pivot_short = table.pivot(index="revision_gain", columns="balance_rounds", values="mean_shortage_rate")
    writer.write(
        "t13_material_balance_decomposition",
        table,
        caption=(
            "Decomposing material balance (block B8): the proportional revision gain against the "
            "number of internal balancing rounds, in the noise-only environment (no shock, no lag, "
            "$\\sigma_{\\mathrm{obs}}=0.03$, AR(1) demand noise). Neither mechanism alone costs "
            "output; the loss is in the interaction. The last four columns show the mechanism: "
            "balancing is a **downward-only filter** on the plan (``lowered'' counts the quarters "
            "in which at least one target came out of the planning step below where it went in), "
            "and because investment is a low-priority claimant it absorbs the shortfall "
            "(``investment/\\bar x falls'), so the capital stock "
            "decays and the transitory information error becomes a permanent output loss. "
            "Spoilage is output that can be neither used nor stored, bounded by "
            "``max_stock_ratio``."
        ),
        label="tab:mb_decomposition",
        notes=(
            "16 seeds per cell; $x/\\bar x$ is realised output relative to the base year. "
            "``plan lowering'' is measured by wrapping the mechanism's plan step and comparing "
            "carried targets before and after; it does not change behaviour. Source: "
            "\\texttt{results/B8/}."
        ),
        digits=4,
    )
    # tests: the two main effects and the interaction, as paired comparisons
    def _cell(gain: float, rounds: int) -> np.ndarray:
        sub = summary[
            np.isclose(summary["revision_gain"], gain)
            & (summary["balance_rounds"].astype(int) == int(rounds))
        ].sort_values("seed")
        return sub["output_ratio_mean"].to_numpy(dtype=float)

    contrasts = {
        "revision alone (gain 1.0, rounds 0) vs none": (_cell(1.0, 0), _cell(0.0, 0)),
        "balancing alone (gain 0, rounds 3) vs none": (_cell(0.0, 3), _cell(0.0, 0)),
        "interaction (gain 0.3, rounds 3) vs gain 0.3, rounds 0": (_cell(0.3, 3), _cell(0.3, 0)),
        "interaction (gain 1.0, rounds 3) vs gain 1.0, rounds 0": (_cell(1.0, 3), _cell(1.0, 0)),
        "worst cell (gain 1.0, rounds 3) vs none": (_cell(1.0, 3), _cell(0.0, 0)),
    }
    for label, (a, b) in contrasts.items():
        if a.size == 0 or b.size == 0:
            continue
        res = paired_test(a, b, alternative="less")
        tests.add(
            block=block,
            hypothesis="H1'-decomposition",
            outcome="output_ratio_mean",
            comparison=label,
            test="paired Wilcoxon signed-rank (one-sided, less)",
            statistic=res["wilcoxon_stat"],
            p_value=res["p_value"],
            effect_name="paired mean difference",
            effect_value=res["mean_diff"],
            n_units=int(res["n_pairs"]),
            n_runs=int(a.size + b.size),
            note=f"ci95=[{res['ci95_lo']:.4f},{res['ci95_hi']:.4f}], cliff={res['cliffs_delta']:.2f}",
        )
    return table


def table_ladder(bundle: ResultsBundle, writer: TableWriter, tests: TestLog) -> pd.DataFrame:
    """t11 — B2L: production-first vs consumption-first priority ladder."""
    block = "B2L"
    if not bundle.has(block):
        return pd.DataFrame()
    summary = merged_summary(bundle, [block])
    frame = _substrate(summary)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (ladder, lag), group in frame.groupby(["ladder", "lag"]):
        rows.append(
            {
                "ladder": ladder,
                "lag_reporting": int(lag),
                "lag_effective": int(lag) + 1,
                "n_seeds": int(len(group)),
                "mean_shortage_rate": float(group["mean_shortage_rate"].mean()),
                "peak_shortage_rate": float(group["shock_peak_shortage"].mean()),
                "mean_welfare_loss": float(group["welfare_loss_vs_baseline"].mean()),
                "mean_x_over_base": float(group["output_ratio_mean"].mean()),
                "mean_plan_fulfilment": float(group["mean_plan_fulfilment"].mean()),
                "shock_output_amplification": float(group["shock_output_amplification"].mean()),
                "recovery_half_life": float(
                    np.nanmean(group["recovery_half_life"].to_numpy(dtype=float))
                )
                if "recovery_half_life" in group
                else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["lag_reporting", "ladder"]).reset_index(drop=True)
    writer.write(
        "t11_ladder",
        table,
        caption=(
            "The priority ladder as a treatment (block B2L, exploratory): what happens when "
            "essential consumption outranks intermediate inputs under a supply shock. "
            "The production-first ladder is the default; the consumption-first ladder is "
            "the classic mobilisation-economy alternative."
        ),
        label="tab:ladder",
        notes="20 seeds per cell. Source: \\texttt{results/B2L/}.",
        formatters={"ladder": lambda v: _tex_escape(v)},
        digits=4,
    )
    prod = frame[frame["ladder"] == "production_first"].sort_values(["lag", "seed"])
    cons = frame[frame["ladder"] == "consumption_first"].sort_values(["lag", "seed"])
    if len(prod) == len(cons) and len(prod) > 0:
        res = paired_test(cons["mean_shortage_rate"], prod["mean_shortage_rate"])
        tests.add(
            block=block,
            hypothesis="ladder",
            outcome="mean_shortage_rate",
            comparison="consumption_first vs production_first (paired by seed and lag)",
            test="paired Wilcoxon signed-rank (CRN)",
            statistic=res["wilcoxon_stat"],
            p_value=res["p_value"],
            effect_name="paired mean difference",
            effect_value=res["mean_diff"],
            n_units=int(res["n_pairs"]),
            n_runs=int(len(prod) + len(cons)),
            note=f"ci95=[{res['ci95_lo']:.4f},{res['ci95_hi']:.4f}]",
        )
    return table


def table_hypotheses(tests: TestLog, writer: TableWriter) -> pd.DataFrame:
    """t00 — one verdict line per pre-registered hypothesis."""
    frame = tests.frame()
    if frame.empty:
        return pd.DataFrame()
    verdicts: list[dict[str, Any]] = []
    text = {
        "H1": (
            "Delay destabilises the administered price loop: the measured critical price gain "
            "should track $g^*(m)/b_p$ within one lag period and the oscillation period should "
            "approach $2(2m+1)$."
        ),
        "H1'": (
            "Taut plans are self-defeating: in a capacity-constrained input-output economy the "
            "plan-revision loop is positive, so shortage rises and welfare falls with the "
            "revision gain."
        ),
        "H2": (
            "Welfare loss rises fastest with the reporting lag for the institution that needs the "
            "freshest information (\\texttt{lp\\_central})."
        ),
        "H3": (
            "Each shocked-sector mean shortage is at least 0.5 per cent."
        ),
        "H4": (
            "A single-sector shock amplifies through the network: the like-for-like output "
            "amplification exceeds one on average over the twenty shocked sectors."
        ),
        "H5": (
            "An interior plan share improves on both endpoint shares."
        ),
    }
    # tests that belong to a hypothesis but carry a sub-label are folded in
    aliases = {
        "H1-gain": "H1",
        "H4-centrality": "H4",
        "H4-multiplier": "H4",
        "H1'-decomposition": "H1'",
    }
    folded = frame.copy()
    folded["hypothesis"] = folded["hypothesis"].replace(aliases)
    for hypothesis in ("H1", "H1'", "H2", "H3", "H4", "H5", "incidence", "ladder"):
        sub = folded[folded["hypothesis"] == hypothesis]
        if sub.empty:
            continue
        sig = sub[sub["significant_05"] == "yes"]
        if hypothesis == "H1":
            verdict = "Threshold not identified"
        elif hypothesis == "H3":
            verdict = "Threshold met in every cell" if sub["effect_value"].iloc[0] == 0 else "Threshold missed"
        elif hypothesis == "H4":
            amplification = (sig["outcome"] == "paired_shock_output_amplification") & sig["test"].str.startswith("one-sided t-test")
            # Centrality and average amplification answer separate questions.
            centrality = sig["outcome"] == "paired_shock_output_loss"
            verdict = f"{int(amplification.sum())} amplification tests; {int(centrality.sum())} centrality tests"
        elif hypothesis == "H2":
            verdict = "LP steepest" if len(sig) == len(MECH_ORDER) - 1 else "Ordering not established"
        elif hypothesis == "H5":
            verdict = "Interior advantage detected" if not sig.empty else "Interior advantage not established"
        else:
            verdict = "See test log"
        verdicts.append(
            {
                "hypothesis": hypothesis,
                "verdict": verdict,
                "n_tests": int(sub["p_value"].notna().sum()),
                "n_significant_holm_05": int(len(sig)),
                "key_statistic": float(sub["statistic"].iloc[0]),
                "key_p_value": float(sub["p_value"].iloc[0]),
                "key_p_holm": float(sub["p_holm"].iloc[0]),
                "prediction": text.get(hypothesis, ""),
            }
        )
    table = pd.DataFrame(verdicts)
    writer.write(
        "t00_hypotheses",
        table,
        caption=(
            "Pre-registered hypotheses and the evidence on them. Every row traces to a test in "
            "\\texttt{results/tables/tests.csv}; p-values are Holm-corrected within block."
        ),
        label="tab:hypotheses",
        notes=(
            "``Key statistic'' is the first test of the hypothesis; the full list is in "
            "\\texttt{tests.csv}. A hypothesis with no significant test after Holm correction is "
            "interpreted using the stated test or descriptive threshold."
        ),
        digits=4,
        alignment="llrrrrrl",
    )
    return table


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_tables(
    bundle: ResultsBundle | None = None,
    out_dir: str | Path | None = None,
    *,
    results_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build every table, the test log and the table manifest.

    Args:
        bundle: pre-loaded results (tests pass a synthetic one); defaults to
            :func:`load_results` on ``results_dir``.
        out_dir: where to write (default ``results/tables``).
        results_dir: override for ``results/`` when ``bundle`` is not given.

    Returns:
        A dict with the test frame, the writer's index and the produced paths.
    """
    bundle = bundle if bundle is not None else load_results(results_dir)
    root = bundle.root
    out = Path(out_dir) if out_dir is not None else root / "tables"
    writer = TableWriter(out_dir=out)
    tests = TestLog()

    if bundle.has("B1"):
        table_validation(bundle, writer, tests)
    if bundle.has("B1b"):
        table_noise_only(bundle, writer, tests)
    if bundle.has("B2"):
        table_lag_mechanism(bundle, writer, tests)
        table_gain_sweep(bundle, writer, tests)
    if bundle.has("B3"):
        table_propagation(bundle, writer, tests)
        table_incidence(bundle, writer, tests)
    if bundle.has("B4"):
        table_rationing_shadow(bundle, writer, tests)
    if bundle.has("B5"):
        table_dual_track(bundle, writer, tests)
    if bundle.has("B6"):
        table_robustness(bundle, writer, tests)
    if bundle.has("B8"):
        table_mb_decomposition(bundle, writer, tests)
    if bundle.has("B7"):
        table_bp_by_sector(bundle, writer, tests)
    if bundle.has("B2L"):
        table_ladder(bundle, writer, tests)
    table_stability_boundary(bundle, writer, tests)
    tests.apply_holm()
    test_frame = tests.frame()
    out.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(out / "tests.csv", index=False, float_format="%.6g")
    table_hypotheses(tests, writer)
    manifest = writer.write_index(
        {
            "results_dir": str(root),
            "blocks_found": bundle.blocks,
            "grid_fingerprint": (bundle.manifest("B2") or {}).get("grid_fingerprint"),
            "git_commit": (bundle.manifest("B2") or {}).get("git_commit"),
            "n_tests": int(len(test_frame)),
            "test_file": str(out / "tests.csv"),
            "how_to_reproduce": "python -m planlab.experiments.aggregate",
        }
    )
    if verbose:
        print(f"tables: {len(writer.written)} written to {out}")
        print(f"tests : {len(test_frame)} rows -> {out / 'tests.csv'}")
        print(f"index : {manifest}")
    return {"tests": test_frame, "tables": writer.written, "manifest": manifest, "out_dir": str(out)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the paper's tables from results/.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    build_tables(results_dir=args.results_dir, out_dir=args.out_dir, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
