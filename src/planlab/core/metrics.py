"""Outcome metrics.

Design rules (see ``docs/PROTOCOL.md`` §5):

* every metric is a scalar in a flat ``dict[str, float]`` so that period records
  concatenate into a dataframe without schema gymnastics;
* aggregate output is never reported on its own — it is always accompanied by
  distribution and shortage measures;
* nothing here is a "true" social welfare function.  :func:`welfare` is a
  cardinalisation with a documented and perturbable shape.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .state import PeriodRecord, Technology

__all__ = [
    "welfare",
    "subsistence_vector",
    "period_metrics",
    "detrended_deviation",
    "autocorrelation",
    "run_metrics",
    "penalised_welfare",
]

_CHUNK = 1e-3  # subsistence floor as a share of base final consumption


def subsistence_vector(tech: Technology) -> np.ndarray:
    """``c0`` in ``sum_i w_i log(C_i + c0_i)``; keeps the log finite at zero."""
    base = np.maximum(tech.base_final, 1e-6)
    return _CHUNK * base


def welfare(tech: Technology, C: np.ndarray) -> float:
    """``sum_i w_i log(C_i + c0_i)`` — the cardinalisation used throughout.

    Weights ``w_i`` are normative scenario parameters, not estimates.
    """
    c0 = subsistence_vector(tech)
    return float(np.dot(tech.consumption_weights, np.log(np.maximum(C, 0.0) + c0)))


def period_metrics(
    *,
    tech: Technology,
    x: np.ndarray,
    Z: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
    invest: np.ndarray,
    inventory_end: np.ndarray,
    demand: np.ndarray,
    unmet: np.ndarray,
    price: np.ndarray,
    capacity: np.ndarray,
    plan_targets: np.ndarray,
    iterations: int,
    prev_x: np.ndarray | None = None,
    prev_price: np.ndarray | None = None,
) -> dict[str, float]:
    """All per-period metrics, as a flat dictionary."""
    eps = 1e-12
    dem_tot = float(demand.sum())
    shortage_rate = float(unmet.sum()) / dem_tot if dem_tot > eps else 0.0
    essentials = tech.essentials > 0.5
    dem_ess = float(demand[essentials].sum())
    ess_shortage = (
        float(unmet[essentials].sum()) / dem_ess if dem_ess > eps else 0.0
    )
    plan_ratio = np.where(plan_targets > eps, x / np.maximum(plan_targets, eps), 1.0)
    plan_dev = float(np.mean(np.abs(plan_ratio - 1.0)))
    plan_ful = float(np.mean(np.clip(plan_ratio, 0.0, 2.0)))
    cap = np.maximum(capacity, eps)
    util = float(np.mean(np.clip(x / cap, 0.0, 1.0)))
    inv_ratio = float(np.mean(inventory_end / np.maximum(demand, eps)))
    lprice = np.log(np.maximum(price, eps))
    out = {
        "demand_total": dem_tot,
        "consumption_total": float(C.sum()),
        "output_total": float(x.sum()),
        "investment_total": float(invest.sum()),
        "gov_total": float(G.sum()),
        "shortage_rate": shortage_rate,
        "shortage_rate_essentials": ess_shortage,
        "unmet_total": float(unmet.sum()),
        "welfare": welfare(tech, C),
        "plan_fulfilment": plan_ful,
        "plan_deviation": plan_dev,
        "capital_utilisation": util,
        "inventory_ratio": inv_ratio,
        "price_dispersion": float(np.std(lprice)),
        "price_level": float(np.mean(lprice)),
        "iterations": float(iterations),
        "consumption_dispersion": float(
            np.std(np.where(demand > eps, C / np.maximum(demand, eps), 1.0))
        ),
    }
    if prev_price is not None:
        out["price_volatility"] = float(np.std(lprice - np.log(np.maximum(prev_price, eps))))
    if prev_x is not None:
        denom = np.maximum(prev_x, eps)
        out["adjustment_cost"] = float(np.mean(np.abs(x - prev_x) / denom))
        out["output_volatility_period"] = float(np.std(np.log(np.maximum(x, eps) / denom)))
    else:
        out["adjustment_cost"] = 0.0
        out["output_volatility_period"] = 0.0
    return out


def detrended_deviation(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Deviation of a series from its centred moving average (a cheap HP substitute).

    Used for the pre-registered oscillation measures; documented as such because it
    differs slightly from an HP filter at the endpoints.
    """
    s = np.asarray(series, dtype=float).reshape(-1)
    if s.size == 0:
        return s.copy()
    w = int(min(window, max(3, s.size)))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return s - s.mean()
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(s, pad, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    return s - smooth[: s.size]


def autocorrelation(series: np.ndarray, max_lag: int = 4) -> np.ndarray:
    """Sample autocorrelation for lags ``1..max_lag`` (zeros where undefined)."""
    s = np.asarray(series, dtype=float).reshape(-1)
    out = np.zeros(max_lag)
    if s.size < 3:
        return out
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom <= 1e-300:
        return out
    for lag in range(1, max_lag + 1):
        if lag >= s.size:
            break
        out[lag - 1] = float(np.dot(s[:-lag], s[lag:]) / denom)
    return out


def penalised_welfare(period: Mapping[str, float], weights: Mapping[str, float],
                      baseline_welfare: float | None = None) -> float:
    """``W - l1*shortage - l2*inventory - l3*plan_deviation - l4*dispersion - l5*adjustment``.

    ``baseline_welfare`` (if given) is subtracted first, so the penalty terms are
    expressed against the same origin as the welfare loss used elsewhere.
    """
    w = welfare_or_zero(period)
    base = 0.0 if baseline_welfare is None else float(baseline_welfare)
    return float(
        (w - base)
        - weights.get("shortage", 0.0) * period.get("shortage_rate", 0.0)
        - weights.get("inventory", 0.0) * period.get("inventory_ratio", 0.0)
        - weights.get("plan_deviation", 0.0) * period.get("plan_deviation", 0.0)
        - weights.get("inequality", 0.0) * period.get("consumption_dispersion", 0.0)
        - weights.get("adjustment", 0.0) * period.get("adjustment_cost", 0.0)
    )


def welfare_or_zero(period: Mapping[str, float]) -> float:
    return float(period.get("welfare", 0.0))


def run_metrics(
    records: Sequence[PeriodRecord],
    *,
    scenario,
    tech: Technology,
    shock_intensity: float = 0.0,
) -> dict[str, float]:
    """Run-level aggregates over the post-burn-in window.

    Periods before ``scenario.burn_in`` are discarded so that the initial stock and
    capital position does not contaminate the comparison.
    """
    burn = int(scenario.burn_in)
    used = [r for r in records if r.t > burn]
    if not used:
        used = list(records)
    beta = float(scenario.discount)

    def series(name: str) -> np.ndarray:
        return np.array([float(r.metrics.get(name, 0.0)) for r in used])

    shortage = series("shortage_rate")
    welfare_s = series("welfare")
    disc = np.array([beta ** (r.t - used[0].t) for r in used])
    disc = disc / disc.sum()
    weights = scenario.objective

    out: dict[str, float] = {
        "n_periods": float(len(used)),
        "mean_shortage_rate": float(np.mean(shortage)),
        "max_shortage_rate": float(np.max(shortage)),
        "sd_shortage_rate": float(np.std(shortage)),
        "mean_shortage_essentials": float(np.mean(series("shortage_rate_essentials"))),
        "cum_welfare": float(np.dot(welfare_s, disc) * len(used)),
        "mean_welfare": float(np.mean(welfare_s)),
        "sd_welfare": float(np.std(welfare_s)),
        "mean_plan_fulfilment": float(np.mean(series("plan_fulfilment"))),
        "mean_plan_deviation": float(np.mean(series("plan_deviation"))),
        "mean_inventory_ratio": float(np.mean(series("inventory_ratio"))),
        "mean_capital_utilisation": float(np.mean(series("capital_utilisation"))),
        "mean_price_dispersion": float(np.mean(series("price_dispersion"))),
        "mean_price_volatility": float(np.mean(series("price_volatility"))),
        "mean_iterations": float(np.mean(series("iterations"))),
        "max_iterations": float(np.max(series("iterations"))),
        "mean_consumption_dispersion": float(np.mean(series("consumption_dispersion"))),
        "mean_output": float(np.mean(series("output_total"))),
        "mean_consumption": float(np.mean(series("consumption_total"))),
    }

    # --- oscillation diagnostics on the (aggregate) price level ---------------
    price_level = np.array([float(np.mean(r.price)) for r in used])
    logp = np.log(np.maximum(price_level, 1e-12))
    dev = detrended_deviation(logp)
    out["oscillation_amplitude"] = float(np.std(dev))
    ac = autocorrelation(dev, max_lag=4)
    for i, a in enumerate(ac, start=1):
        out[f"acf_lag{i}"] = float(a)
    out["oscillation_period"] = _dominant_period(dev)

    # sector-level price paths: amplitude averaged over sectors
    price_mat = np.array([r.price for r in used])
    logpm = np.log(np.maximum(price_mat, 1e-12))
    devs = np.array([detrended_deviation(logpm[:, j]) for j in range(logpm.shape[1])])
    out["oscillation_amplitude_sector_mean"] = float(np.mean(np.std(devs, axis=1)))

    # --- stability index -----------------------------------------------------
    out["stability_index"] = float(1.0 / (1.0 + np.std(shortage) / max(np.mean(shortage), 1e-9)))

    # --- welfare penalties (normative composite) ----------------------------
    mean_period = {k: float(np.mean(series(k))) for k in (
        "shortage_rate", "inventory_ratio", "plan_deviation",
        "consumption_dispersion", "adjustment_cost",
    )}
    mean_period["welfare"] = out["mean_welfare"]
    out["penalised_welfare"] = penalised_welfare(mean_period, weights.as_dict())

    # --- shock response ------------------------------------------------------
    first_start = min((s.start for s in scenario.shocks), default=0)
    last_end = max((s.start + s.duration for s in scenario.shocks), default=0)
    shock_periods = [r for r in used if first_start <= r.t <= last_end + 12]
    if scenario.shocks and shock_periods:
        out["shock_peak_shortage"] = float(max(r.metrics["shortage_rate"] for r in shock_periods))
        out["shock_amplification"] = (
            float(out["shock_peak_shortage"] / shock_intensity) if shock_intensity > 0 else 0.0
        )
        # like-for-like amplification: aggregate output loss against the capacity loss
        # that caused it.  The shortage-based ratio above deliberately compares a
        # demand-side share with a supply-side share and is therefore always larger;
        # this one answers "did the network amplify the production impulse?".
        base_total = float(np.sum(tech.base_output))
        out["shock_output_loss"] = float(
            max((base_total - float(np.sum(r.x))) / base_total for r in shock_periods)
        )
        out["shock_output_amplification"] = (
            float(out["shock_output_loss"] / shock_intensity) if shock_intensity > 0 else 0.0
        )
        pre = [r.metrics["shortage_rate"] for r in used if r.t < first_start]
        base_level = (
            float(np.mean(pre)) if pre else float(np.median(shortage[: max(1, len(shortage) // 4)]))
        )
        out["pre_shock_shortage"] = base_level
        out["recovery_half_life"] = _recovery_half_life(
            shortage, base_level, start=first_start, t0=used[0].t
        )
    else:
        out["shock_peak_shortage"] = 0.0
        out["shock_amplification"] = 0.0
        out["pre_shock_shortage"] = 0.0
        out["shock_output_loss"] = 0.0
        out["shock_output_amplification"] = 0.0
        out["recovery_half_life"] = 0.0

    return out


def _dominant_period(dev: np.ndarray) -> float:
    """Period of the strongest spectral peak, in periods (NaN if too short)."""
    if dev.size < 8:
        return float("nan")
    s = dev - dev.mean()
    if np.allclose(s, 0.0):
        return float("nan")
    n = 1 << int(np.ceil(np.log2(s.size)))
    spec = np.abs(np.fft.rfft(s, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if freqs[k] <= 0:
        return float("nan")
    return float(1.0 / freqs[k])


def _recovery_half_life(
    shortage: np.ndarray, baseline: float, start: int, t0: int = 0
) -> float:
    """Periods until the shortage rate has closed half the gap to its pre-shock level.

    ``shortage`` is indexed from period ``t0`` (the first period after burn-in), while
    ``start`` is an *absolute* period number, so the offset is removed before slicing.
    """
    off = int(max(0, start - t0))
    seg = shortage[off:]
    if seg.size == 0:
        return float("nan")
    peak_idx = int(np.argmax(seg))
    peak = float(seg[peak_idx])
    if peak <= baseline:
        return 0.0
    target = baseline + 0.5 * (peak - baseline)
    # search *after* the peak: the shortage is still below half its peak before the
    # shock has worked through the network, which would otherwise return zero
    idx = np.where(seg[peak_idx:] <= target)[0]
    return float(idx[0]) if idx.size else float("nan")
