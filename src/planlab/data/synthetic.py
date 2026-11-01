"""Documented synthetic I-O generator: the offline fallback robustness substrate.

The generator is *not* a model of a real economy and must never be presented as
one.  It produces a table with the **same schema** as the Eurostat aggregate, is
**calibrated to the Eurostat totals**, and has a **different internal
structure** — which is exactly what makes it useful as a substrate for the
robustness block (B6) of the pre-registered design.

Calibration (all numbers are copied from the Eurostat aggregate, none invented)
-----------------------------------------------------------------------------
* gross output                 ``x_i``                — copied verbatim;
* intermediate budget of ``j`` ``c_j = sum_i Z_ij``   — copied verbatim, so the
  sectoral GVA share ``1 - c_j/x_j`` matches the real table;
* total final demand           ``Y = sum_i y_i``      — copied verbatim, so the
  aggregate balance ``A x + y = x`` is exact by construction;
* final-demand composition     ``clip(y_i/x_i, 5%, 90%)``, rescaled so that
  ``sum_i y_i = Y`` — preserves which sectors mostly serve final demand;
* household / government / investment / export split — real proportions.

Structure
---------
The intermediate flow matrix is drawn as

``K[i, j] = exp(1.5*[i == j] + 0.6*[A^eurostat_ij > 0] + N(0, sigma))``

so that the synthetic economy has the *same trading topology* (which sectors
supply which) but randomly drawn magnitudes, then rescaled to the calibrated
margins with iterative proportional fitting (RAS).  RAS preserves both margins
exactly, hence the balance identity holds to machine precision.

Because the column budgets ``c_j`` are the real ones and ``sum_i Z_ij <= c_j``
is enforced by the margin matching, ``sum_i A_ij < 1`` holds by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .aggregate import AggregatedSIOT
from .eurostat import to_coefficients

__all__ = ["SyntheticResult", "generate_synthetic", "DEFAULT_SEED", "DEFAULT_SIGMA"]

DEFAULT_SEED = 20200101
DEFAULT_SIGMA = 0.9

#: Clip bounds for the final-demand / output ratio.
_PHI_LOW = 0.05
_PHI_HIGH = 0.90
#: Diagonal and observed-link bonuses of the random topology.
_DIAGONAL_BONUS = 1.5
_LINK_BONUS = 0.6
_RAS_MAX_ITER = 500
_RAS_TOL = 1e-12


@dataclass(frozen=True)
class SyntheticResult:
    """A generated table plus the calibration record.

    Attributes:
        table: The synthetic table, structurally identical to the Eurostat
            aggregate (same 20 sectors, same value units).
        calibration: JSON-serialisable record of what was taken from the real
            data and what was drawn.
    """

    table: AggregatedSIOT
    calibration: Mapping[str, Any] = field(default_factory=dict)


def generate_synthetic(
    reference: AggregatedSIOT,
    *,
    seed: int = DEFAULT_SEED,
    sigma: float = DEFAULT_SIGMA,
) -> SyntheticResult:
    """Generate a synthetic table calibrated to ``reference`` (the Eurostat aggregate).

    Args:
        reference: Real aggregated table supplying ``x``, the sectoral
            intermediate budgets, the total final demand and its composition.
        seed: RNG seed; identical seed and inputs give identical arrays.
        sigma: Log-noise standard deviation of the random topology.

    Returns:
        :class:`SyntheticResult` whose ``table`` has the same shape and units as
        ``reference``.

    Raises:
        ValueError: if the calibrated margins are not consistent enough to draw
            a non-negative table.
    """
    rng = np.random.default_rng(seed)
    x = np.array(reference.x, dtype=float)
    n = x.size
    positive = x > 0
    if not positive.any():
        raise ValueError("reference table has no positive gross output")

    # --- calibrated margins -------------------------------------------------
    column_budget = reference.Z.sum(axis=0)  # c_j: intermediate spending of j
    y_real = np.array(reference.y_total, dtype=float)
    total_final = float(y_real.sum())
    phi = np.divide(y_real, x, out=np.zeros(n), where=positive)
    phi = np.clip(phi, _PHI_LOW, _PHI_HIGH)
    raw_row = np.where(positive, (1.0 - phi) * x, 0.0)
    scale = float(column_budget.sum()) / float(raw_row.sum())
    row_budget = raw_row * scale

    # Final demand that closes the accounting identity: y0 = x - row_budget.
    y = x - row_budget
    if (y < -1e-9).any():
        raise ValueError("calibrated final demand would be negative; reference is inconsistent")
    y = np.clip(y, 0.0, None)
    # Re-balance tiny residuals introduced by the clip.
    row_budget = x - y
    if abs(row_budget.sum() - column_budget.sum()) > 1e-6 * max(1.0, column_budget.sum()):
        # Absorb the difference proportionally into the row targets.
        ratio = column_budget.sum() / row_budget.sum()
        row_budget = row_budget * ratio
        y = x - row_budget
        if (y < -1e-9).any():
            raise ValueError("could not reconcile final demand with the intermediate budget")

    # --- random topology scaled to the margins (RAS) ------------------------
    support = (reference.Z > 0).astype(float) * _LINK_BONUS
    topo = np.exp(
        _DIAGONAL_BONUS * np.eye(n) + support + rng.normal(0.0, sigma, size=(n, n))
    )
    topo = np.maximum(topo, 1e-12)
    Z, ras_iterations = _ras(topo, row_budget, column_budget)

    # --- final demand composition (real proportions) ------------------------
    components = _split_final_demand(reference, y)
    component_sum = sum(components.values())
    if abs(component_sum.sum() - y.sum()) > 1e-6 * max(1.0, y.sum()):
        raise ValueError("final-demand split did not reproduce the total")

    table = AggregatedSIOT(
        sectors=reference.sectors,
        Z=Z,
        x=x,
        y_total=y,
        household=components["household"],
        government=components["government"],
        gfcf=components["gfcf"],
        inventories=components["inventories"],
        exports=components["exports"],
        imports=reference.imports.copy(),
        compensation=reference.compensation.copy(),
        gva=reference.gva.copy(),
        intermediate_consumption=reference.intermediate_consumption.copy(),
        n_source=reference.n,
    )

    calibration: dict[str, Any] = {
        "generator": "planlab.data.synthetic",
        "version": "1",
        "seed": int(seed),
        "sigma": float(sigma),
        "copied_from_reference": {
            "gross_output": "sum %0.1f" % float(x.sum()),
            "intermediate_budget": "sum %0.1f" % float(column_budget.sum()),
            "total_final_demand": "sum %0.1f" % float(total_final),
            "compensation_of_employees": "sum %0.1f" % float(reference.compensation.sum()),
            "gross_value_added": "sum %0.1f" % float(reference.gva.sum()),
        },
        "drawn": {
            "intermediate_flow_matrix": (
                "log-normal topology with diagonal/link bonuses, RAS-scaled to the "
                "calibrated row and column margins"
            ),
            "final_demand_split": "real component proportions applied to the calibrated total",
        },
        "ras": {"max_iter": _RAS_MAX_ITER, "tol": _RAS_TOL, "iterations": int(ras_iterations)},
        "clipping": {"phi_low": _PHI_LOW, "phi_high": _PHI_HIGH},
        "rounding_note": (
            "The published table is rounded to 0.01 MIO EUR, so x - Z*1 and the "
            "reported y total differ by ~2 MIO EUR.  The synthetic table preserves "
            "the column budgets (which fix the sectoral GVA shares) exactly and lets "
            "that sub-ppm rounding pass into the final-demand total."
        ),
        "final_demand_total_abs_diff_vs_reference": float(
            abs(float(y.sum()) - total_final)
        ),
        "achieved_totals": {
            "gross_output": float(x.sum()),
            "total_intermediate_uses": float(Z.sum()),
            "total_final_demand": float(y.sum()),
            "household_consumption": float(components["household"].sum()),
        },
        "identity_residual_max_abs": float(
            np.max(np.abs(to_coefficients(Z, x) @ x + y - x))
        ),
    }
    return SyntheticResult(table=table, calibration=calibration)


# --------------------------------------------------------------------- helpers
def _ras(
    K: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray
) -> tuple[np.ndarray, int]:
    """Iterative proportional fitting of ``K`` onto the given margins.

    Args:
        K: Positive seed matrix.
        row_targets: Desired row sums.
        col_targets: Desired column sums (``sum(row_targets) == sum(col_targets)``).

    Returns:
        ``(matrix, iterations)`` — the rescaled non-negative matrix and the number
        of sweeps used.

    Raises:
        ValueError: if the margins are inconsistent or RAS fails to converge.
    """
    row_targets = np.asarray(row_targets, dtype=float)
    col_targets = np.asarray(col_targets, dtype=float)
    total_row = row_targets.sum()
    total_col = col_targets.sum()
    if abs(total_row - total_col) > 1e-6 * max(1.0, total_row):
        raise ValueError(
            f"RAS margins are not consistent: sum(rows)={total_row:.6f} "
            f"!= sum(cols)={total_col:.6f}"
        )
    X = np.array(K, dtype=float)
    tolerance = _RAS_TOL * max(1.0, float(total_row))
    residual = np.inf
    iterations = 0
    for iterations in range(1, _RAS_MAX_ITER + 1):
        row_sums = X.sum(axis=1)
        X *= np.divide(row_targets, row_sums, out=np.ones_like(row_targets), where=row_sums > 0)[:, None]
        col_sums = X.sum(axis=0)
        X *= np.divide(col_targets, col_sums, out=np.ones_like(col_targets), where=col_sums > 0)[None, :]
        residual = max(
            float(np.max(np.abs(X.sum(axis=1) - row_targets))) if row_targets.size else 0.0,
            float(np.max(np.abs(X.sum(axis=0) - col_targets))) if col_targets.size else 0.0,
        )
        if residual <= tolerance:
            return X, iterations
    if residual > 1e-6 * max(1.0, float(total_row)):  # pragma: no cover - defensive
        raise ValueError(f"RAS did not converge (residual {residual:.3e})")
    return X, iterations


def _split_final_demand(reference: AggregatedSIOT, y: np.ndarray) -> dict[str, np.ndarray]:
    """Split ``y`` into its five components.

    Per-sector proportions come from ``reference``; the household total is then
    rescaled to the reference household total so that both "total final demand"
    and "household final consumption" match the real aggregate exactly, which is
    what the calibration claim says.  The remaining components absorb the
    difference so that ``household + government + gfcf + inventories + exports
    == y`` still holds per sector.

    Args:
        reference: Real aggregated table (source of the proportions and totals).
        y: Synthesised total final use per sector.

    Returns:
        ``{component: array}`` summing to ``y``.
    """
    n = y.size
    total_final = float(reference.y_total.sum())
    household_target = float(reference.household.sum())
    ref_total = np.array(reference.y_total, dtype=float)
    safe = np.where(ref_total != 0, ref_total, 1.0)
    shares = {
        "household": np.divide(reference.household, safe, out=np.zeros(n), where=ref_total != 0),
        "government": np.divide(reference.government, safe, out=np.zeros(n), where=ref_total != 0),
        "gfcf": np.divide(reference.gfcf, safe, out=np.zeros(n), where=ref_total != 0),
        "inventories": np.divide(reference.inventories, safe, out=np.zeros(n), where=ref_total != 0),
        "exports": np.divide(reference.exports, safe, out=np.zeros(n), where=ref_total != 0),
    }
    # Fall back to the economy-wide proportions where the reference had no final demand.
    fallback = {
        name: (float(np.sum(getattr(reference, name))) / total_final) if total_final else 0.0
        for name in shares
    }
    for name, share in shares.items():
        share[ref_total == 0] = fallback[name]
    # Normalise so the components sum to one for every sector.
    stack = np.vstack([shares[name] for name in shares])
    denom = stack.sum(axis=0)
    stack = np.divide(stack, denom, out=np.full_like(stack, 0.2), where=denom != 0)
    names = list(shares)
    components = {name: stack[i] * y for i, name in enumerate(names)}

    # Rescale the household component onto the real household total, letting the
    # other four absorb the difference so the per-sector sum stays exact.
    raw_household = float(components["household"].sum())
    if household_target > 0 and raw_household > 0:
        lam = household_target / raw_household
        household = components["household"] * lam
        if (household > y + 1e-9).any():
            raise ValueError(
                "cannot match the reference household total without exceeding total final use"
            )
        rest_y = y - household
        rest_names = [name for name in names if name != "household"]
        rest_share = np.vstack([components[name] for name in rest_names])
        total_rest = rest_share.sum(axis=0)
        rest_share = np.divide(
            rest_share, total_rest, out=np.zeros_like(rest_share), where=total_rest > 0
        )
        components["household"] = household
        for i, name in enumerate(rest_names):
            components[name] = rest_share[i] * rest_y
    return components
