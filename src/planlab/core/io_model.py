"""Construction and validation of :class:`~planlab.core.state.Technology`.

The data layer (:mod:`planlab.data`) is responsible for *producing* the arrays from
a published input–output table; this module is responsible for turning raw arrays
into a validated :class:`Technology` and for checking the properties the engine
relies on (productivity, non-negativity, positive base output).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .state import Technology

__all__ = [
    "build_technology",
    "toy_technology",
    "normalise_sector_names",
    "ensure_viable",
]


def normalise_sector_names(sectors: Sequence[str]) -> tuple[str, ...]:
    """Coerce to a tuple of unique ASCII-ish identifiers."""
    names = tuple(str(s).strip() for s in sectors)
    if len(set(names)) != len(names):
        raise ValueError("sector names must be unique")
    if any(not nm for nm in names):
        raise ValueError("sector names must be non-empty")
    return names


def ensure_viable(A: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale ``A`` down until every sector's final demand residual is non-negative.

    A published I-O table can show a negative residual ``x_i - (A x)_i`` for sectors
    whose supply is topped up by imports.  Our closed-economy model has no imports,
    so such a table is made viable by a *uniform* scaling of the coefficient matrix,
    ``A -> theta * A`` with ``theta <= 1``.  The interpretation is that the scaling
    absorbs the import content of intermediate use into value added; it preserves
    the identity ``x = A x + y`` exactly and keeps ``y >= 0``.  ``theta`` is
    reported so the adjustment is auditable.

    Returns
    -------
    (A_scaled, theta)
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    Ax = A @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Ax > 0, x / np.maximum(Ax, 1e-300), np.inf)
    theta = float(min(1.0, float(np.min(ratio))))
    if theta < 1.0:
        theta = max(theta * (1.0 - 1e-9), 0.0)
        A = A * theta
    return A, theta


def build_technology(
    *,
    sectors: Sequence[str],
    A: Any,
    labor: Any,
    kappa: Any,
    base_output: Any,
    base_final: Any,
    consumption_weights: Any | None = None,
    essentials: Any | None = None,
    meta: Mapping[str, Any] | None = None,
    adjust_viability: bool = True,
) -> Technology:
    """Assemble a validated :class:`Technology`.

    Parameters
    ----------
    A:
        ``(n, n)`` direct-input coefficient matrix, ``A[i, j]`` = units of good
        ``i`` needed per unit of gross output of sector ``j``.
    labor:
        ``(n,)`` labour input per unit of gross output.
    kappa:
        ``(n,)`` capital-output ratio (``capacity = capital / kappa``).
    base_output, base_final:
        ``(n,)`` base-year gross output and **total final demand**.  PlanLab is a
        closed-economy model, so final demand must be the residual
        ``x - A x``; the base year is then an exact fixed point of the model
        (``A x + y = x``).  If ``base_final`` is omitted it is computed from the
        identity.  If it is supplied and does not satisfy the identity, a
        :class:`ValueError` is raised: a silently inconsistent technology produces
        fake bottlenecks, which is the single most dangerous bug in this model.
    consumption_weights, essentials:
        optional ``(n,)``; defaults are ``1.0`` and all-non-essential.

    Defaults for ``consumption_weights`` and ``essentials`` are intentionally
    neutral: any normative weighting must come from a documented scenario.
    """
    sectors = normalise_sector_names(sectors)
    n = len(sectors)
    base_output = np.asarray(base_output, dtype=float).reshape(-1)
    if base_output.shape != (n,):
        raise ValueError(f"base_output must be ({n},)")
    A_arr = np.asarray(A, dtype=float)
    if adjust_viability:
        A_arr, theta = ensure_viable(A_arr, base_output)
    else:
        theta = 1.0
    residual = base_output - A_arr @ base_output
    scale = max(1.0, float(np.max(np.abs(base_output))))
    if base_final is None:
        base_final_arr = np.maximum(residual, 0.0)
    else:
        base_final_arr = np.asarray(base_final, dtype=float).reshape(-1)
        err = float(np.max(np.abs(base_final_arr - residual))) / scale
        if err > 1e-6:
            raise ValueError(
                "base_final is inconsistent with A @ base_output: the model requires "
                f"base_final == base_output - A @ base_output (relative error {err:.2e}). "
                "Pass base_final=None to compute it, or aggregate the I-O table so that "
                "final demand is the residual of gross output over intermediate use."
            )
    meta_dict = dict(meta or {})
    meta_dict.setdefault("viability_theta", theta)
    meta_dict.setdefault("base_final_definition", "total final demand (x - A x)")
    w = (
        np.ones(n)
        if consumption_weights is None
        else np.asarray(consumption_weights, dtype=float).reshape(-1)
    )
    ess = (
        np.zeros(n)
        if essentials is None
        else np.asarray(essentials, dtype=float).reshape(-1)
    )
    return Technology(
        sectors=sectors,
        A=A_arr,
        labor=np.asarray(labor, dtype=float),
        kappa=np.asarray(kappa, dtype=float),
        consumption_weights=w,
        essentials=ess,
        base_output=base_output,
        base_final=base_final_arr,
        meta=meta_dict,
    )


def toy_technology(n: int = 6, seed: int = 12345) -> Technology:
    """A small random-but-productive technology, **for unit tests only**.

    The paper's results never use this: the real substrates come from
    :mod:`planlab.data`.  The generator is seeded so that tests are deterministic.
    """
    rng = np.random.default_rng(seed)
    # a sparse, productive random matrix: scale rows so that column sums are < 0.6
    raw = rng.random((n, n)) * (rng.random((n, n)) < 0.5)
    np.fill_diagonal(raw, rng.random(n) * 0.5 + 0.1)
    colsum = raw.sum(axis=0)
    A = raw * (0.55 / np.maximum(colsum, 1e-9))
    base_output = rng.uniform(80.0, 160.0, size=n)
    A, _theta = ensure_viable(A, base_output)
    base_final = base_output - A @ base_output
    labor = (1.0 - A.sum(axis=0)) * 0.6
    kappa = rng.uniform(2.0, 4.0, size=n)
    essentials = np.zeros(n)
    essentials[[0, 1]] = 1.0
    weights = np.ones(n)
    weights[essentials > 0] = 1.2
    return build_technology(
        sectors=[f"s{i + 1:02d}" for i in range(n)],
        A=A,
        labor=labor,
        kappa=kappa,
        base_output=base_output,
        base_final=base_final,
        consumption_weights=weights,
        essentials=essentials,
        meta={"source": "toy_technology", "seed": seed, "warning": "unit-test fixture"},
    )
