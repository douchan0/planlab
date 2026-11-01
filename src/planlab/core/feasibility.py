"""The feasibility programme (the short side of the market).

Given capacity, inventories and the claims that outrank production inputs, the
largest feasible gross-output vector solves

    maximise   sum_j x_j
    subject to A x - x <= r - reserve          (goods balance, per good)
               0 <= x <= ub                    (targets x capacity)
               labour . x <= L                 (labour)

Using a programme rather than a proportional cascade is not a technical
preference.  A cascade that shrinks every sector proportionally cannot
*rebalance* the production mix; in a capacity-constrained input-output economy it
converges to zero rather than to the rebalanced plan.  The programme is the
standard short-side (non-substitution / Dreze-Benassy) allocation.

The LP is only solved when the target vector is actually infeasible, which never
happens in calm periods, so the common case costs one vector comparison.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linprog

__all__ = ["feasible_output", "is_feasible"]


def is_feasible(
    A: np.ndarray,
    x: np.ndarray,
    inventory: np.ndarray,
    reserve: np.ndarray,
    labor: np.ndarray | None = None,
    labour_supply: float = 0.0,
    tol: float = 1e-9,
) -> bool:
    """Cheap check of whether ``x`` satisfies the goods and labour constraints."""
    scale = max(1.0, float(np.max(np.abs(inventory) + np.abs(x))))
    lhs = A @ x - x
    if np.max(lhs - (inventory - reserve)) > tol * scale:
        return False
    if labor is not None and labour_supply > 0:
        if float(np.dot(labor, x)) > labour_supply * (1.0 + 1e-12):
            return False
    return True


def feasible_output(
    A: np.ndarray,
    ub: np.ndarray,
    inventory: np.ndarray,
    reserve: np.ndarray | None = None,
    labor: np.ndarray | None = None,
    labour_supply: float = 0.0,
    *,
    objective: np.ndarray | None = None,
    tol: float = 1e-9,
) -> tuple[np.ndarray, str]:
    """Largest feasible gross output not exceeding ``ub``.

    Parameters
    ----------
    A:
        ``(n, n)`` direct-input coefficient matrix.
    ub:
        ``(n,)`` upper bound on gross output (plan targets capped by capacity).
    inventory:
        ``(n,)`` beginning-of-period stocks ``r``.
    reserve:
        ``(n,)`` goods already committed to claims that outrank production inputs
        (e.g. essential consumption under a "consumption first" ladder).  Clipped
        to ``inventory`` so that the programme is always feasible.
    labor, labour_supply:
        optional labour constraint.
    objective:
        optional ``(n,)`` weights in the maximisation; default is all ones
        (maximise total gross output).

    Returns
    -------
    ``(x, status)`` where status is ``"optimal"``, ``"infeasible-ub"`` (the target
    vector was already feasible, returned unchanged) or ``"fallback"``.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.zeros(0), "empty"
    ub = np.maximum(np.asarray(ub, dtype=float), 0.0)
    inventory = np.maximum(np.asarray(inventory, dtype=float), 0.0)
    if reserve is None:
        reserve = np.zeros(n)
    reserve = np.clip(np.asarray(reserve, dtype=float), 0.0, inventory)

    if is_feasible(A, ub, inventory, reserve, labor, labour_supply, tol):
        return ub.copy(), "infeasible-ub"

    c = -(np.ones(n) if objective is None else np.asarray(objective, dtype=float))
    A_ub = [A - np.eye(n)]
    b_ub = [inventory - reserve]
    if labor is not None and labour_supply and labour_supply > 0:
        A_ub.append(np.asarray(labor, dtype=float).reshape(1, -1))
        b_ub.append(np.array([float(labour_supply)]))
    try:
        res = linprog(
            c,
            A_ub=np.vstack(A_ub),
            b_ub=np.concatenate(b_ub),
            bounds=[(0.0, float(ub[i])) for i in range(n)],
            method="highs",
        )
        if res.success:
            x = np.clip(res.x, 0.0, ub)
            return x, "optimal"
    except Exception:  # pragma: no cover - solver absence / numerical failure
        pass

    x = ub.copy()
    for _ in range(80):
        if is_feasible(A, x, inventory, reserve, labor, labour_supply, tol):
            return x, "fallback"
        x = x * 0.9
    return np.zeros(n), "infeasible"
