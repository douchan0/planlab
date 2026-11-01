"""Private helpers shared by the mechanism implementations.

Nothing here is part of the frozen interface of ``docs/ARCHITECTURE.md``: these are
validation and masking utilities that keep the four institution modules short and
make every parameter's valid range explicit and enforced (the experiment layer
passes parameters straight from scenario YAML, so an out-of-range value must fail
loudly at construction time rather than produce silently meaningless numbers).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..core.state import Technology

__all__ = ["require_range", "require_int", "sector_mask"]


def require_range(
    name: str,
    value: Any,
    lo: float,
    hi: float,
    *,
    lo_open: bool = False,
    hi_open: bool = False,
) -> float:
    """Validate that ``value`` lies in ``[lo, hi]`` (or the open variant) and return it.

    Raises
    ------
    ValueError
        If the value is not finite or outside the documented range.
    """
    v = float(value)
    left = "(" if lo_open else "["
    right = ")" if hi_open else "]"
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    ok = (v > lo if lo_open else v >= lo) and (v < hi if hi_open else v <= hi)
    if not ok:
        raise ValueError(f"{name} must lie in {left}{lo}, {hi}{right}, got {v}")
    return v


def require_int(name: str, value: Any, lo: int, hi: int) -> int:
    """Validate an integer parameter in ``[lo, hi]``."""
    if isinstance(value, bool) or not float(value).is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if not lo <= v <= hi:
        raise ValueError(f"{name} must lie in [{lo}, {hi}], got {v}")
    return v


def sector_mask(tech: Technology, spec: Any) -> np.ndarray:
    """Boolean ``(n,)`` mask selecting "priority" sectors.

    ``spec`` accepts

    ``"essentials"`` / ``"essential"``
        the sectors flagged essential in the substrate;
    ``"all"`` / ``"*"``
        every sector;
    ``"none"`` / ``None``
        no sector;
    a sector name
        that one sector;
    an iterable of names and/or integer indices.
    """
    n = tech.n
    sectors = list(tech.sectors)
    lower = [s.lower() for s in sectors]

    def _one(name: str) -> int:
        try:
            return lower.index(str(name).strip().lower())
        except ValueError:
            raise ValueError(f"unknown sector {name!r}; known: {sectors}") from None

    if spec is None:
        return np.zeros(n, dtype=bool)
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"all", "*"}:
            return np.ones(n, dtype=bool)
        if key in {"none", ""}:
            return np.zeros(n, dtype=bool)
        if key in {"essentials", "essential"}:
            return np.asarray(tech.essentials, dtype=float) > 0.5
        mask = np.zeros(n, dtype=bool)
        mask[_one(key)] = True
        return mask
    if isinstance(spec, (bytes, bytearray)) or not isinstance(spec, Iterable):
        raise ValueError(f"priority_sectors must be a string or an iterable, got {spec!r}")
    mask = np.zeros(n, dtype=bool)
    for item in spec:  # type: ignore[union-attr]
        if isinstance(item, str):
            mask[_one(item)] = True
        else:
            idx = int(item)
            if not 0 <= idx < n:
                raise ValueError(f"priority sector index {idx} out of range [0, {n})")
            mask[idx] = True
    return mask
