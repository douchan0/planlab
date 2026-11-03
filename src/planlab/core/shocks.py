"""Exogenous shock processes.

A scenario declares a list of shocks; :class:`ShockPath` evaluates them for a given
period and returns multiplicative factors applied to capacity and to household
demand.  Everything is deterministic given the shock list, so a mechanism
comparison with identical shocks is *paired* by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["Shock", "ShockPath", "parse_shocks"]

_ALLOWED = {"capacity", "demand", "labour", "price"}


@dataclass(frozen=True)
class Shock:
    """One impulse.

    ``size`` is interpreted per type:

    * ``capacity``: multiplier during the shock (``0.7`` = lose 30% of capacity)
    * ``demand``: multiplier on the affected household basket (``1.2`` = +20%)
    * ``labour``: multiplier on labour supply
    * ``price``: additive log-price shock (``0.1`` = +10% administered price)

    ``duration <= 0`` means permanent for the remaining horizon.  When
    ``ramp > 0`` the shock fades linearly back to 1 over ``ramp`` periods after
    ``duration``.
    """

    type: str
    start: int
    duration: int
    size: float
    sector: int | str | None = None
    ramp: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.type not in _ALLOWED:
            raise ValueError(f"unknown shock type {self.type!r}; allowed: {sorted(_ALLOWED)}")
        if self.start < 0:
            raise ValueError("shock.start must be >= 0")
        if self.type == "labour" and self.size < 0:
            raise ValueError("labour shock.size must be >= 0")
        if self.type in {"capacity", "demand"} and self.size <= 0:
            raise ValueError("shock.size must be > 0 for capacity and demand shocks")

    def factor(self, t: int) -> float:
        """Multiplicative factor at period ``t`` (1.0 outside the shock window)."""
        if t < self.start:
            return 1.0
        end = self.start + self.duration if self.duration > 0 else 10**9
        if t < end:
            return float(self.size)
        if self.ramp > 0 and t < end + self.ramp:
            frac = 1.0 - (t - end + 1) / (self.ramp + 1.0)
            return float(1.0 + (self.size - 1.0) * frac)
        return 1.0

    def active(self, t: int) -> bool:
        return self.factor(t) != 1.0


def _sector_index(sector: int | str | None, n: int) -> np.ndarray:
    if sector is None or (isinstance(sector, str) and sector.lower() in {"all", "*"}):
        return np.arange(n)
    if isinstance(sector, (int, np.integer)):
        idx = int(sector)
        if not 0 <= idx < n:
            raise ValueError(f"sector index {idx} out of range [0,{n})")
        return np.array([idx])
    raise ValueError(f"unresolved sector specifier {sector!r}: resolve names before building")


class ShockPath:
    """Evaluates a list of shocks for a given technology."""

    def __init__(self, shocks: Sequence[Shock], sectors: Sequence[str]):
        self.sectors = tuple(sectors)
        self.n = len(self.sectors)
        self.shocks = tuple(_resolve_sector(s, self.sectors) for s in shocks)

    # ------------------------------------------------------------------ queries
    def capacity_factor(self, t: int) -> np.ndarray:
        out = np.ones(self.n)
        for s in self.shocks:
            if s.type == "capacity":
                out[_idx(s.sector, self.n)] *= s.factor(t)
        return out

    def demand_factor(self, t: int) -> np.ndarray:
        out = np.ones(self.n)
        for s in self.shocks:
            if s.type == "demand":
                out[_idx(s.sector, self.n)] *= s.factor(t)
        return out

    def labour_factor(self, t: int) -> float:
        f = 1.0
        for s in self.shocks:
            if s.type == "labour":
                f *= s.factor(t)
        return f

    def price_shock(self, t: int) -> np.ndarray:
        out = np.zeros(self.n)
        for s in self.shocks:
            if s.type == "price":
                out[_idx(s.sector, self.n)] += (s.size if s.active(t) else 0.0)
        return out

    def describe(self) -> list[dict]:
        return [
            {
                "type": s.type,
                "sector": None if s.sector is None else self.sectors[_idx(s.sector, self.n)[0]],
                "start": s.start,
                "duration": s.duration,
                "size": s.size,
                "ramp": s.ramp,
            }
            for s in self.shocks
        ]

    def peak_capacity_loss(self, base_output: Sequence[float] | None = None) -> float:
        """Capacity destroyed at the trough, as a share of base-year aggregate output.

        With ``base_output`` supplied the shock is weighted by the sector's weight in
        the economy, which is the economically meaningful denominator: losing half of
        a sector that is 2% of output is a 1% shock, not a 50% one.  Calling this
        function without weights returns the unweighted worst percentage cut and is
        only appropriate for diagnostics.

        This is the denominator of the pre-registered amplification statistic,
        ``amplification = peak shortage share / peak capacity-loss share``; values
        above one mean the network amplified the direct impulse.
        """
        cap = [s for s in self.shocks if s.type == "capacity"]
        if not cap:
            return 0.0
        if base_output is not None:
            weights = np.asarray(base_output, dtype=float).reshape(-1)
            total = float(weights.sum())
            if total <= 0:
                return 0.0
            loss = np.zeros(self.n)
            for s in cap:
                idx = _idx(s.sector, self.n)
                loss[idx] = np.maximum(loss[idx], 1.0 - min(s.size, 1.0))
            return float(np.dot(loss, weights) / total)
        return float(max((1.0 - min(s.size, 1.0)) for s in cap))


def _idx(sector: int | str | None, n: int) -> np.ndarray:
    return _sector_index(sector, n)


def _resolve_sector(shock: Shock, sectors: Sequence[str]) -> Shock:
    if shock.type == "capacity" or shock.type == "demand" or shock.type == "price":
        sector = shock.sector
        if isinstance(sector, str) and sector.lower() not in {"all", "*"}:
            if sector not in sectors:
                raise ValueError(f"shock refers to unknown sector {sector!r}")
            return Shock(
                type=shock.type,
                start=shock.start,
                duration=shock.duration,
                size=shock.size,
                sector=sectors.index(sector),
                ramp=shock.ramp,
                meta=dict(shock.meta or {}),
            )
    return shock


def parse_shocks(raw: Sequence[Mapping[str, Any]] | None) -> tuple[Shock, ...]:
    """Build :class:`Shock` objects from a scenario's YAML list."""
    if not raw:
        return ()
    out = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError(f"shock entries must be mappings, got {type(item)!r}")
        out.append(
            Shock(
                type=str(item["type"]),
                start=int(item.get("start", 0)),
                duration=int(item.get("duration", 1)),
                size=float(item["size"]),
                sector=item.get("sector"),
                ramp=int(item.get("ramp", 0)),
                meta={k: v for k, v in item.items() if k not in
                      {"type", "start", "duration", "size", "sector", "ramp"}},
            )
        )
    return tuple(out)
