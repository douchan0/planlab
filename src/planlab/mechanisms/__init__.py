"""Mechanism registry.

``get_mechanism(name, technology, params)`` is the single entry point used by the
engine and the experiment runner.  Institutions register themselves by setting
``name`` and being imported here.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import CoordinationMechanism
from .hybrid_dual_track import HybridDualTrack
from .lp_central import CentralLP
from .market import Market
from .material_balance import MaterialBalance
from .rationing_priority import RationingPriority
from .tatonnement import Tatonnement

__all__ = [
    "CoordinationMechanism",
    "get_mechanism",
    "available_mechanisms",
    "register",
    "CentralLP",
    "HybridDualTrack",
    "Market",
    "MaterialBalance",
    "RationingPriority",
    "Tatonnement",
]

_REGISTRY: dict[str, type[CoordinationMechanism]] = {}


def register(cls: type[CoordinationMechanism]) -> type[CoordinationMechanism]:
    """Register a mechanism class under its ``name``."""
    if not getattr(cls, "name", None) or cls.name in {"", "base"}:
        raise ValueError(f"{cls!r} must define a unique non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


for _cls in (CentralLP, MaterialBalance, RationingPriority, Tatonnement, Market,
             HybridDualTrack):
    register(_cls)


def available_mechanisms() -> list[str]:
    """Sorted list of registered mechanism ids."""
    return sorted(_REGISTRY)


def get_mechanism(
    name: str,
    technology: Any,
    params: Mapping[str, Any] | None = None,
) -> CoordinationMechanism:
    """Instantiate a mechanism by registry key.

    A *fresh instance* is created on every call: mechanisms carry state (targets,
    prices) across periods, so reusing one across runs would leak information
    between Monte Carlo replicates.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown mechanism {name!r}; available: {', '.join(available_mechanisms())}"
        ) from exc
    return cls(technology, params)
