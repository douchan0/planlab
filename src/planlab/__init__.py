"""PlanLab — a mechanism-switchable laboratory for planned-economy coordination.

Public entry points::

    from planlab import load_technology, get_mechanism, run_scenario

The package is deliberately layered: ``core`` holds the physics and accounting,
``mechanisms`` holds the institutions, ``experiments`` holds the experimental
design, ``data`` holds the reproducible data layer.

Licence: MIT (see LICENSE).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Technology",
    "State",
    "Observation",
    "Plan",
    "Allocation",
    "PeriodRecord",
    "Scenario",
    "get_mechanism",
    "run_scenario",
    "load_technology",
]


def __getattr__(name: str):  # lazy imports keep `import planlab` cheap
    if name in {"Technology", "State", "Observation", "Plan", "Allocation", "PeriodRecord"}:
        from .core import state as _state

        return getattr(_state, name)
    if name == "Scenario":
        from .core.scenarios import Scenario

        return Scenario
    if name == "get_mechanism":
        from .mechanisms import get_mechanism

        return get_mechanism
    if name == "run_scenario":
        from .core.engine import run_scenario

        return run_scenario
    if name == "load_technology":
        from .data import load_technology

        return load_technology
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
