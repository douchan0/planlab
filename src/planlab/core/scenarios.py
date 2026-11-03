"""Scenario definitions and YAML loading.

A scenario is the complete, self-describing input of one simulation: technology
substrate, mechanism, information regime, shock list, demand process and
normative weights.  No scenario relies on hidden defaults beyond the documented
ones in the dataclass definitions, and every field that enters the welfare
function is recorded so that a run is reproducible from its YAML file alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .shocks import Shock, parse_shocks

__all__ = [
    "InfoRegime",
    "DemandProcess",
    "ObjectiveWeights",
    "HouseholdParams",
    "Scenario",
    "load_scenario",
    "scenario_from_dict",
    "list_scenarios",
    "scenario_hash",
]


@dataclass(frozen=True)
class InfoRegime:
    """How stale and noisy the coordinator's picture of the economy is."""

    lag: int = 0
    noise: float = 0.0
    observe_capacity: bool = True
    observe_inventory: bool = True
    observe_backlog: bool = True
    planning_horizon: int = 1

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError("info.lag must be >= 0")
        if self.noise < 0:
            raise ValueError("info.noise must be >= 0")
        if self.planning_horizon < 1:
            raise ValueError("info.planning_horizon must be >= 1")


@dataclass(frozen=True)
class DemandProcess:
    """Household demand: trend, AR(1) noise and price/income response."""

    growth: float = 0.0
    ar1: float = 0.0
    sigma: float = 0.0
    price_elasticity: float = 0.0
    income_elasticity: float = 1.0

    def __post_init__(self) -> None:
        if not (-0.999 < self.ar1 < 1.0):
            raise ValueError("demand.ar1 must be in (-1, 1)")
        if self.sigma < 0:
            raise ValueError("demand.sigma must be >= 0")
        if self.price_elasticity < 0:
            raise ValueError("demand.price_elasticity must be >= 0")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights of the normative loss terms.  These are *policy choices*, never
    estimates of a "true" social welfare function; they are recorded and perturbed."""

    shortage: float = 1.0
    inventory: float = 0.1
    plan_deviation: float = 0.5
    inequality: float = 0.0
    adjustment: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "shortage": self.shortage,
            "inventory": self.inventory,
            "plan_deviation": self.plan_deviation,
            "inequality": self.inequality,
            "adjustment": self.adjustment,
        }


@dataclass(frozen=True)
class HouseholdParams:
    """Demand aggregation and rationing perception."""

    #: share of household demand that must be met before "essential" claims are satisfied
    essential_priority: float = 0.0
    #: fraction of *unmet* demand carried over as backlog
    backlog_persistence: float = 0.5
    #: anticipatory stockpiling response to observed shortages (Kornai channel)
    hoarding: float = 0.0


@dataclass(frozen=True)
class Scenario:
    """One complete experimental condition."""

    id: str
    mechanism: str
    technology: Mapping[str, Any]
    periods: int = 60
    seeds: tuple[int, ...] = (0,)
    burn_in: int = 20
    shocks: tuple[Shock, ...] = ()
    info: InfoRegime = field(default_factory=InfoRegime)
    demand: DemandProcess = field(default_factory=DemandProcess)
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    household: HouseholdParams = field(default_factory=HouseholdParams)
    mechanism_params: Mapping[str, Any] = field(default_factory=dict)
    name: str = ""
    # --- calibration knobs (documented defaults, overridable per scenario) ---
    stock_ratio: float = 0.15
    #: storage capacity ceiling, as a multiple of base-year output per sector.
    #: Output that cannot be sold and cannot be stored is spoilage (recorded, not
    #: silently accumulated): without a bound the residual-to-stock accounting rule
    #: permits unbounded accumulation of unsalable output, which is a property of the
    #: accounting rule rather than of the economy.
    max_stock_ratio: float = 2.0
    investment_share: float = 0.12
    gov_share: float = 0.10
    depreciation: float = 0.02
    discount: float = 0.98
    labor_supply_factor: float = 1.15
    indexation: float = 0.0
    price_bounds: tuple[float, float] = (0.05, 20.0)
    tags: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario.id is required")
        if self.periods < 1:
            raise ValueError("periods must be >= 1")
        if not 0 <= self.burn_in < self.periods:
            raise ValueError("burn_in must satisfy 0 <= burn_in < periods")
        if self.stock_ratio < 0:
            raise ValueError("stock_ratio must be >= 0")
        if self.max_stock_ratio <= 0:
            raise ValueError("max_stock_ratio must be strictly positive")
        if not 0 <= self.investment_share < 1:
            raise ValueError("investment_share must be in [0, 1)")
        if not 0 <= self.gov_share < 1:
            raise ValueError("gov_share must be in [0, 1)")
        if not 0 <= self.depreciation < 1:
            raise ValueError("depreciation must be in [0, 1)")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        lo, hi = self.price_bounds
        if not (0 < lo < hi):
            raise ValueError("price_bounds must satisfy 0 < lo < hi")
        if not math.isfinite(self.labor_supply_factor) or self.labor_supply_factor < 0:
            raise ValueError("labor_supply_factor must be finite and >= 0")
        if any(shock.type == "price" for shock in self.shocks):
            raise ValueError("price shocks are not supported in scenarios; use an experimental price probe")
        if not self.seeds:
            raise ValueError("at least one seed is required")

    # ------------------------------------------------------------------ helpers
    @property
    def max_lag_reachable(self) -> int:
        return self.burn_in

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable representation (used for hashing and manifests)."""
        return {
            "id": self.id,
            "name": self.name,
            "mechanism": self.mechanism,
            "technology": dict(self.technology),
            "periods": self.periods,
            "seeds": list(self.seeds),
            "burn_in": self.burn_in,
            "shocks": [
                {
                    "type": s.type,
                    "start": s.start,
                    "duration": s.duration,
                    "size": s.size,
                    "sector": s.sector,
                    "ramp": s.ramp,
                }
                for s in self.shocks
            ],
            "info": {
                "lag": self.info.lag,
                "noise": self.info.noise,
                "observe_capacity": self.info.observe_capacity,
                "observe_inventory": self.info.observe_inventory,
                "observe_backlog": self.info.observe_backlog,
                "planning_horizon": self.info.planning_horizon,
            },
            "demand": {
                "growth": self.demand.growth,
                "ar1": self.demand.ar1,
                "sigma": self.demand.sigma,
                "price_elasticity": self.demand.price_elasticity,
                "income_elasticity": self.demand.income_elasticity,
            },
            "objective": self.objective.as_dict(),
            "household": {
                "essential_priority": self.household.essential_priority,
                "backlog_persistence": self.household.backlog_persistence,
                "hoarding": self.household.hoarding,
            },
            "mechanism_params": dict(self.mechanism_params),
            "stock_ratio": self.stock_ratio,
            "max_stock_ratio": self.max_stock_ratio,
            "investment_share": self.investment_share,
            "gov_share": self.gov_share,
            "depreciation": self.depreciation,
            "discount": self.discount,
            "labor_supply_factor": self.labor_supply_factor,
            "indexation": self.indexation,
            "price_bounds": list(self.price_bounds),
            "tags": list(self.tags),
        }

    def replace(self, **changes: Any) -> "Scenario":
        base = self.to_dict()
        base.update(changes)
        return scenario_from_dict(base)


def scenario_from_dict(data: Mapping[str, Any]) -> Scenario:
    """Build a :class:`Scenario` from a (YAML-decoded) mapping."""
    if not isinstance(data, Mapping):
        raise TypeError("scenario must be a mapping")
    known = {
        "id", "name", "mechanism", "technology", "periods", "seeds", "burn_in",
        "shocks", "info", "demand", "objective", "household", "mechanism_params",
        "stock_ratio", "max_stock_ratio", "investment_share", "gov_share", "depreciation", "discount",
        "labor_supply_factor", "indexation", "price_bounds", "tags",
    }
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
    for required in ("id", "mechanism", "technology"):
        if required not in data:
            raise ValueError(f"scenario is missing required key {required!r}")
    info = InfoRegime(**dict(data.get("info") or {}))
    demand = DemandProcess(**dict(data.get("demand") or {}))
    objective = ObjectiveWeights(**dict(data.get("objective") or {}))
    household = HouseholdParams(**dict(data.get("household") or {}))
    pb = data.get("price_bounds", (0.05, 20.0))
    return Scenario(
        id=str(data["id"]),
        name=str(data.get("name", data["id"])),
        mechanism=str(data["mechanism"]),
        technology=dict(data["technology"]),
        periods=int(data.get("periods", 60)),
        seeds=tuple(int(s) for s in data.get("seeds", (0,))),
        burn_in=int(data.get("burn_in", 20)),
        shocks=parse_shocks(data.get("shocks")),
        info=info,
        demand=demand,
        objective=objective,
        household=household,
        mechanism_params=dict(data.get("mechanism_params") or {}),
        stock_ratio=float(data.get("stock_ratio", 0.15)),
        max_stock_ratio=float(data.get("max_stock_ratio", 2.0)),
        investment_share=float(data.get("investment_share", 0.12)),
        gov_share=float(data.get("gov_share", 0.10)),
        depreciation=float(data.get("depreciation", 0.02)),
        discount=float(data.get("discount", 0.98)),
        labor_supply_factor=float(data.get("labor_supply_factor", 1.15)),
        indexation=float(data.get("indexation", 0.0)),
        price_bounds=(float(pb[0]), float(pb[1])),
        tags=tuple(str(t) for t in data.get("tags", ())),
        raw=dict(data),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load one scenario YAML file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise TypeError(f"{p} must contain a YAML mapping")
    sc = scenario_from_dict(data)
    if p.stem != sc.id:
        raise ValueError(f"scenario id {sc.id!r} does not match filename {p.stem!r}")
    return sc


def list_scenarios(directory: str | Path | None = None) -> list[Path]:
    """All ``*.yaml`` scenario files, sorted."""
    from planlab import paths

    d = Path(directory) if directory is not None else paths.scenarios_dir()
    if not d.exists():
        return []
    return sorted(d.glob("*.yaml"))


def scenario_hash(scenario: Scenario) -> str:
    """Stable content hash of a scenario, used in manifests."""
    payload = json.dumps(scenario.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_technology_spec(spec: Mapping[str, Any]):
    """Resolve a scenario's ``technology`` block into a :class:`Technology`.

    Kept here (rather than in the data layer) so that the engine never imports the
    data layer directly and can be tested with an injected technology.
    """
    from .io_model import toy_technology

    spec = dict(spec)
    if "toy" in spec:
        return toy_technology(n=int(spec.get("toy", 6)))
    try:
        from planlab.data import load_technology
    except Exception as exc:  # pragma: no cover - offline fallback
        raise RuntimeError(
            "the data layer is unavailable; pass an explicit Technology to run_scenario"
        ) from exc
    return load_technology(
        source=str(spec.get("source", "eurostat")),
        year=int(spec.get("year", 2020)),
        n_sectors=int(spec.get("n_sectors", 20)),
    )
