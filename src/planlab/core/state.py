"""Core data structures of PlanLab.

Everything the engine and the mechanisms exchange is defined here.  These classes
are deliberately dumb containers: all behaviour (accounting, feasibility,
rationing) lives in :mod:`planlab.core.engine`, so that institutions can be swapped
without touching the physics of the economy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = [
    "Technology",
    "State",
    "Observation",
    "Plan",
    "Allocation",
    "Realisation",
    "PeriodRecord",
    "MECHANISM_ROW_HOUSEHOLD",
    "MECHANISM_ROW_GOV",
    "MECHANISM_ROW_INVEST",
    "MECHANISM_ROW_STOCK",
    "TOL_ACCOUNTING",
]

#: absolute tolerance used when auditing the per-sector balance identity
TOL_ACCOUNTING = 1e-8

# Row offsets of the non-intermediate claimants in ``Allocation.priority``.
# Intermediates occupy rows ``0 .. n-1`` (one per demanding sector).
MECHANISM_ROW_HOUSEHOLD = "household"
MECHANISM_ROW_GOV = "gov"
MECHANISM_ROW_INVEST = "invest"
MECHANISM_ROW_STOCK = "stock"


def _as_vector(name: str, value: Any, n: int, *, nonneg: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if nonneg and np.any(arr < 0):
        raise ValueError(f"{name} contains negative values")
    return arr


@dataclass(frozen=True)
class Technology:
    """A Leontief technology plus the socioeconomic vectors the model needs.

    All quantities are in constant-price value units of the source table
    (see ``docs/DATA_SCHEMA.md``).  Nothing here is period specific.
    """

    sectors: tuple[str, ...]
    A: np.ndarray
    labor: np.ndarray
    kappa: np.ndarray
    consumption_weights: np.ndarray
    essentials: np.ndarray
    base_output: np.ndarray
    base_final: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        sectors = tuple(str(s) for s in self.sectors)
        n = len(sectors)
        if n == 0:
            raise ValueError("Technology needs at least one sector")
        A = np.array(self.A, dtype=float, copy=True)
        if A.shape != (n, n):
            raise ValueError(f"A must be ({n},{n}), got {A.shape}")
        if not np.all(np.isfinite(A)):
            raise ValueError("A contains non-finite values")
        if np.any(A < 0):
            raise ValueError("A contains negative coefficients")
        colsum = A.sum(axis=0)
        if np.any(colsum >= 1.0):
            bad = np.where(colsum >= 1.0)[0]
            raise ValueError(
                "A must be productive: every column sum < 1 "
                f"(violating sectors: {[sectors[i] for i in bad]}, sums={colsum[bad]})"
            )
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "labor", _as_vector("labor", self.labor, n))
        kappa = _as_vector("kappa", self.kappa, n)
        if np.any(kappa <= 0):
            raise ValueError("kappa must be strictly positive")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(
            self, "consumption_weights",
            _as_vector("consumption_weights", self.consumption_weights, n),
        )
        ess = _as_vector("essentials", self.essentials, n)
        object.__setattr__(self, "essentials", (ess > 0.5).astype(float))
        base_output = _as_vector("base_output", self.base_output, n)
        if np.any(base_output <= 0):
            raise ValueError("base_output must be strictly positive for every sector")
        object.__setattr__(self, "base_output", base_output)
        object.__setattr__(self, "base_final", _as_vector("base_final", self.base_final, n))
        object.__setattr__(self, "meta", dict(self.meta or {}))

    # ------------------------------------------------------------------ helpers
    @property
    def n(self) -> int:
        return len(self.sectors)

    def capacity_from_capital(self, capital: np.ndarray) -> np.ndarray:
        """Gross-output capacity implied by the capital stock."""
        return np.asarray(capital, dtype=float) / self.kappa

    def sector_index(self, name: str) -> int:
        try:
            return self.sectors.index(name)
        except ValueError as exc:  # pragma: no cover - convenience
            raise KeyError(f"unknown sector {name!r}") from exc

    def leontief_inverse(self) -> np.ndarray:
        return np.linalg.inv(np.eye(self.n) - self.A)

    def validate_identity(self, x: np.ndarray, y: np.ndarray, rtol: float = 1e-6) -> None:
        """Check ``x = A x + y`` on a candidate equilibrium (used by tests/validation)."""
        resid = self.A @ x + y - x
        scale = max(1.0, float(np.max(np.abs(x))))
        if np.max(np.abs(resid)) > rtol * scale:
            raise ValueError(
                f"input-output identity violated: max residual {np.max(np.abs(resid)):.3e} "
                f"> {rtol:.1e} * {scale:.3e}"
            )


@dataclass
class State:
    """Mutable per-run state.  Mechanisms must treat it as read-only."""

    t: int
    x: np.ndarray            # realised gross output of the previous period
    inventory: np.ndarray    # beginning-of-period stocks
    capital: np.ndarray
    capacity: np.ndarray
    price: np.ndarray
    backlog: np.ndarray
    gov_demand: np.ndarray

    def copy(self) -> "State":
        return State(
            t=self.t,
            x=self.x.copy(),
            inventory=self.inventory.copy(),
            capital=self.capital.copy(),
            capacity=self.capacity.copy(),
            price=self.price.copy(),
            backlog=self.backlog.copy(),
            gov_demand=self.gov_demand.copy(),
        )

    @classmethod
    def initial(cls, tech: Technology, *, stock_ratio: float, price_level: float = 1.0) -> "State":
        n = tech.n
        inventory = stock_ratio * tech.base_output
        # capital consistent with base-year output: K = kappa * x
        capital = tech.kappa * tech.base_output
        return cls(
            t=0,
            x=tech.base_output.copy(),
            inventory=inventory,
            capital=capital,
            capacity=tech.capacity_from_capital(capital),
            price=np.full(n, price_level),
            backlog=np.zeros(n),
            gov_demand=np.zeros(n),
        )


@dataclass(frozen=True)
class Observation:
    """What a coordinator is allowed to see: a stale, noisy picture of the economy.

    No field of this object is ever current truth when ``lag > 0``; mechanisms must
    not reach past it into :class:`State` for policy-relevant quantities.
    """

    t: int
    lag: int
    demand: np.ndarray
    capacity: np.ndarray
    inventory: np.ndarray
    price: np.ndarray
    backlog: np.ndarray
    labour: float
    stock_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: realised shortage rate per sector (relative unmet demand), from period t-lag.
    #: This is the *feedback signal* of a revision rule: a planner or price board
    #: can only react to shortages it has already seen.
    shortage: np.ndarray | None = None
    #: government and investment deliveries *planned* by the coordinator itself.
    #: These are policy instruments, not observations, so they are not delayed.
    gov: np.ndarray | None = None
    invest: np.ndarray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)



@dataclass
class Plan:
    """Output of a mechanism's planning stage."""

    targets: np.ndarray          # (n,) gross output targets
    final_targets: np.ndarray    # (n,) intended deliveries to households
    priority: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.targets = np.asarray(self.targets, dtype=float).reshape(-1)
        self.final_targets = np.asarray(self.final_targets, dtype=float).reshape(-1)
        if self.targets.shape != self.final_targets.shape:
            raise ValueError("targets and final_targets must have the same shape")
        if not np.all(np.isfinite(self.targets)) or np.any(self.targets < 0):
            raise ValueError("plan targets must be finite and non-negative")
        if not np.all(np.isfinite(self.final_targets)) or np.any(self.final_targets < 0):
            raise ValueError("plan final_targets must be finite and non-negative")


@dataclass
class Allocation:
    """A mechanism's claim on the available supply, before feasibility is enforced.

    Row convention
    --------------
    ``claims[:n]`` are the *input requirements of the sectors*: ``claims[j, i]`` is
    how much of good ``i`` sector ``j`` asks for.  The remaining rows are final
    claimants, one row per priority group, labelled with a prefix:

    ==============  ==========================================================
    ``final:*``     household deliveries (several rows = several priority groups,
                    e.g. ``final:essential`` and ``final:other``)
    ``gov:*``       government / collective consumption
    ``invest:*``    investment goods
    ``stock:*``     desired stockbuilding
    ==============  ==========================================================

    ``tiers`` gives the priority of each row (lower is served first); the engine
    fills goods one at a time, in ascending tier order, pro-rata within a tier.
    Splitting households into ``essential`` and ``other`` rows is how a
    food-and-energy-first rationing institution is expressed.
    """

    claims: np.ndarray
    tiers: np.ndarray
    labels: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        claims = np.asarray(self.claims, dtype=float)
        if claims.ndim != 2:
            raise ValueError("claims must be a 2-D (rows, goods) array")
        m, n = claims.shape
        if m < n:
            raise ValueError(
                f"allocation needs at least n={n} rows (one per sector), got {m}"
            )
        if np.any(claims < 0) or not np.all(np.isfinite(claims)):
            raise ValueError("claims must be finite and non-negative")
        tiers = np.asarray(self.tiers, dtype=float).reshape(-1)
        if tiers.shape != (m,):
            raise ValueError(f"tiers must have shape ({m},), got {tiers.shape}")
        if not np.all(np.isfinite(tiers)):
            raise ValueError("tiers must be finite")
        labels = tuple(self.labels) if self.labels else tuple(
            [f"inter:{j:02d}" for j in range(n)]
            + [f"final:{k:02d}" for k in range(m - n)]
        )
        if len(labels) != m:
            raise ValueError(f"labels must have {m} entries, got {len(labels)}")
        for lab in labels[:n]:
            if not lab.startswith("inter:"):
                raise ValueError(f"row {lab!r}: the first n rows must be 'inter:*'")
        valid = ("final:", "gov:", "invest:", "stock:")
        for lab in labels[n:]:
            if not lab.startswith(valid):
                raise ValueError(
                    f"row {lab!r}: final claimants must start with one of {valid}"
                )
        if not any(lab.startswith("final:") for lab in labels[n:]):
            raise ValueError("allocation must contain at least one 'final:*' claim row")
        self.claims = claims
        self.tiers = tiers
        self.labels = labels
        self.meta = dict(self.meta or {})

    # ------------------------------------------------------------------ views
    @property
    def n(self) -> int:
        return self.claims.shape[1]

    def role_mask(self, prefix: str) -> np.ndarray:
        """Boolean mask of the claimant rows whose label starts with ``prefix``."""
        return np.array([lab.startswith(prefix) for lab in self.labels], dtype=bool)

    def split_grants(self, grants: np.ndarray) -> dict[str, np.ndarray]:
        """Map granted quantities back to the four economic uses.

        Returns a dict with ``intermediates`` (``(n, n)``, ``[j, i]`` = good i
        granted to sector j), ``final``, ``gov``, ``invest`` and ``stock``
        (each summed over their rows).
        """
        grants = np.asarray(grants, dtype=float)
        n = self.n
        out = {
            "intermediates": grants[:n, :],
            "final": grants[self.role_mask("final:")].sum(axis=0),
            "gov": grants[self.role_mask("gov:")].sum(axis=0),
            "invest": grants[self.role_mask("invest:")].sum(axis=0),
            "stock": grants[self.role_mask("stock:")].sum(axis=0),
        }
        return out

    @classmethod
    def build(
        cls,
        *,
        intermediates: np.ndarray,
        final: np.ndarray,
        gov: np.ndarray | None = None,
        invest: np.ndarray | None = None,
        to_inventory: np.ndarray | None = None,
        essentials: np.ndarray | None = None,
        tier_inter: float = 0.0,
        tier_final: float = 1.0,
        tier_essential: float | None = None,
        tier_final_other: float = 2.0,
        tier_gov: float = 3.0,
        tier_invest: float = 4.0,
        tier_stock: float = 5.0,
        meta: dict | None = None,
    ) -> "Allocation":
        """Canonical constructor used by the shipped mechanisms.

        ``intermediates`` must be ``(n, n)`` with row ``j`` = sector j's input
        requirement.  If ``essentials`` (a boolean ``(n,)`` mask) is given, the
        household claim is split into a ``final:essential`` row and a
        ``final:other`` row so that essential goods can be protected.

        Default priority ladder (lower tier = served first)::

            0  intermediate inputs  production must not be starved
            1  final:essential      food, energy, health, transport
            2  final:other          non-essential consumption
            3  gov                  collective consumption
            4  invest               capital formation
            5  stock                stockbuilding

        The ordering is a *modelling choice with institutional content*, not a
        technical detail.  Putting intermediate inputs first is the "production
        first" ladder of a mobilisation economy; putting essential consumption
        first (``tier_inter=1.0, tier_final=0.0``) is the "consumption first"
        ladder.  The latter is available as a treatment and is *unstable*: fixed
        final claims crowd out the inputs needed to produce the goods those claims
        are made of, so a small shock can spiral.  That contrast is one of the
        results of the paper, not a defect of the engine.
        """
        n = np.asarray(final).shape[0]
        rows = [np.asarray(intermediates, dtype=float)]
        tiers = [np.full(n, float(tier_inter))]
        labels = [f"inter:{j:02d}" for j in range(n)]
        final = np.asarray(final, dtype=float)
        if essentials is None:
            rows.append(final.reshape(1, -1))
            tiers.append(np.array([float(tier_final)]))
            labels.append("final:00")
        else:
            ess = np.asarray(essentials, dtype=float).reshape(-1)
            if ess.shape != (n,):
                raise ValueError("essentials mask must have shape (n,)")
            mask = ess > 0.5
            t_ess = float(tier_final if tier_essential is None else tier_essential)
            if mask.any():
                rows.append(np.where(mask, final, 0.0).reshape(1, -1))
                tiers.append(np.array([t_ess]))
                labels.append("final:essential")
            if (~mask).any():
                rows.append(np.where(~mask, final, 0.0).reshape(1, -1))
                tiers.append(np.array([float(tier_final_other)]))
                labels.append("final:other")
        for name, vec, tier in (
            ("gov", gov, tier_gov),
            ("invest", invest, tier_invest),
            ("stock", to_inventory, tier_stock),
        ):
            if vec is None:
                continue
            rows.append(np.asarray(vec, dtype=float).reshape(1, -1))
            tiers.append(np.array([float(tier)]))
            labels.append(f"{name}:00")
        return cls(
            claims=np.vstack(rows),
            tiers=np.concatenate(tiers),
            labels=tuple(labels),
            meta=dict(meta or {}),
        )


@dataclass(frozen=True)
class Realisation:
    """Read-only summary of what actually happened in a period (for price updates)."""

    x: np.ndarray
    Z: np.ndarray
    C: np.ndarray
    G: np.ndarray
    invest: np.ndarray
    inventory_end: np.ndarray
    demand: np.ndarray
    unmet: np.ndarray
    supply: np.ndarray
    capacity: np.ndarray

    @property
    def excess_demand(self) -> np.ndarray:
        return self.demand - self.C

    @property
    def shortage_rate(self) -> float:
        d = float(self.demand.sum())
        return float(self.unmet.sum()) / d if d > 0 else 0.0


@dataclass
class PeriodRecord:
    t: int
    x: np.ndarray
    Z: np.ndarray
    C: np.ndarray
    G: np.ndarray
    invest: np.ndarray
    inventory_end: np.ndarray
    demand: np.ndarray
    unmet: np.ndarray
    price: np.ndarray
    capacity: np.ndarray
    plan_targets: np.ndarray
    plan_final_targets: np.ndarray
    supply: np.ndarray
    plan_status: str
    iterations: int
    metrics: dict = field(default_factory=dict)
