"""The institution interface.

A coordination mechanism is the *only* thing that differs between the treatments
of the experiment.  It makes three decisions per period and nothing else:

``form_plan``
    what gross output to aim for, given the observation;
``allocate``
    who has a claim on what, and in which priority order;
``update_prices``
    what next period's prices are.

Everything quantitative (accounting, feasibility, rationing arithmetic) lives in
:mod:`planlab.core.engine`, which is what makes the comparison identified: two
mechanisms differ in institutions, never in algebra.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from ..core.state import (
    Allocation,
    Observation,
    Plan,
    Realisation,
    State,
    Technology,
)

__all__ = ["CoordinationMechanism", "PRICE_EPS"]


PRICE_EPS = 1e-9


class CoordinationMechanism(ABC):
    """Base class for all institutions.

    Subclasses set :attr:`name` (the registry key), :attr:`default_params` and
    implement the three decisions.  The default price rule keeps administered
    prices fixed, which is the historical default of a planned economy.
    """

    name: str = "base"
    default_params: Mapping[str, Any] = {}

    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        self.tech = technology
        self.n = technology.n
        cfg = dict(self.default_params)
        cfg.update(dict(params or {}))
        self.config = cfg
        #: the planner's own carried-over output targets (an instrument, not a shock)
        self.targets = np.asarray(technology.base_output, dtype=float).copy()
        self.prices = np.ones(self.n)
        self.t = 0

    # ------------------------------------------------------------------ API
    @abstractmethod
    def form_plan(self, obs: Observation, state: State) -> Plan:
        """Decide gross-output targets from the (stale, noisy) observation."""

    @abstractmethod
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        """State who claims what, and with which priority."""

    def update_prices(
        self,
        obs: Observation,
        state: State,
        plan: Plan,
        alloc: Allocation,
        realised: Realisation,
    ) -> np.ndarray:
        """Return next period's price vector (default: administered prices fixed)."""
        return np.asarray(state.price, dtype=float).copy()

    # -------------------------------------------------------------- helpers
    def param(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def observed_shortage(self, obs: Observation) -> np.ndarray:
        """The feedback signal: realised shortage rates, ``k`` periods old."""
        if obs.shortage is not None:
            return np.clip(np.asarray(obs.shortage, dtype=float), 0.0, 1.0)
        return np.zeros(self.n)

    def requested_final(self, obs: Observation) -> np.ndarray:
        """Total household claim: this period's demand plus carried-over backlog."""
        carry = float(self.param("backlog_carry", 0.0))
        backlog = obs.backlog if obs.backlog is not None else np.zeros(self.n)
        return np.maximum(np.asarray(obs.demand, dtype=float) + carry * backlog, 0.0)

    def gov_claim(self, obs: Observation) -> np.ndarray:
        if obs.gov is None:
            return np.zeros(self.n)
        return np.maximum(np.asarray(obs.gov, dtype=float), 0.0)

    def invest_claim(self, obs: Observation) -> np.ndarray:
        if obs.invest is None:
            return np.zeros(self.n)
        return np.maximum(np.asarray(obs.invest, dtype=float), 0.0)

    def stock_claim(self, obs: Observation) -> np.ndarray:
        """Desired stockbuilding: top up towards a target stock level."""
        ratio = float(self.param("stock_target_ratio", 0.0))
        if ratio <= 0:
            return np.zeros(self.n)
        target = ratio * self.tech.base_output
        inv = obs.inventory if obs.inventory is not None else np.zeros(self.n)
        return np.maximum(target - inv, 0.0)

    def input_requirements(self, targets: np.ndarray) -> np.ndarray:
        """``(n, n)`` matrix with row ``j`` = sector j's input requirement."""
        return np.asarray(self.tech.A * np.asarray(targets, dtype=float).reshape(1, -1)).T

    def build_allocation(
        self,
        *,
        targets: np.ndarray,
        obs: Observation,
        final: np.ndarray | None = None,
        split_essentials: bool = True,
        tier_inter: float = 0.0,
        tier_final: float = 1.0,
        tier_final_other: float = 2.0,
        tier_gov: float = 3.0,
        tier_invest: float = 4.0,
        tier_stock: float = 5.0,
        include: tuple[str, ...] = ("gov", "invest", "stock"),
    ) -> Allocation:
        """Assemble the canonical claim set for a plan."""
        fin = self.requested_final(obs) if final is None else np.asarray(final, dtype=float)
        alloc = Allocation.build(
            intermediates=self.input_requirements(targets),
            final=fin,
            gov=self.gov_claim(obs) if "gov" in include else None,
            invest=self.invest_claim(obs) if "invest" in include else None,
            to_inventory=self.stock_claim(obs) if "stock" in include else None,
            essentials=self.tech.essentials if split_essentials else None,
            tier_inter=tier_inter,
            tier_final=tier_final,
            tier_final_other=tier_final_other,
            tier_gov=tier_gov,
            tier_invest=tier_invest,
            tier_stock=tier_stock,
            meta={"mechanism": self.name},
        )
        return alloc

    # ---------------------------------------------------------- price rules
    def admin_price_rule(self, obs: Observation, state: State, gain: float) -> np.ndarray:
        """Administered price revision: ``log p' = log p + gain * shortage``.

        This is the Arthurian/Gosplan rule: raise the price of a good whose shortage
        has been observed, by a fraction of that shortage.  Delayed feedback of this
        form is the object of Proposition 1 in the paper.
        """
        if gain <= 0:
            return np.asarray(state.price, dtype=float).copy()
        s = self.observed_shortage(obs)
        logp = np.log(np.maximum(state.price, PRICE_EPS)) + float(gain) * s
        return np.exp(logp)

    def excess_demand_rule(self, obs: Observation, state: State, gain: float,
                           excess: np.ndarray) -> np.ndarray:
        """Price revision driven by excess demand relative to observed supply."""
        if gain <= 0:
            return np.asarray(state.price, dtype=float).copy()
        denom = np.maximum(np.asarray(obs.demand, dtype=float), PRICE_EPS)
        ratio = np.clip(np.asarray(excess, dtype=float) / denom, -1.0, 1.0)
        logp = np.log(np.maximum(state.price, PRICE_EPS)) + float(gain) * ratio
        return np.exp(logp)
