"""Competitive market with within-period Walrasian tâtonnement.

**Institution.** No plan: firms bring supply, a market maker iterates a price vector
against excess demand *inside* the period until the market clears (or an iteration cap
is reached), the clearing price is carried into the next period, and goods are handed
out pro rata among the final claimants who pay that price.  Nobody is privileged:
households, the state and investors stand in the same queue.

**Historical referent.** The Walrasian auctioneer / Lange's "competitive solution", and
by extension the market-clearing benchmark of the computable general equilibrium
literature.  It is the price-coordinated counterfactual to quotas and material
balances.

**What "clearing" means under a Leontief technology.** With fixed technical
coefficients, intermediate demand is *technically determined*: no price changes the
input requirements ``A x``.  Only the final-use flow can clear.  The market therefore
clears

    flow supply  =  y - A y           (gross output net of intermediate use, y <= capacity)
    demand(p)    =  household(p) + government + investment + stockbuilding

goods by goods, and the inventory buffer is *not* auctioned: it is a stock the market
does not offer, drawn down by the engine's rationing only when the flow is
insufficient.  This is the only definition of clearing that is not degenerate at the
base year (a buffer-inclusive market would report permanent excess supply equal to the
buffer and deflate the price to the floor).

**Experimental treatments.** ``claim_elasticity`` (how strongly buyers' claims fall
with the price the market maker posts) and ``price_step`` / ``max_clearing_iters`` /
``clearing_tol`` (how fast, how long and how exactly the auctioneer iterates) are the
knobs; ``capacity_ceiling`` governs how far the supply plan may exceed observed
capacity; ``price_floor`` / ``price_ceiling`` must match the scenario's ``price_bounds``
so that the posted price is the price the claims were computed at.  Stockbuilding is
*not* a bidder in the auction: it is the residual claimant of the priority ladder.

**Relation to the scenario's demand elasticity.** The household price response is
*this mechanism's* demand schedule: at market prices a household buys
``exp(-claim_elasticity * log p)`` of its notional basket.  The engine's own
``demand.price_elasticity`` is a scenario-level preference parameter; if a scenario
switches both on, the two responses compose, so the recommended configuration for
market runs is ``demand.price_elasticity = 0`` unless the experiment is explicitly
about the composition.  Unlike the Lange board, the market price here is *not* driven
by the recorded shortage rate (it is computed by clearing against observed supply), so
this claim deflation cannot feed back into the price rule and cannot manufacture the
shortage that moves the price.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.state import Allocation, Observation, Plan, Realisation, State, Technology
from ._util import require_int, require_range
from .base import PRICE_EPS, CoordinationMechanism

__all__ = ["Market"]


class Market(CoordinationMechanism):
    """Price-clearing market: iterate prices against excess demand within the period."""

    name = "market"

    default_params: Mapping[str, Any] = {
        #: household claim response to the posted price, per unit of log price
        "claim_elasticity": 0.75,
        #: log-price step of the auctioneer per clearing iteration
        "price_step": 0.25,
        #: maximum clearing iterations per period
        "max_clearing_iters": 80,
        #: relative gap at which the market is declared clear
        "clearing_tol": 0.001,
        #: ceiling on the supply plan as a multiple of observed capacity
        "capacity_ceiling": 1.50,
        #: share of the observed backlog re-submitted as this period's household claim
        #: (a queue: granted-but-not-demanded quantities end up in inventory, which is
        #: what rebuilds the buffer stock after a shock)
        "backlog_carry": 0.30,
        #: desired stock level as a multiple of base output
        "stock_target_ratio": 0.15,
        #: bounds on the planned target, relative to base-year output
        "max_target_ratio": 2.50,
        "min_target_ratio": 0.25,
        #: band inside which the market posts its clearing price.  Keep it equal to the
        #: scenario's ``price_bounds``: the price vector the mechanism actually posts
        #: must be the one its claims were computed at.
        "price_floor": 0.05,
        "price_ceiling": 20.0,
    }

    # ------------------------------------------------------------------ setup
    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(technology, params)
        self._claim_elasticity = require_range(
            "claim_elasticity", self.param("claim_elasticity", 0.75), 0.0, 5.0, lo_open=True
        )
        self._price_step = require_range(
            "price_step", self.param("price_step", 0.25), 0.0, 1.0, lo_open=True
        )
        self._max_iters = require_int(
            "max_clearing_iters", self.param("max_clearing_iters", 80), 1, 400
        )
        self._clearing_tol = require_range(
            "clearing_tol", self.param("clearing_tol", 0.001), 0.0, 1.0, lo_open=True
        )
        self._capacity_ceiling = require_range(
            "capacity_ceiling", self.param("capacity_ceiling", 1.5), 0.0, 5.0, lo_open=True
        )
        self._backlog_carry = require_range(
            "backlog_carry", self.param("backlog_carry", 0.3), 0.0, 1.0
        )
        self._price_floor = require_range(
            "price_floor", self.param("price_floor", 0.05), 0.0, 1.0, lo_open=True, hi_open=True
        )
        self._price_ceiling = require_range(
            "price_ceiling", self.param("price_ceiling", 20.0), 1.0, 1e6, hi_open=True
        )
        if not self._price_floor < self._price_ceiling:
            raise ValueError("price_floor must be below price_ceiling")
        self._stock_ratio = require_range(
            "stock_target_ratio", self.param("stock_target_ratio", 0.15), 0.0, 5.0
        )
        self._max_ratio = require_range(
            "max_target_ratio", self.param("max_target_ratio", 2.5), 0.0, 10.0, lo_open=True
        )
        self._min_ratio = require_range(
            "min_target_ratio", self.param("min_target_ratio", 0.25), 0.0, 1.0, lo_open=True
        )
        if self._min_ratio > self._max_ratio:
            raise ValueError("min_target_ratio must not exceed max_target_ratio")
        #: Leontief inverse, cached: it is a property of the constant technology.
        try:
            self._leontief_inverse = np.linalg.inv(np.eye(self.n) - np.asarray(self.tech.A, float))
        except np.linalg.LinAlgError as exc:  # pragma: no cover - productive A guarantees this
            raise ValueError("technology (I - A) is singular; A must be productive") from exc
        self._diag: dict[str, float] = {"clearing_iters": 0.0, "clearing_gap": 0.0}
        self.config.update(
            claim_elasticity=self._claim_elasticity,
            price_step=self._price_step,
            max_clearing_iters=self._max_iters,
            clearing_tol=self._clearing_tol,
            capacity_ceiling=self._capacity_ceiling,
            max_target_ratio=self._max_ratio,
            min_target_ratio=self._min_ratio,
        )

    # ------------------------------------------------------------------ helpers
    def _final_claims(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """``(household, other)`` *notional* final claims implied by the observation.

        Stockbuilding is deliberately **excluded**: it is the residual claimant of the
        priority ladder (tier 5), funded out of leftovers, not a buyer that competes in
        the auction.  Including it would make a price-inelastic claim crowd household
        demand out of the clearing computation and drive the price to its ceiling for
        no economic reason.  Government and investment are genuine buyers at the market
        price (they are quantity-constrained, not marginal).
        """
        household = self.requested_final(obs)
        other = self.gov_claim(obs) + self.invest_claim(obs)
        return household, other

    def _supply_plan(self, obs: Observation) -> np.ndarray:
        """Firms supply the Leontief requirement of the final demand they observe."""
        household, other = self._final_claims(obs)
        requirement = self._leontief_inverse @ (household + other)
        base = self.tech.base_output
        ceiling = np.minimum(
            self._capacity_ceiling * np.maximum(np.asarray(obs.capacity, dtype=float), 0.0),
            self._max_ratio * base,
        )
        return np.clip(np.minimum(requirement, ceiling), self._min_ratio * base,
                       self._max_ratio * base)

    def _quantity_at_price(self, household: np.ndarray, price: np.ndarray) -> np.ndarray:
        """Household claim at the posted price (the market's demand schedule)."""
        logp = np.clip(np.log(np.maximum(price, PRICE_EPS)), -20.0, 20.0)
        return household * np.exp(-self._claim_elasticity * logp)

    def _clear(self, household: np.ndarray, other: np.ndarray, supply_flow: np.ndarray,
               start: np.ndarray) -> tuple[np.ndarray, int, float]:
        """Iterate the price against excess demand until the flow market clears.

        The posted price is confined to ``[price_floor, price_ceiling]``: there is no
        point computing claims at a price the exchange is not allowed to post, and in a
        capacity-constrained Leontief economy the demand schedule can need an
        arbitrarily high price to clear a good whose entire output is an intermediate.
        """
        price = np.clip(np.asarray(start, dtype=float), self._price_floor, self._price_ceiling)
        scale = np.maximum(np.maximum(household + other, supply_flow), 1e-12)
        gap = 0.0
        iters = 0
        for iters in range(1, self._max_iters + 1):
            demand = self._quantity_at_price(household, price) + other
            gap = float(np.max(np.abs(demand - supply_flow) / scale))
            if gap <= self._clearing_tol:
                break
            step = self._price_step * np.clip((demand - supply_flow) / scale, -1.0, 1.0)
            price = np.clip(price * np.exp(step), self._price_floor, self._price_ceiling)
        return price, iters, gap

    # ------------------------------------------------------------------ plan
    def form_plan(self, obs: Observation, state: State) -> Plan:
        """Plan supply, then clear the final-goods market within the period."""
        targets = self._supply_plan(obs)
        self.targets = targets

        household, other = self._final_claims(obs)
        y = np.minimum(
            targets,
            self._capacity_ceiling * np.maximum(np.asarray(obs.capacity, dtype=float), 0.0),
        )
        supply_flow = np.maximum(y - np.asarray(self.tech.A, dtype=float) @ y, 0.0)
        start = np.asarray(state.price, dtype=float).reshape(-1)
        clearing_price, iters, gap = self._clear(household, other, supply_flow, start)

        self.prices = clearing_price.copy()
        self._diag = {"clearing_iters": float(iters), "clearing_gap": float(gap)}
        final_targets = np.minimum(household, supply_flow + np.maximum(
            np.asarray(obs.inventory, dtype=float), 0.0))
        return Plan(
            targets=targets.copy(),
            final_targets=np.maximum(final_targets, 0.0),
            meta={
                "status": "ok",
                "institution": self.name,
                "clearing_iters": float(iters),
                "clearing_gap": float(gap),
                "clearing_price_mean": float(np.mean(clearing_price)),
                "lag": obs.lag,
                "effective_lag": obs.meta.get("effective_lag", obs.lag + 1),
            },
        )

    # -------------------------------------------------------------- allocate
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        """Pro rata among *all* final claimants at the clearing price; no priorities."""
        household, other = self._final_claims(obs)
        household_claim = self._quantity_at_price(household, self.prices)
        alloc = self.build_allocation(
            targets=plan.targets,
            obs=obs,
            final=household_claim,
            split_essentials=True,
            tier_inter=0.0,
            tier_final=1.0,
            tier_final_other=1.0,
            tier_gov=1.0,
            tier_invest=1.0,
            tier_stock=5.0,
        )
        # government, investment and stockbuilding are quantity buyers at the market
        # price: they are not re-priced (they are budget-constrained, not marginal).
        alloc.meta.update(
            institution=self.name,
            clearing_price_mean=float(np.mean(self.prices)),
            clearing_price_dispersion=float(np.std(np.log(np.maximum(self.prices, PRICE_EPS)))),
            clearing_iters=self._diag["clearing_iters"],
            clearing_gap=self._diag["clearing_gap"],
            household_claim_scale=float(
                np.mean(household_claim / np.maximum(household, 1e-12))
            ),
        )
        return alloc

    # ---------------------------------------------------------------- prices
    def update_prices(
        self,
        obs: Observation,
        state: State,
        plan: Plan,
        alloc: Allocation,
        realised: Realisation,
    ) -> np.ndarray:
        """Carry the clearing price computed by the auctioneer this period.

        ``realised`` is ignored: the market maker's price is the one that cleared the
        flow it could see, not a reaction to this period's realised deliveries.
        """
        return np.asarray(self.prices, dtype=float).copy()
