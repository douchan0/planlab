"""Dual-track coordination: a plan track at administered prices plus a market track.

**Institution.** Each good is split between two tracks.  A share ``theta`` is delivered
to plan claimants at an administered (plan) price, in an administrative priority order:
intermediate inputs first, then essential household deliveries, then other household
deliveries, then government, then investment.  The remaining ``1 - theta`` is sold on a
residual market at a market price that is revised in response to the excess demand the
board observes on that residual tier; the residual buyers are served pro rata, because
nobody in the market track has a rank.

**Historical referent.** China's dual-track reform after 1978 (plan track at state
prices, market track at negotiated prices for output above quota) and, more generally,
the "market socialism" hybrids in which an administered core coexists with a
marketised margin; also the two-tier pricing systems of late socialist economies.

**The experimental treatment is ``theta``** in ``[0, 1]``:

* ``theta = 1`` — pure plan: one administered price, priority tiers, no market track;
* ``theta = 0`` — pure market: no priorities (every final claimant pro rata), and the
  price is the residual-market price;
* interior ``theta`` — the plan track is served first, so it is *protected* when supply
  binds, and the market track absorbs the whole imbalance at a price that diverges from
  the plan price.  The plan-track share and the market/plan price gap are the classic
  dual-track statistics and are reported in ``Plan.meta``/``Allocation.meta``.

**Prices.** The plan price is administratively fixed by default (``admin_gain = 0``) and
can be indexed to the observed shortage (``admin_gain > 0``).  The market price follows
the *residual tier's* observed excess demand:

    gap_track = (residual demand - residual supply) / max(residual demand, residual supply)
    p_market'  = p_market * exp(market_gain * gap_track)

both sides of which are constructed only from :class:`Observation` (observed demand,
inventory, capacity) and the mechanism's own instruments — never from current-period
truth, and never from the recorded shortage rate of the *plan* track.  Because the
market-track gap is measured against the supply it can see rather than against the
recorded shortage, the market price cannot inflate the shortage that drives it (the
mechanical loop that would appear in a mechanism whose price rule feeds on the recorded
shortage, as in :mod:`planlab.mechanisms.tatonnement`).

The market-track price is confined to ``[price_floor, price_ceiling]`` (keep them equal
to the scenario's ``price_bounds``) so that the price the board records and the price
its claims are computed at are the same object.

**Claims are not deflated by the market price.**  Household claims are combined across
the two tracks and submitted in full: the engine's shortage metric is defined against
*notional* demand, so a mechanism that re-prices claims down would convert price
rationing into recorded shortage.  Household price response is a preference parameter of
the scenario (``demand.price_elasticity``), common to all mechanisms.
"""

from __future__ import annotations

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
from ._util import require_range
from .base import PRICE_EPS, CoordinationMechanism

__all__ = ["HybridDualTrack"]

# tier layout of the claim matrix (row order is documented in ``allocate``)
_TIER_INTER = 0.0
_TIER_PLAN_ESSENTIAL = 1.0
_TIER_PLAN_OTHER = 2.0
_TIER_PLAN_GOV = 3.0
_TIER_PLAN_INVEST = 4.0
_TIER_MARKET = 5.0
_TIER_STOCK = 6.0


class HybridDualTrack(CoordinationMechanism):
    """Plan track (share ``theta``, administered prices) + residual market track."""

    name = "hybrid_dual_track"

    default_params: Mapping[str, Any] = {
        #: share of each good sold at the plan price — THE experimental treatment
        "theta": 0.50,
        #: price responsiveness of the residual market
        "market_gain": 0.25,
        #: indexation of the administered (plan) price to the observed shortage
        "admin_gain": 0.00,
        #: speed at which the plan track's output quotas follow observed needs
        "quota_speed": 0.50,
        #: ceiling on the supply plan as a multiple of observed capacity
        "capacity_ceiling": 1.25,
        #: share of the observed backlog re-submitted as this period's household claim
        "backlog_carry": 0.30,
        #: desired stock level as a multiple of base output
        "stock_target_ratio": 0.15,
        #: bounds on the planned target, relative to base-year output
        "max_target_ratio": 2.50,
        "min_target_ratio": 0.25,
        #: band inside which the residual market posts its price; keep it equal to the
        #: scenario's ``price_bounds`` so posted prices and claim magnitudes agree
        "price_floor": 0.05,
        "price_ceiling": 20.0,
    }

    # ------------------------------------------------------------------ setup
    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(technology, params)
        self._theta = require_range("theta", self.param("theta", 0.5), 0.0, 1.0)
        self._market_gain = require_range(
            "market_gain", self.param("market_gain", 0.25), 0.0, 2.0
        )
        self._admin_gain = require_range("admin_gain", self.param("admin_gain", 0.0), 0.0, 2.0)
        self._quota_speed = require_range(
            "quota_speed", self.param("quota_speed", 0.5), 0.0, 1.0, lo_open=True
        )
        self._capacity_ceiling = require_range(
            "capacity_ceiling", self.param("capacity_ceiling", 1.25), 0.0, 5.0, lo_open=True
        )
        self._backlog_carry = require_range(
            "backlog_carry", self.param("backlog_carry", 0.3), 0.0, 1.0
        )
        self._stock_ratio = require_range(
            "stock_target_ratio", self.param("stock_target_ratio", 0.15), 0.0, 5.0
        )
        self._price_floor = require_range(
            "price_floor", self.param("price_floor", 0.05), 0.0, 1.0, lo_open=True, hi_open=True
        )
        self._price_ceiling = require_range(
            "price_ceiling", self.param("price_ceiling", 20.0), 1.0, 1e6, hi_open=True
        )
        if not self._price_floor < self._price_ceiling:
            raise ValueError("price_floor must be below price_ceiling")
        self._max_ratio = require_range(
            "max_target_ratio", self.param("max_target_ratio", 2.5), 0.0, 10.0, lo_open=True
        )
        self._min_ratio = require_range(
            "min_target_ratio", self.param("min_target_ratio", 0.25), 0.0, 1.0, lo_open=True
        )
        if self._min_ratio > self._max_ratio:
            raise ValueError("min_target_ratio must not exceed max_target_ratio")
        try:
            self._leontief_inverse = np.linalg.inv(np.eye(self.n) - np.asarray(self.tech.A, float))
        except np.linalg.LinAlgError as exc:  # pragma: no cover - productive A guarantees this
            raise ValueError("technology (I - A) is singular; A must be productive") from exc
        #: quota carried by the plan track (its own instrument, never an observation)
        self._quota = np.asarray(self.tech.base_output, dtype=float).copy()
        self._p_plan = np.ones(self.n)
        self._p_market = np.ones(self.n)
        self._diag: dict[str, float] = {"plan_track_share": 0.0, "market_price_dispersion": 0.0}
        self.prices = self._effective_price()
        self.config.update(
            theta=self._theta,
            market_gain=self._market_gain,
            admin_gain=self._admin_gain,
            quota_speed=self._quota_speed,
            capacity_ceiling=self._capacity_ceiling,
            max_target_ratio=self._max_ratio,
            min_target_ratio=self._min_ratio,
        )

    # ------------------------------------------------------------------ helpers
    def _effective_price(self) -> np.ndarray:
        """The price a buyer actually faces on average: ``theta * plan + (1-theta) * market``."""
        p = self._theta * self._p_plan + (1.0 - self._theta) * self._p_market
        return np.clip(np.maximum(p, PRICE_EPS), self._price_floor, self._price_ceiling)

    def _final_claims(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """``(household, other)`` notional final claims.

        Stockbuilding is excluded from the two *demand* sides used for planning and
        pricing: it is the residual claimant (tier 6) funded out of leftovers, so
        putting it into a price-inelastic demand block would crowd households out of
        the residual market and ratchet the market price for no economic reason.
        """
        household = self.requested_final(obs)
        other = self.gov_claim(obs) + self.invest_claim(obs)
        return household, other

    # ------------------------------------------------------------------ plan
    def form_plan(self, obs: Observation, state: State) -> Plan:
        """Blend the plan track's quota rule with the market track's demand forecast."""
        A = self.tech.A
        base = self.tech.base_output
        household, other = self._final_claims(obs)
        final_claim = household + other

        # --- plan track: administrative quotas, revised towards observed needs ----
        needs = A @ self._quota + final_claim
        self._quota = self._quota + self._quota_speed * (needs - self._quota)
        # --- market track: supply follows the Leontief requirement of observed demand
        market_supply = self._leontief_inverse @ final_claim

        ceiling = np.minimum(
            self._capacity_ceiling * np.maximum(np.asarray(obs.capacity, dtype=float), 0.0),
            self._max_ratio * base,
        )
        blended = self._theta * self._quota + (1.0 - self._theta) * market_supply
        targets = np.clip(np.minimum(blended, ceiling), self._min_ratio * base,
                          self._max_ratio * base)
        self.targets = targets

        # available final-goods flow the tracks can draw on (observed quantities)
        y = np.minimum(targets, self._capacity_ceiling * np.maximum(obs.capacity, 0.0))
        flow = np.maximum(y - A @ y, 0.0)
        plan_claim = self._theta * final_claim
        #: share of the observed final flow that the plan track claims (theta in calm
        #: conditions, rising above theta when the flow shrinks)
        total_flow = float(np.sum(flow))
        share = float(np.sum(plan_claim)) / total_flow if total_flow > 1e-12 else self._theta
        self._diag = {
            "plan_track_share": float(min(1.0, share)),
            "market_price_dispersion": float(
                np.std(np.log(np.maximum(self._p_market, PRICE_EPS)))
            )
            * (1.0 if self._theta < 1.0 else 0.0),
        }
        final_targets = np.minimum(
            household,
            flow + np.maximum(np.asarray(obs.inventory, dtype=float), 0.0),
        )
        return Plan(
            targets=targets.copy(),
            final_targets=np.maximum(final_targets, 0.0),
            meta={
                "status": "ok",
                "institution": self.name,
                "theta": self._theta,
                "plan_track_share": self._diag["plan_track_share"],
                "market_price_dispersion": self._diag["market_price_dispersion"],
                "effective_price_mean": float(np.mean(self._effective_price())),
                "lag": obs.lag,
                "effective_lag": obs.meta.get("effective_lag", obs.lag + 1),
            },
        )

    # -------------------------------------------------------------- allocate
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        """Two claim rows per final use: plan track (ranked) then market track (pro rata).

        Row layout (``n = number of sectors``)::

            0 .. n-1                      intermediate requirements   tier 0
            n                             final:essential:plan        tier 1
            n+1                           final:other:plan            tier 2
            n+2                           gov:plan                    tier 3
            n+3                           invest:plan                 tier 4
            n+4 .. n+7                    final:/gov:/invest: market  tier 5
            n+8                           stock:00                    tier 6

        This is the canonical :class:`~planlab.core.state.Allocation` format (rows of
        claims, a tier per row, labels carrying the four role prefixes); it is built
        directly rather than through :meth:`build_allocation` only because the
        dual track needs two rows per final use, which the helper does not express.
        """
        n = self.n
        household, other = self._final_claims(obs)
        gov = self.gov_claim(obs)
        invest = self.invest_claim(obs)
        stock = self.stock_claim(obs)
        essentials = np.asarray(self.tech.essentials, dtype=float) > 0.5
        theta = self._theta
        residual = 1.0 - theta

        inter = self.input_requirements(plan.targets)
        rows = [
            inter,
            (np.where(essentials, household, 0.0) * theta).reshape(1, -1),
            (np.where(~essentials, household, 0.0) * theta).reshape(1, -1),
            (gov * theta).reshape(1, -1),
            (invest * theta).reshape(1, -1),
            (np.where(essentials, household, 0.0) * residual).reshape(1, -1),
            (np.where(~essentials, household, 0.0) * residual).reshape(1, -1),
            (gov * residual).reshape(1, -1),
            (invest * residual).reshape(1, -1),
            stock.reshape(1, -1),
        ]
        claims = np.vstack(rows)
        tiers = np.concatenate([
            np.full(n, _TIER_INTER),
            np.array([
                _TIER_PLAN_ESSENTIAL,
                _TIER_PLAN_OTHER,
                _TIER_PLAN_GOV,
                _TIER_PLAN_INVEST,
                _TIER_MARKET,
                _TIER_MARKET,
                _TIER_MARKET,
                _TIER_MARKET,
                _TIER_STOCK,
            ], dtype=float),
        ])
        labels = (
            tuple(f"inter:{j:02d}" for j in range(n))
            + (
                "final:essential:plan",
                "final:other:plan",
                "gov:plan",
                "invest:plan",
                "final:essential:market",
                "final:other:market",
                "gov:market",
                "invest:market",
                "stock:00",
            )
        )
        plan_rows = slice(n, n + 4)
        market_rows = slice(n + 4, n + 8)
        alloc = Allocation(
            claims=claims,
            tiers=tiers,
            labels=labels,
            meta={
                "mechanism": self.name,
                "institution": self.name,
                "theta": theta,
                "plan_track_claims": float(claims[plan_rows].sum()),
                "market_track_claims": float(claims[market_rows].sum()),
                "plan_track_share": self._diag["plan_track_share"],
                "market_price_dispersion": self._diag["market_price_dispersion"],
                "market_price_mean": float(np.mean(self._p_market)),
                "plan_price_mean": float(np.mean(self._p_plan)),
                "market_plan_price_ratio": float(
                    np.mean(self._p_market / np.maximum(self._p_plan, PRICE_EPS))
                ),
            },
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
        """Index the plan price (optionally) and reprice the residual market track.

        Both updates use only the observation and the mechanism's own instruments.
        The market-track gap compares the *residual* demand ``(1-theta) * claim`` with
        the *residual* supply ``flow - theta * claim`` it can see, clipped to
        ``[-1, 1]``, so a market price above the plan price falls back as soon as the
        observed flow recovers.
        """
        A = self.tech.A
        household, other = self._final_claims(obs)
        final_claim = household + other
        theta = self._theta
        residual = 1.0 - theta

        # --- plan track: administered price, fixed unless indexed --------------
        if self._admin_gain > 0.0 and obs.shortage is not None:
            shortage = self.observed_shortage(obs)
            self._p_plan = np.maximum(
                self._p_plan * np.exp(self._admin_gain * shortage), PRICE_EPS
            )

        # --- market track: excess demand on the residual tier -------------------
        cap_obs = np.maximum(np.asarray(obs.capacity, dtype=float), 0.0)
        y = np.minimum(plan.targets, self._capacity_ceiling * cap_obs)
        flow = np.maximum(y - A @ y, 0.0)
        demand_residual = residual * final_claim
        supply_residual = np.maximum(flow - theta * final_claim, 0.0)
        denom = np.maximum(np.maximum(demand_residual, supply_residual), PRICE_EPS)
        gap = np.clip((demand_residual - supply_residual) / denom, -1.0, 1.0)
        if self._market_gain > 0.0 and residual > 0.0:
            self._p_market = np.clip(
                self._p_market * np.exp(self._market_gain * gap),
                self._price_floor,
                self._price_ceiling,
            )

        self.prices = self._effective_price()
        return self.prices.copy()
