"""Lange-type market socialism: an administered price board plus a supply response.

**Institution.** A central price board (Lange's "Central Planning Board", emulated in
practice by the CPB/price commissions of the socialist economies) sets prices
administratively and revises them in proportion to the *shortage it has already
observed*; enterprises plan their output in response to the observed price signal;
and the residual allocation among claimants follows a wider willingness-to-pay order
than under quotas.  There is no within-period auction and no central quantity plan:
the board's instrument is the price vector.

**Historical referent.** Oskar Lange's market-socialist model (1936/37) and the
"shadow-price" coordination school; the administered price revisions of the Soviet
*Goskomtsen* and of the post-1965 reforms.

**The experimental treatment is ``price_gain``** (``eta`` in the paper's Proposition 1).
The price rule is deliberately the *exact* object of the delay theorem::

    log p_{t+1} = log p_t + price_gain * s_observed

where ``s_observed`` is :attr:`Observation.shortage` — the realised shortage rate of
period ``t - 1 - lag``.  The engine's observation already carries the reporting lag
(``effective_lag = lag + 1``), so the mechanism adds **no** extra period of lag and
reads **no** current-period truth.  This is what makes the simulated stability
boundary comparable with ``g*(m) = 2 sin(pi / (2(2m+1)))`` for ``m = lag + 1``.

**Supply response.** Enterprises raise planned output for goods whose observed price
stands above the observed *input cost* of producing them (and cut it for goods whose
price lags their costs).  The signal is scale-invariant — both price and cost are
measured at the same observed price level — so a general rise of the price level
(general inflation) does not expand every target, only a change in *relative* prices
reallocates the plan.  This is the mechanism through which prices have a real effect
in a Leontief economy with fixed technical coefficients.

**Prices ration capital formation, not subsistence consumption.** Investment claims are
the price-sensitive residual buyer: dear goods are bought for investment only in
smaller quantities, which frees supply for household deliveries.  Household claims are
submitted in full (see the note on ``claim_elasticity`` below) because the shortage
metric is defined against *notional* household demand.

.. warning::

   ``claim_elasticity`` (default ``0.0``) would deflate household claims by
   ``exp(-claim_elasticity * log p)``.  It is available as a diagnostic treatment but
   is *off by default*, because this mechanism's price rule feeds on the recorded
   shortage rate: deflating claims lowers granted deliveries, which raises the
   recorded shortage, which raises prices again, giving the purely mechanical loop
   ``u_{t+1} = (1 + price_gain * claim_elasticity) u_t`` — unconditionally explosive
   and unrelated to production.  Price-mediated demand belongs in the household
   demand process of the scenario (``demand.price_elasticity``), which is common to
   all mechanisms and therefore keeps the comparison identified.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.state import Allocation, Observation, Plan, Realisation, State, Technology
from ._util import require_range
from .base import PRICE_EPS, CoordinationMechanism

__all__ = ["Tatonnement"]


class Tatonnement(CoordinationMechanism):
    """Lange price board: delayed shortage-proportional price revision."""

    name = "tatonnement"

    default_params: Mapping[str, Any] = {
        #: feedback gain of the price rule (eta in Proposition 1) — the treatment
        "price_gain": 0.20,
        #: responsiveness of planned output to the observed price/cost signal
        "supply_gain": 0.15,
        #: clip on the observed price/cost signal (in logs) before it drives targets
        "relative_price_clip": 0.50,
        #: price sensitivity of investment claims (0 = price-inelastic investment)
        "investment_price_elasticity": 0.50,
        #: price sensitivity of household claims (0 = claims are notional demand;
        #: see the module warning before switching this on)
        "claim_elasticity": 0.0,
        #: ceiling on the planned target as a multiple of observed capacity
        "capacity_ceiling": 1.50,
        #: share of the observed backlog added to this period's household claim
        "backlog_carry": 0.0,
        #: desired stock level as a multiple of base output
        "stock_target_ratio": 0.15,
        #: bounds on the planned target, relative to base-year output
        "max_target_ratio": 2.50,
        "min_target_ratio": 0.25,
    }

    # ------------------------------------------------------------------ setup
    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(technology, params)
        self._price_gain = require_range("price_gain", self.param("price_gain", 0.2), 0.0, 2.0)
        self._supply_gain = require_range("supply_gain", self.param("supply_gain", 0.15), 0.0, 1.0)
        self._rel_clip = require_range(
            "relative_price_clip", self.param("relative_price_clip", 0.5), 0.0, 2.0, lo_open=True
        )
        self._inv_elasticity = require_range(
            "investment_price_elasticity",
            self.param("investment_price_elasticity", 0.5),
            0.0,
            5.0,
        )
        self._claim_elasticity = require_range(
            "claim_elasticity", self.param("claim_elasticity", 0.0), 0.0, 5.0
        )
        self._capacity_ceiling = require_range(
            "capacity_ceiling", self.param("capacity_ceiling", 1.5), 0.0, 5.0, lo_open=True
        )
        self._backlog_carry = require_range(
            "backlog_carry", self.param("backlog_carry", 0.0), 0.0, 1.0
        )
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
        #: base-year unit cost of each sector at the numeraire (p = 1, wage = 1);
        #: used to normalise the price/cost signal so that it vanishes in the base year
        self._base_cost = np.asarray(self.tech.A.T @ np.ones(self.n), dtype=float) + np.asarray(
            self.tech.labor, dtype=float
        )
        self._base_cost = np.maximum(self._base_cost, PRICE_EPS)
        self.config.update(
            price_gain=self._price_gain,
            supply_gain=self._supply_gain,
            relative_price_clip=self._rel_clip,
            investment_price_elasticity=self._inv_elasticity,
            claim_elasticity=self._claim_elasticity,
            capacity_ceiling=self._capacity_ceiling,
            max_target_ratio=self._max_ratio,
            min_target_ratio=self._min_ratio,
        )

    # ------------------------------------------------------------------ plan
    def _observed_price_signal(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Observed ``log(p / cost)`` relative to its base-year value, and the price level.

        ``cost`` is evaluated at the prices the board observes (its own instrument),
        with the mean observed price as the wage/numeraire, so the signal is exactly
        invariant to the price level: only *relative* prices move a plan.
        """
        p = np.maximum(np.asarray(obs.price, dtype=float), PRICE_EPS)
        pbar = float(np.mean(p)) if p.size else 1.0
        pbar = max(pbar, PRICE_EPS)
        cost = np.asarray(self.tech.A.T @ p, dtype=float) + np.asarray(self.tech.labor, dtype=float) * pbar
        cost = np.maximum(cost, PRICE_EPS)
        # p / cost relative to the base-year markup 1 / base_cost  (exactly 1 in the base year)
        markup = np.log(p * self._base_cost / cost)
        return np.clip(markup, -self._rel_clip, self._rel_clip), pbar

    def form_plan(self, obs: Observation, state: State) -> Plan:
        """Plan output at the base norm, tilted by the observed price/cost signal."""
        A = self.tech.A
        base = self.tech.base_output
        markup, _pbar = self._observed_price_signal(obs)

        # supply response: a sector whose observed price exceeds its observed input cost
        # relative to the base-year markup expands; one whose price lags contracts.
        targets = base * np.exp(self._supply_gain * markup)
        ceiling = np.minimum(self._capacity_ceiling * np.maximum(obs.capacity, 0.0),
                             self._max_ratio * base)
        targets = np.clip(np.minimum(targets, ceiling), self._min_ratio * base,
                          self._max_ratio * base)
        self.targets = targets

        final_targets = np.minimum(
            self.requested_final(obs),
            np.maximum(targets - A @ targets, 0.0)
            + np.maximum(np.asarray(obs.inventory, dtype=float), 0.0),
        )
        return Plan(
            targets=targets.copy(),
            final_targets=np.maximum(final_targets, 0.0),
            meta={
                "status": "ok",
                "institution": self.name,
                "price_gain": self._price_gain,
                "lag": obs.lag,
                "effective_lag": obs.meta.get("effective_lag", obs.lag + 1),
                "mean_markup_signal": float(np.mean(markup)),
                "target_total": float(targets.sum()),
            },
        )

    # -------------------------------------------------------------- allocate
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        """Willingness-to-pay ladder, with the price signal acting inside each tier.

        Claim *magnitudes* are the observed notional demand: the board does not ration
        household demand by price inside the period (see the module warning), it
        rations by willingness-to-pay tier.  The price-mediated part of the mechanism
        is the investment claim, which is deflated for goods whose observed price is
        high relative to the price level (dear capital goods are bought later).
        """
        alloc = self.build_allocation(
            targets=plan.targets,
            obs=obs,
            split_essentials=True,
            tier_inter=0.0,
            tier_final=1.0,
            tier_final_other=2.0,
            tier_gov=3.0,
            tier_invest=4.0,
            tier_stock=5.0,
        )
        # price-mediated claims: investment is the residual buyer of dear goods
        p = np.maximum(np.asarray(obs.price, dtype=float), PRICE_EPS)
        pbar = max(float(np.mean(p)) if p.size else 1.0, PRICE_EPS)
        dear = np.clip(np.log(p / pbar), -self._rel_clip, self._rel_clip)
        alloc.claims[alloc.role_mask("invest:")] *= np.exp(-self._inv_elasticity * dear)
        if self._claim_elasticity > 0.0:
            dear_level = np.clip(
                np.log(p), -np.log(1.0 / PRICE_EPS), np.log(1.0 / PRICE_EPS)
            )
            alloc.claims[alloc.role_mask("final:")] *= np.exp(-self._claim_elasticity * dear_level)
        alloc.meta.update(
            institution=self.name,
            price_gain=self._price_gain,
            claim_elasticity=self._claim_elasticity,
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
        """The Proposition-1 rule, verbatim.

        ``log p_{t+1} = log p_t + price_gain * s_obs`` with ``s_obs = Observation.shortage``.

        * ``state.price`` is ``p_t`` — the board's own administered price this period
          (an instrument, not an observation).
        * ``obs.shortage`` is the realised shortage rate of period ``t - 1 - lag``,
          i.e. the *observed* gap.  No extra lag is added and ``realised`` (this
          period's truth) is deliberately **not** consulted.
        * ``price_gain = 0`` returns the price vector untouched, so a run is exactly a
          fixed-administered-price economy.

        The fallback below is only reachable for observations constructed without a
        shortage vector (the engine always provides one); it keeps the mechanism
        well defined for unit tests that build ``Observation`` by hand.
        """
        if self._price_gain <= 0.0:
            return np.asarray(state.price, dtype=float).copy()
        if obs.shortage is not None:
            return self.admin_price_rule(obs, state, self._price_gain)
        # --- defensive fallback: no shortage vector in the observation -----------
        cap_obs = np.maximum(np.asarray(obs.capacity, dtype=float), 0.0)
        inv_obs = np.maximum(np.asarray(obs.inventory, dtype=float), 0.0)
        believed_supply = inv_obs + np.minimum(plan.targets, cap_obs)
        believed_claim = (
            self.requested_final(obs) + self.gov_claim(obs) + self.invest_claim(obs)
        )
        return self.excess_demand_rule(
            obs, state, self._price_gain, believed_claim - believed_supply
        )
