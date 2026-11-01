"""Administrative quota rationing (the "quota system" institution).

**Institution.** Output is governed by *quotas*: a hard, administratively set ceiling
on each sector's gross output, fixed in advance and revised only when the approving
authority sees a change in (i) the needs it is obliged to cover or (ii) the capacity
it has approved.  Scarce goods are then handed out by administrative rank: production
inputs, then the designated priority sectors' inputs, then essential household
deliveries, then other household deliveries, then government, then investment.  The
allocating office does not trade; prices are administered and, by default, not revised
at all (that is the whole point: quotas, not prices, do the rationing).

**Historical referent.** The ministerial quota regime of the Soviet-type economy — the
supply orders (``fondy``/``naryady``) issued by the Gossnab ministries, and the
wartime rationing systems that gave the same function to consumer goods.  Its defining
feature, relative to material-balance planning, is that the plan is *rigid*: there is
no iterative balancing office reconciling resources and needs round by round, only a
ceiling with an administrative priority order behind it.

**Experimental treatments.** ``revision_interval`` is the lag-sensitivity treatment used
in block B2: a plan document fixed for several quarters is re-approved only on revision
dates, so between them it carries *both* a stale quota and the capacity it was approved
against (on the 6-sector fixture a 40% capacity shock costs 7.4% mean shortage at
``revision_interval = 1`` and 24.6% at ``revision_interval = 16``).  ``quota_speed`` is
the inertia of the revision itself, ``capacity_ceiling`` the quota's strictness (how far
a quota may exceed the approved capacity), and ``priority_sectors`` selects which
sectors' inputs are ranked first.  ``price_gain`` exists only so that an administered
price revision can be switched on as a contrast; it defaults to ``0.0`` (prices fixed).

**What differs from** :class:`planlab.mechanisms.material_balance.MaterialBalance`.
Material balance revises targets *in proportion to the observed shortage rate* and then
runs internal consistency passes (``balance_rounds``) that make the plan nearly
resource-feasible.  This mechanism never looks at the shortage rate: quotas chase the
*needs* they are supposed to cover (``A q + final claims``) and are then cut by the
administratively approved capacity.  The resulting plan is "taut" — it is not balanced
against available resources at all, and reality discovers the difference every period.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.state import Allocation, Observation, Plan, Realisation, State, Technology
from ._util import require_int, require_range, sector_mask
from .base import CoordinationMechanism

__all__ = ["RationingPriority"]

#: tier of the intermediate-input rows of priority sectors, and of the rest
_TIER_INTER_PRIORITY = 0.0
_TIER_INTER_OTHER = 0.5
_TIER_ESSENTIAL = 1.0
_TIER_FINAL_OTHER = 2.0
_TIER_GOV = 3.0
_TIER_INVEST = 4.0
_TIER_STOCK = 5.0


class RationingPriority(CoordinationMechanism):
    """Administrative quotas with a strict priority ladder and fixed prices."""

    name = "rationing_priority"

    default_params: Mapping[str, Any] = {
        #: speed at which a quota is revised towards observed needs (1.0 = jump)
        "quota_speed": 0.50,
        #: how often the quotas are revised, in periods (1 = every period).  A plan
        #: document fixed for several quarters sees information that is
        #: ``revision_interval - 1`` periods staler on top of the reporting lag, so this
        #: is the mechanism's lag-sensitivity treatment for block B2.
        "revision_interval": 1,
        #: hard ceiling on a quota, as a multiple of the *approved* (observed) capacity
        "capacity_ceiling": 1.00,
        #: sectors whose intermediate inputs are ranked ahead of the rest
        "priority_sectors": "essentials",
        #: share of the observed backlog re-submitted as this period's household claim
        #: (a queue; granted-but-not-demanded quantities end up in inventory)
        "backlog_carry": 0.30,
        #: desired stock level as a multiple of base output (residual claimant, tier 5)
        "stock_target_ratio": 0.15,
        #: bounds on the planned quota, relative to base-year output
        "max_target_ratio": 2.50,
        "min_target_ratio": 0.25,
        #: administered price revision gain (0.0 = prices fixed, the default)
        "price_gain": 0.00,
    }

    # ------------------------------------------------------------------ setup
    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(technology, params)
        self._quota_speed = require_range(
            "quota_speed", self.param("quota_speed", 0.5), 0.0, 1.0, lo_open=True
        )
        self._revision_interval = require_int(
            "revision_interval", self.param("revision_interval", 1), 1, 400
        )
        self._capacity_ceiling = require_range(
            "capacity_ceiling", self.param("capacity_ceiling", 1.0), 0.0, 5.0, lo_open=True
        )
        self._backlog_carry = require_range(
            "backlog_carry", self.param("backlog_carry", 0.3), 0.0, 1.0
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
        self._price_gain = require_range("price_gain", self.param("price_gain", 0.0), 0.0, 2.0)
        self._priority = sector_mask(self.tech, self.param("priority_sectors", "essentials"))
        #: the approved quota document and the capacity it was approved against
        #: (both are administrative documents, revised only on revision dates)
        self._quota = np.asarray(self.tech.base_output, dtype=float).copy()
        self._approved_ceiling = np.asarray(self.tech.base_output, dtype=float).copy()
        self._periods_seen = 0
        self._last_revision = 0
        # config is the public record of the mechanism; keep the validated values in it
        self.config.update(
            quota_speed=self._quota_speed,
            revision_interval=self._revision_interval,
            capacity_ceiling=self._capacity_ceiling,
            backlog_carry=self._backlog_carry,
            stock_target_ratio=self._stock_ratio,
            max_target_ratio=self._max_ratio,
            min_target_ratio=self._min_ratio,
            price_gain=self._price_gain,
        )

    # ------------------------------------------------------------------ plan
    def form_plan(self, obs: Observation, state: State) -> Plan:
        """Revise each quota towards the needs it must cover, then apply the ceiling.

        ``needs = A q_{t-1} + (observed final claims)``: the resources the quota
        commits the economy to deliver.  The revision is partial (``quota_speed``)
        because a quota is an administrative document, not a variable.  The quota is
        then cut to the *approved* capacity, which the authority sees with a reporting
        lag exactly like every other observation — the delay of the information regime
        is therefore a real constraint on quotas, not a cosmetic one.
        """
        n = self.n
        A = self.tech.A
        base = self.tech.base_output
        cap_obs = np.maximum(np.asarray(obs.capacity, dtype=float), 0.0)

        final_claim = (
            self.requested_final(obs)
            + self.gov_claim(obs)
            + self.invest_claim(obs)
            + self.stock_claim(obs)
        )
        self._periods_seen += 1
        revised = (
            self._periods_seen % self._revision_interval == 0
            or self._revision_interval == 1
        )
        if revised:
            # needs = the resources the quota commits the economy to deliver
            needs = A @ self._quota + final_claim
            self._quota = self._quota + self._quota_speed * (needs - self._quota)
            # the approved capacity is re-approved together with the quota, so a plan
            # document fixed for several quarters also carries a stale capacity
            self._approved_ceiling = self._capacity_ceiling * cap_obs
            self._last_revision = self._periods_seen
        quota_new = self._quota

        floor = self._min_ratio * base
        ceiling = np.maximum(np.minimum(self._approved_ceiling, self._max_ratio * base), floor)
        targets = np.clip(np.minimum(quota_new, ceiling), floor, self._max_ratio * base)

        self.targets = targets
        final_targets = np.minimum(
            self.requested_final(obs),
            np.maximum(targets - A @ targets, 0.0) + np.maximum(
                np.asarray(obs.inventory, dtype=float), 0.0
            ),
        )
        return Plan(
            targets=targets.copy(),
            final_targets=np.maximum(final_targets, 0.0),
            meta={
                "status": "ok",
                "institution": self.name,
                "quota_speed": self._quota_speed,
                "lag": obs.lag,
                "effective_lag": obs.meta.get("effective_lag", obs.lag + 1),
                #: fraction of the quota vector that was cut by the approved ceiling
                "ceiling_bite": float(np.mean(targets < quota_new - 1e-12)),
                "quota_total": float(targets.sum()),
                "revised": float(revised),
                "revision_interval": float(self._revision_interval),
                "periods_since_revision": float(self._periods_seen - self._last_revision),
            },
        )

    # -------------------------------------------------------------- allocate
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        """Strict administrative ladder; intermediate inputs of priority sectors first."""
        alloc = self.build_allocation(
            targets=plan.targets,
            obs=obs,
            split_essentials=True,
            tier_inter=_TIER_INTER_PRIORITY,
            tier_final=_TIER_ESSENTIAL,
            tier_final_other=_TIER_FINAL_OTHER,
            tier_gov=_TIER_GOV,
            tier_invest=_TIER_INVEST,
            tier_stock=_TIER_STOCK,
        )
        # split the intermediate block: priority sectors' inputs outrank the rest.
        alloc.tiers[: self.n] = np.where(
            self._priority, _TIER_INTER_PRIORITY, _TIER_INTER_OTHER
        )
        alloc.meta.update(
            institution=self.name,
            priority_sectors=[s for s, m in zip(self.tech.sectors, self._priority) if m],
            ladder={
                "inter:priority": _TIER_INTER_PRIORITY,
                "inter:other": _TIER_INTER_OTHER,
                "final:essential": _TIER_ESSENTIAL,
                "final:other": _TIER_FINAL_OTHER,
                "gov": _TIER_GOV,
                "invest": _TIER_INVEST,
                "stock": _TIER_STOCK,
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
        """Administered prices: fixed unless ``price_gain`` switches revision on.

        With the default ``price_gain = 0`` the price vector is carried over unchanged
        and the observation is not consulted at all, which is the institution's
        defining assumption.  ``realised`` is ignored: a quota office does not look at
        this period's deliveries when posting next period's prices.
        """
        return self.admin_price_rule(obs, state, self._price_gain)
