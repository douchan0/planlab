"""Gosplan-style material-balance planning.

The institution is the classical one: for each good, compare *resources* (output
plus stock) with *needs* (intermediate requirements plus final claims), and adjust
the plan until the two sides are consistent.  Two features matter for the paper:

1. **Proportional target revision under delayed feedback.**  Each period the plan
   revises output targets in proportion to the shortage it has *already observed*:
   ``T_t = T_{t-1} * (1 + eta * s_{t-k})``.  This is exactly the feedback loop whose
   stability boundary is derived in Proposition 1; ``revision_gain`` is the
   experimental knob.
2. **Balancing rounds as planning effort.**  ``balance_rounds`` is the number of
   internal consistency passes the planning apparatus performs before issuing the
   plan.  With zero rounds the plan is a naive revision; with many it is close to
   being resource-feasible.  The count is a first-class object of study: how much
   iteration does a material-balance system need per period to stay coordinated?

   **The balancing step is one-sided.**  ``feasible_output`` maximises subject to
   ``0 <= x <= ub`` and therefore returns ``balanced <= ub``, so a round can only move
   a target *down*: ``T_t <= T_{t-1} * (1 + eta * s_{t-k})`` elementwise.  The
   proportional revision, by contrast, can push targets up.  Together they are
   asymmetric: under measurement noise the revision answers the noise by raising
   targets, the balancing office can only cut them, and the plan ratchets downwards
   with nothing to undo it.  Either feature alone is harmless — on the 6-sector
   fixture, ``revision_gain = 1`` with zero rounds and ``revision_gain = 0`` with three
   rounds both keep ``x / base`` at 1.0000, while the two together give 0.8854.  This
   is the interaction the paper's H1' rests on; it is checked, not merely argued, in
   ``tests/test_mech_material_balance.py``.

Historically, Gosplan's *material balances* did exactly this for key products, with
prices carrying little allocative weight (so by default prices are administrative
and revised slowly, if at all).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.feasibility import feasible_output
from ..core.state import Allocation, Observation, Plan, Realisation, State, Technology
from .base import CoordinationMechanism

__all__ = ["MaterialBalance"]


class MaterialBalance(CoordinationMechanism):
    """Gosplan-style material balance: resource-need balancing plus proportional revision.

    Historical referent: Soviet material balances (Gosplan), where key products were
    balanced commodity by commodity and output targets were renegotiated downwards
    when the balances did not close.

    Experimental treatments
    -----------------------
    ``revision_gain``
        How hard the plan moves towards closing an observed shortage. This is the
        gain of the delay-difference system analysed in the paper; in a
        capacity-constrained input-output economy it is a *positive* feedback loop
        (raising targets raises intermediate demand), so it is the object of H1'.
    ``balance_rounds``
        Internal consistency passes before the plan is issued: zero means a taut plan
        is issued and reality finds out, many means the plan is close to
        resource-feasible. A measure of planning effort.
    ``ladder``
        ``production_first`` (default) or ``consumption_first`` priority order.
    """

    name = "material_balance"

    default_params: Mapping[str, Any] = {
        # feedback gain of the target revision (eta in Prop. 1)
        "revision_gain": 0.30,
        # cap on the multiplicative revision per period, so a shock cannot explode the plan
        "revision_clip": 0.50,
        # internal consistency passes (planning effort)
        "balance_rounds": 3,
        # administered price revision gain (0 = fixed administrative prices)
        "price_gain": 0.05,
        "essentials_first": 1.0,
        "backlog_carry": 0.30,
        "stock_target_ratio": 0.10,
        # bounds on the planned target, relative to base-year output
        "max_target_ratio": 2.50,
        "min_target_ratio": 0.25,
        # "production_first" (default) or "consumption_first" priority ladder
        "ladder": "production_first",
    }

    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(technology, params)
        self._essentials_first = bool(float(self.param("essentials_first", 1.0)) > 0.5)

    def _ladder(self) -> tuple[float, float, float, float, float, float]:
        """Tier vector: (intermediates, essential final, other final, gov, invest, stock)."""
        if str(self.param("ladder", "production_first")) == "consumption_first":
            return (1.0, 0.0, 2.0, 3.0, 4.0, 5.0)
        return (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    # ------------------------------------------------------------------ plan
    def form_plan(self, obs: Observation, state: State) -> Plan:
        n = self.n
        eta = float(self.param("revision_gain", 0.3))
        clip = float(self.param("revision_clip", 0.5))
        rounds = int(self.param("balance_rounds", 3))
        A = self.tech.A
        AT = A.T
        mask = AT > 0.0
        safe_AT = np.where(mask, AT, 1.0)

        cap_est = np.maximum(np.asarray(obs.capacity, dtype=float), 0.0)
        inv_est = np.maximum(np.asarray(obs.inventory, dtype=float), 0.0)
        shortage = self.observed_shortage(obs)

        # --- 1. proportional revision of the carried-over plan ---------------
        step = np.clip(eta * shortage, -clip, clip)
        targets = self.targets * (1.0 + step)
        targets = np.minimum(targets, float(self.param("max_target_ratio", 2.5))
                             * self.tech.base_output)
        targets = np.maximum(targets, float(self.param("min_target_ratio", 0.25))
                             * self.tech.base_output)
        targets = np.minimum(targets, np.maximum(cap_est, 0.0) * 1.5 + 1e-12)

        # --- 2. material balancing: make the plan resource-feasible ----------
        # The balancing office solves the same *balance* the engine will enforce, but
        # on its (delayed, noisy) picture of inventories and capacity.  Each round it
        # moves halfway towards the balanced plan, which is what a real planning
        # apparatus does: the plan is negotiated down in stages, not all at once.
        # ``balance_rounds = 0`` therefore means "issue a naive plan and let reality
        # find out" -- the classic taut plan, with low fulfilment.
        n = self.n
        A = self.tech.A
        essential_demand = np.where(self.tech.essentials > 0.5, self.requested_final(obs), 0.0)
        reserve = essential_demand if self._ladder()[1] < self._ladder()[0] else np.zeros(n)
        detail = {"rounds": float(rounds), "fill_min": 1.0}
        for _ in range(max(0, rounds)):
            balanced, _status = feasible_output(
                A, targets, inv_est, np.minimum(reserve, inv_est),
                self.tech.labor, float(obs.labour or 0.0),
            )
            detail["fill_min"] = float(
                min(detail["fill_min"], float(np.min(balanced / np.maximum(targets, 1e-12))))
            ) if n else detail["fill_min"]
            targets = 0.5 * (targets + balanced)
            if np.max(np.abs(targets - balanced)) <= 1e-6 * max(1.0, float(np.max(targets))):
                break

        self.targets = targets
        final_targets = np.minimum(self.requested_final(obs), np.maximum(A @ targets, 0.0) + inv_est)
        return Plan(
            targets=targets.copy(),
            final_targets=np.maximum(final_targets, 0.0),
            meta={
                "status": "ok",
                "balance_rounds": rounds,
                "rerun_bound": detail["fill_min"],
                "eta": eta,
                "lag": obs.lag,
            },
        )

    # -------------------------------------------------------------- allocate
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        ti = self._ladder()
        return self.build_allocation(
            targets=plan.targets,
            obs=obs,
            split_essentials=self._essentials_first,
            tier_inter=ti[0],
            tier_final=ti[1],
            tier_final_other=ti[2],
            tier_gov=ti[3],
            tier_invest=ti[4],
            tier_stock=ti[5],
        )

    # ---------------------------------------------------------------- prices
    def update_prices(
        self,
        obs: Observation,
        state: State,
        plan: Plan,
        alloc: Allocation,
        realised: Realisation,
    ) -> np.ndarray:
        return self.admin_price_rule(obs, state, float(self.param("price_gain", 0.0)))
