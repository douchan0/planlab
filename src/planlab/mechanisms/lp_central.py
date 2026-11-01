"""The omniscient central planner (Kantorovich / central-LP benchmark).

This mechanism is deliberately *unrealistic*: it is the information-rich upper
bound against which the informational cost of the other institutions is measured.
Each period it solves

    maximise   sum_i w_i * y_i  -  mu * sum_i x_i
    subject to A x + y <= x          (material balance, goods i)
               0 <= x <= capacity_hat
               0 <= y <= demand_hat

using the *observed* (i.e. delayed and noisy) capacity and demand.  It needs the
full current state of the economy to be reported, which is precisely the
assumption that Hayek and the socialist-calculation debate attacked; making the
information lag a treatment turns that argument into a measurable prediction.

The dual variables of the material-balance constraints are the **shadow prices** of
the plan, and are used as the mechanism's price signal (Kantorovich's
``resolving multipliers``).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import linprog

from ..core.state import Allocation, Observation, Plan, Realisation, State, Technology
from .base import PRICE_EPS, CoordinationMechanism

__all__ = ["CentralLP"]


class CentralLP(CoordinationMechanism):
    """Omniscient central LP (Kantorovich benchmark) with shadow prices as signals.

    Historical referent: Kantorovich's optimal plan and the "resolving multipliers"
    of the 1939 monograph; the information-rich upper bound of the calculation
    debate.

    Experimental treatments
    -----------------------
    The mechanism has no behavioural parameter: it is the benchmark that needs the
    full current state. ``output_penalty`` only breaks solver degeneracy.
    """

    name = "lp_central"

    default_params: Mapping[str, Any] = {
        # small penalty on gross output to break degeneracy (1e-3 << welfare weights)
        "output_penalty": 1e-3,
        # use the LP duals as the price signal
        "use_shadow_prices": 1.0,
        # cap on the planned output as a multiple of observed capacity
        "capacity_slack": 1.0,
        "tier_all_equal": 1.0,
    }

    def __init__(self, technology: Technology, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(technology, params)
        self._shadow = np.ones(self.n)
        self._last_status = "not_solved"

    # ------------------------------------------------------------------ plan
    def form_plan(self, obs: Observation, state: State) -> Plan:
        n = self.n
        A = self.tech.A
        w = np.asarray(self.tech.consumption_weights, dtype=float)
        mu = float(self.param("output_penalty", 1e-3))

        cap = np.maximum(np.asarray(obs.capacity, dtype=float), 0.0) * float(
            self.param("capacity_slack", 1.0)
        )
        # the plan must provision for *all* final uses: household, government and
        # investment.  Omitting government and investment would systematically
        # under-size the plan and manufacture a shortage that is an artefact.
        demand = self.requested_final(obs) + self.gov_claim(obs) + self.invest_claim(obs)

        # variables z = [x (n), y (n)]; minimise c.z
        c = np.concatenate([mu * np.ones(n), -w])
        # material balance rows: A[i,:] x - x_i + y_i <= 0
        A_ub = np.zeros((n, 2 * n))
        A_ub[:, :n] = A - np.eye(n)
        A_ub[:, n:] = np.eye(n)
        b_ub = np.zeros(n)
        bounds = [(0.0, float(cap[i])) for i in range(n)] + [
            (0.0, float(demand[i])) for i in range(n)
        ]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:  # pragma: no cover - numerical failure path
            self._last_status = f"failed: {res.message}"
            targets = np.minimum(self.base_targets(obs), cap)
            return Plan(targets=targets, final_targets=np.zeros(n),
                        meta={"status": self._last_status})

        x = np.maximum(res.x[:n], 0.0)
        y = np.maximum(res.x[n:], 0.0)
        self._last_status = "optimal"
        self._shadow = self._extract_shadow_prices(res, n)
        self.targets = x.copy()
        return Plan(
            targets=x,
            final_targets=np.minimum(y, demand),
            meta={
                "status": "ok",
                "objective": float(-res.fun),
                "solver": "highs",
                "lag": obs.lag,
            },
        )

    def base_targets(self, obs: Observation) -> np.ndarray:
        return np.asarray(self.tech.base_output, dtype=float).copy()

    # -------------------------------------------------------------- allocate
    def allocate(
        self, obs: Observation, state: State, plan: Plan, supply: np.ndarray
    ) -> Allocation:
        # the planner's priority ordering is fully encoded in the objective, so the
        # physical rationing is pro-rata: no claimant is privileged.
        return self.build_allocation(
            targets=plan.targets,
            obs=obs,
            split_essentials=False,
            tier_inter=0.0,
            tier_final=1.0,
            tier_final_other=2.0,
            tier_gov=3.0,
            tier_invest=4.0,
            tier_stock=5.0,
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
        if float(self.param("use_shadow_prices", 1.0)) <= 0.5:
            return np.asarray(state.price, dtype=float).copy()
        return self._shadow.copy()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _extract_shadow_prices(res: Any, n: int) -> np.ndarray:
        """Normalise the LP duals into a price vector with mean 1."""
        try:
            marg = np.asarray(res.ineqlin.marginals, dtype=float)[:n]
        except Exception:  # pragma: no cover - solver without marginals
            return np.ones(n)
        raw = np.maximum(-marg, 0.0)
        if not np.all(np.isfinite(raw)) or raw.sum() <= 1e-12:
            return np.ones(n)
        p = raw + PRICE_EPS
        return p / float(np.mean(p))
