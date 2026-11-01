"""The simulation engine — the physics of the economy.

The engine owns *everything* quantitative except the three decisions an
institution makes (targets, claims, prices).  It enforces, every period:

1. the per-sector balance identity ``S0 + x + imports = Z.rows + C + G + I + S_end``;
2. feasibility of production given actually available intermediates, capacity and
   labour (the short-side fixed point);
3. non-negativity and finite values everywhere.

If any of these fail the engine raises, because a silent accounting leak would
invalidate every downstream result.  The period order is documented in
``docs/ARCHITECTURE.md`` §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from . import metrics as _metrics
from .feasibility import feasible_output
from .scenarios import Scenario, resolve_technology_spec
from .shocks import ShockPath
from .state import (
    Allocation,
    Observation,
    PeriodRecord,
    Plan,
    Realisation,
    State,
    Technology,
    TOL_ACCOUNTING,
)

__all__ = [
    "AccountingError",
    "balance_residual",
    "ration_by_priority",
    "Simulation",
    "SimulationResult",
    "run_simulation",
    "run_scenario",
    "build_observation",
]


def balance_residual(record: Any) -> float:
    """Relative violation of the per-sector balance identity in one period record.

    The identity is ``S0 + x = sum_j Z_ij + C + G + I + S_end + spoilage``.  Spoilage
    is the part of output that is neither used nor storable (see
    ``Scenario.max_stock_ratio``); omitting it from this check would make a glut look
    like an accounting leak.

    This is the *only* definition of the audit used in the repository: import it
    rather than re-deriving it, so that a change in the accounting rule cannot leave
    one test checking the old identity.
    """
    lhs = np.asarray(record.supply, dtype=float)
    rhs = (
        np.asarray(record.Z, dtype=float).sum(axis=1)
        + np.asarray(record.C, dtype=float)
        + np.asarray(record.G, dtype=float)
        + np.asarray(record.invest, dtype=float)
        + np.asarray(record.inventory_end, dtype=float)
    )
    sp = getattr(record, "spoilage", None)
    if sp is not None and np.size(sp) == np.size(lhs):
        rhs = rhs + np.asarray(sp, dtype=float)
    scale = max(1.0, float(np.max(np.abs(lhs))))
    return float(np.max(np.abs(lhs - rhs))) / scale


class AccountingError(RuntimeError):
    """Raised when the per-sector balance identity fails."""


# ---------------------------------------------------------------------------
# rationing
# ---------------------------------------------------------------------------
def ration_by_priority(
    supply: np.ndarray, claims: np.ndarray, tiers: np.ndarray
) -> np.ndarray:
    """Fill claims good-by-good in ascending tier order, pro-rata within a tier.

    Parameters
    ----------
    supply:
        ``(n,)`` available quantity of each good.
    claims:
        ``(m, n)`` desired quantity of each good by each claimant.
    tiers:
        ``(m,)`` priority of each claimant; lower numbers are served first.

    Returns
    -------
    ``(m, n)`` granted quantities, with ``grants.sum(axis=0) <= supply``.

    Notes
    -----
    This is the engine's *only* rationing primitive.  Institutions differ in the
    claims they submit and in the tiers they assign, never in the filling rule —
    which is what makes mechanism comparisons identified.
    """
    supply = np.asarray(supply, dtype=float)
    claims = np.asarray(claims, dtype=float)
    tiers = np.asarray(tiers, dtype=float)
    if claims.ndim != 2 or claims.shape[1] != supply.size:
        raise ValueError(f"claims must be (m,{supply.size}), got {claims.shape}")
    if tiers.shape != (claims.shape[0],):
        raise ValueError("tiers must have one entry per claimant row")
    grants = np.zeros_like(claims)
    remaining = np.maximum(supply, 0.0).copy()
    for tier in np.unique(tiers):
        rows = tiers == tier
        block = claims[rows, :]
        total = block.sum(axis=0)
        alloc = np.minimum(total, remaining)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(total > 0, alloc / np.maximum(total, 1e-300), 0.0)
        grants[rows, :] = block * scale
        remaining = np.maximum(remaining - alloc, 0.0)
    return grants


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------
def build_observation(
    *,
    t: int,
    lag: int,
    noise: float,
    rng: np.random.Generator,
    hist: Mapping[str, list],
    observe_capacity: bool,
    observe_inventory: bool,
    observe_backlog: bool,
    fallback_capacity: np.ndarray,
    fallback_inventory: np.ndarray,
    fallback_backlog: np.ndarray,
    fallback_labour: float,
    gov_demand: np.ndarray | None = None,
    invest_demand: np.ndarray | None = None,
) -> Observation:
    """Assemble the stale, noisy picture a coordinator is allowed to see.

    Timing convention
    -----------------
    At the start of period ``t`` the most recent *realised* outcomes a statistical
    office could have reported are those of period ``t-1``.  Hence

        ``observed_period = max(0, t - 1 - lag)``

    and the **effective total lag** is ``lag + 1`` periods.  ``lag = 0`` is therefore
    "plans are made on last quarter's returns", which is the historically honest
    best case, not clairvoyance.  All model-to-theory mappings in the paper use the
    effective lag ``m = lag + 1``.
    """
    src = max(0, t - 1 - int(lag))

    def _noisy(vec: np.ndarray) -> np.ndarray:
        if noise <= 0:
            return vec.copy()
        z = rng.standard_normal(vec.size)
        return vec * np.exp(noise * z - 0.5 * noise**2)

    demand = _noisy(np.asarray(hist["demand"][src], dtype=float))
    capacity = (
        _noisy(np.asarray(hist["capacity"][src], dtype=float))
        if observe_capacity
        else np.asarray(fallback_capacity, dtype=float).copy()
    )
    inventory = (
        _noisy(np.asarray(hist["inventory"][src], dtype=float))
        if observe_inventory
        else np.asarray(fallback_inventory, dtype=float).copy()
    )
    backlog = (
        _noisy(np.asarray(hist["backlog"][src], dtype=float))
        if observe_backlog
        else np.asarray(fallback_backlog, dtype=float).copy()
    )
    price = _noisy(np.asarray(hist["price"][src], dtype=float))
    shortage = np.asarray(hist["shortage"][src], dtype=float).copy()
    labour = float(hist["labour"][src]) * (
        1.0 if noise <= 0 else float(np.exp(noise * rng.standard_normal() - 0.5 * noise**2))
    )
    denom = np.maximum(inventory + capacity, 1e-300)
    return Observation(
        t=t,
        lag=int(lag),
        demand=np.maximum(demand, 0.0),
        capacity=np.maximum(capacity, 0.0),
        inventory=np.maximum(inventory, 0.0),
        price=np.maximum(price, 1e-9),
        backlog=np.maximum(backlog, 0.0),
        labour=max(labour, 0.0),
        stock_ratio=inventory / denom,
        shortage=np.maximum(shortage, 0.0),
        gov=None if gov_demand is None else np.asarray(gov_demand, dtype=float).copy(),
        invest=None if invest_demand is None else np.asarray(invest_demand, dtype=float).copy(),
        meta={"observed_period": src, "effective_lag": int(lag) + 1},
    )


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    scenario: Scenario
    technology: Technology
    seed: int
    records: list[PeriodRecord]
    run_metrics: dict[str, float]
    mechanism_name: str = ""
    meta: dict = field(default_factory=dict)

    # -- convenient views ---------------------------------------------------
    @property
    def periods(self) -> int:
        return len(self.records)

    def series(self, name: str) -> np.ndarray:
        return np.array([float(r.metrics.get(name, np.nan)) for r in self.records])

    def sector_panel(self) -> list[dict[str, Any]]:
        """Long-format sector panel (used for network figures)."""
        rows = []
        for r in self.records:
            for j, name in enumerate(self.technology.sectors):
                rows.append(
                    {
                        "t": r.t,
                        "sector": name,
                        "x": float(r.x[j]),
                        "demand": float(r.demand[j]),
                        "consumption": float(r.C[j]),
                        "unmet": float(r.unmet[j]),
                        "inventory": float(r.inventory_end[j]),
                        "price": float(r.price[j]),
                        "capacity": float(r.capacity[j]),
                        "plan_target": float(r.plan_targets[j]),
                    }
                )
        return rows

    def to_frame(self) -> Any:
        """Per-period metrics as a :class:`pandas.DataFrame`."""
        import pandas as pd

        rows = []
        for r in self.records:
            row = {"scenario_id": self.scenario.id, "mechanism": self.mechanism_name,
                   "seed": self.seed, "t": r.t}
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------
class Simulation:
    """One (scenario, mechanism, tech, seed) trajectory."""

    def __init__(
        self,
        technology: Technology,
        mechanism: Any,
        scenario: Scenario,
        seed: int = 0,
        *,
        max_cascade_iter: int = 400,
        cascade_tol: float = 1e-9,
        record_sectors: bool = False,
    ) -> None:
        self.tech = technology
        self.n = technology.n
        self.mechanism = mechanism
        self.scenario = scenario
        self.seed = int(seed)
        self.max_cascade_iter = int(max_cascade_iter)
        self.cascade_tol = float(cascade_tol)
        self.record_sectors = bool(record_sectors)
        self._last_status = "init"
        self.rng = np.random.default_rng(_seed_sequence(scenario.id, self.seed))
        self.shock_path = ShockPath(scenario.shocks, technology.sectors)

        sc = scenario
        self.tf = np.asarray(technology.base_final, dtype=float)          # total final demand
        self.gov_base = float(sc.gov_share) * self.tf
        self.invest_share_vec = np.asarray(technology.base_output, dtype=float)
        self.invest_share_vec = self.invest_share_vec / self.invest_share_vec.sum()
        self.invest_base_total = float(sc.depreciation) * float(
            np.dot(np.ones(self.n), technology.kappa * technology.base_output)
        )
        self.invest_base = self.invest_share_vec * self.invest_base_total
        self.hh_base = self.tf - self.gov_base - self.invest_base
        if np.any(self.hh_base < -1e-9 * max(1.0, float(self.tf.sum()))):
            raise ValueError(
                "government + investment demand exceed total final demand; "
                "reduce gov_share / depreciation"
            )
        self.hh_base = np.maximum(self.hh_base, 0.0)
        self.base_labour = float(np.dot(technology.labor, technology.base_output))
        self.ar_state = np.zeros(self.n)
        self.state = State.initial(technology, stock_ratio=sc.stock_ratio)

    # -- history for the observation ---------------------------------------
    def _initial_hist(self) -> dict[str, list]:
        st = self.state
        return {
            "demand": [self.hh_base.copy()],
            "capacity": [st.capacity.copy()],
            "inventory": [st.inventory.copy()],
            "price": [st.price.copy()],
            "backlog": [st.backlog.copy()],
            "labour": [self.base_labour * self.scenario.labor_supply_factor],
            "shortage": [np.zeros(self.n)],
        }

    # -- the loop -----------------------------------------------------------
    def run(self) -> SimulationResult:
        sc = self.scenario
        tech = self.tech
        n = self.n
        state = self.state
        hist = self._initial_hist()
        records: list[PeriodRecord] = []
        price_prev = state.price.copy()
        x_prev = state.x.copy()

        for t in range(1, sc.periods + 1):
            state.t = t

            # ---- 1. exogenous environment ---------------------------------
            cap_factor = self.shock_path.capacity_factor(t)
            capacity = np.maximum(tech.capacity_from_capital(state.capital) * cap_factor, 0.0)
            state.capacity = capacity
            labour_supply = (
                self.base_labour * sc.labor_supply_factor * self.shock_path.labour_factor(t)
            )

            trend = (1.0 + sc.demand.growth) ** (t - 1)
            demand_shock = self.shock_path.demand_factor(t)
            if sc.demand.sigma > 0 or sc.demand.ar1 != 0.0:
                eps = self.rng.standard_normal(n)
                self.ar_state = sc.demand.ar1 * self.ar_state + sc.demand.sigma * eps
            else:
                self.ar_state = np.zeros(n)
            ar_mult = np.exp(self.ar_state)
            price_response = np.power(
                np.maximum(state.price, 1e-9), -float(sc.demand.price_elasticity)
            )
            wage_bill = float(np.dot(tech.labor, state.x))
            income_ratio = wage_bill / max(self.base_labour, 1e-12)
            income_response = np.clip(income_ratio, 0.2, 5.0) ** float(sc.demand.income_elasticity)
            demand = np.maximum(
                self.hh_base * trend * demand_shock * ar_mult * price_response * income_response,
                0.0,
            )
            demand = demand + float(sc.household.backlog_persistence) * state.backlog
            gov_demand = self.gov_base * trend
            invest_demand = self.invest_base * trend

            # ---- 2. observation (stale + noisy) ---------------------------
            obs = build_observation(
                t=t,
                lag=int(sc.info.lag),
                noise=float(sc.info.noise),
                rng=self.rng,
                hist=hist,
                observe_capacity=sc.info.observe_capacity,
                observe_inventory=sc.info.observe_inventory,
                observe_backlog=sc.info.observe_backlog,
                fallback_capacity=np.asarray(tech.base_output, dtype=float),
                fallback_inventory=np.zeros(n),
                fallback_backlog=np.zeros(n),
                fallback_labour=self.base_labour * sc.labor_supply_factor,
                gov_demand=gov_demand,
                invest_demand=invest_demand,
            )

            # ---- 3. plan --------------------------------------------------
            plan = self.mechanism.form_plan(obs, state)
            _validate_plan(plan, n)
            supply_nominal = state.inventory + np.minimum(plan.targets, capacity)

            # ---- 4. claims ------------------------------------------------
            alloc = self.mechanism.allocate(obs, state, plan, supply_nominal)
            _validate_allocation(alloc, n)

            # ---- 5. feasibility fixed point + accounting ------------------
            (
                x, Z, C, G, Inv, S_end, unmet, iterations, converged, spoilage,
            ) = self._reconcile(
                inventory=state.inventory,
                plan=plan,
                capacity=capacity,
                labour_supply=labour_supply,
                alloc=alloc,
                demand=demand,
                gov_demand=gov_demand,
                invest_demand=invest_demand,
            )

            supply_realised = state.inventory + x
            self._audit(t, inventory=state.inventory, supply=supply_realised, Z=Z,
                        C=C, G=G, Invest=Inv, S_end=S_end, spoilage=spoilage)

            # ---- 6. prices for next period -------------------------------
            realisation = Realisation(
                x=x, Z=Z, C=C, G=G, invest=Inv, inventory_end=S_end,
                demand=demand, unmet=unmet, supply=supply_realised, capacity=capacity,
            )
            price_next = np.asarray(
                self.mechanism.update_prices(obs, state, plan, alloc, realisation),
                dtype=float,
            ).reshape(-1)
            if price_next.shape != (n,):
                raise ValueError(f"update_prices returned shape {price_next.shape}, expected ({n},)")
            lo, hi = sc.price_bounds
            price_next = np.clip(np.nan_to_num(price_next, nan=1.0, posinf=hi, neginf=lo), lo, hi)

            # ---- 7. metrics ----------------------------------------------
            pm = _metrics.period_metrics(
                tech=tech, x=x, Z=Z, C=C, G=G, invest=Inv, inventory_end=S_end,
                demand=demand, unmet=unmet, price=price_next, capacity=capacity,
                plan_targets=plan.targets, iterations=iterations,
                prev_x=x_prev if t > 1 else None,
                prev_price=price_prev if t > 1 else None,
                spoilage=spoilage,
                supply=supply_realised,
            )
            pm["converged"] = float(converged)

            records.append(
                PeriodRecord(
                    t=t, x=x.copy(), Z=Z.copy(), C=C.copy(), G=G.copy(),
                    invest=Inv.copy(), inventory_end=S_end.copy(),
                    demand=demand.copy(), unmet=unmet.copy(),
                    price=price_next.copy(), capacity=capacity.copy(),
                    plan_targets=plan.targets.copy(),
                    plan_final_targets=plan.final_targets.copy(),
                    supply=supply_realised.copy(),
                    spoilage=spoilage.copy(),
                    plan_status=str(plan.meta.get("status", "")),
                    iterations=int(iterations),
                    metrics=pm,
                )
            )

            # ---- 8. state update -----------------------------------------
            x_prev = x.copy()
            price_prev = price_next.copy()
            state.x = x.copy()
            state.inventory = S_end.copy()
            # Investment is distributed across sectors in proportion to the existing
            # capital stock, so that replacement investment exactly offsets
            # depreciation in the base year (a fixed composition would make capacity
            # drift purely as an artefact of differing capital-output ratios).
            k_total = float(state.capital.sum())
            share = (
                state.capital / k_total if k_total > 1e-12
                else np.full(self.n, 1.0 / self.n)
            )
            state.capital = (1.0 - sc.depreciation) * state.capital + Inv.sum() * share
            state.capital = np.maximum(state.capital, 0.0)
            state.price = price_next.copy()
            state.backlog = unmet.copy()
            state.gov_demand = gov_demand.copy()

            hist["demand"].append(demand.copy())
            hist["capacity"].append(capacity.copy())
            hist["inventory"].append(state.inventory.copy())
            hist["price"].append(price_next.copy())
            hist["backlog"].append(unmet.copy())
            hist["labour"].append(float(labour_supply))
            hist["shortage"].append(np.where(demand > 0, unmet / np.maximum(demand, 1e-300), 0.0))

        run_metrics = _metrics.run_metrics(
            records,
            scenario=sc,
            tech=tech,
            shock_intensity=self.shock_path.peak_capacity_loss(self.tech.base_output),
        )
        run_metrics["converged_all"] = float(
            all(r.metrics.get("converged", 1.0) > 0 for r in records)
        )
        return SimulationResult(
            scenario=sc,
            technology=tech,
            seed=self.seed,
            records=records,
            run_metrics=run_metrics,
            mechanism_name=getattr(self.mechanism, "name", type(self.mechanism).__name__),
            meta={"mechanism": getattr(self.mechanism, "config", {})},
        )

    # -- feasibility + accounting ------------------------------------------
    def _reconcile(
        self,
        *,
        inventory: np.ndarray,
        plan: Plan,
        capacity: np.ndarray,
        labour_supply: float,
        alloc: Allocation,
        demand: np.ndarray,
        gov_demand: np.ndarray,
        invest_demand: np.ndarray,
    ):
        """Determine the feasible production vector and allocate the residual.

        Step 1 — *feasible output (the short side).*  Given capacity, inventories and
        the claims that outrank production inputs, the largest feasible gross output
        vector is the solution of

            maximise   sum_j x_j
            subject to A x - x <= r - reserve      (goods balance)
                       0 <= x <= min(targets, capacity)
                       labour . x <= L

        This is the standard non-substitution / short-side allocation (Dreze 1975;
        Benassy 1982): no sector can use more of a good than exists, and output is
        the largest vector consistent with that.  Using a programme instead of a
        proportional cascade matters: a cascade that shrinks every sector
        proportionally cannot *rebalance* the production mix, and in a
        capacity-constrained input-output economy it converges to zero instead of to
        the rebalanced plan.

        Step 2 — *rationing the residual.*  Intermediates are served at their
        technical requirement ``A x``; the remaining supply is distributed across the
        final claimants by the mechanism's priority tiers.
        """
        tech = self.tech
        n = self.n
        A = tech.A
        tiers = np.asarray(alloc.tiers, dtype=float)
        claims = np.array(alloc.claims, dtype=float, copy=True)
        ub = np.minimum(np.maximum(plan.targets, 0.0), np.maximum(capacity, 0.0))
        inv = np.maximum(np.asarray(inventory, dtype=float), 0.0)

        # claims that outrank intermediate inputs (e.g. essential consumption under a
        # "consumption first" ladder) are reserved before production is planned.
        tier_inter = float(np.min(tiers[:n])) if n else 0.0
        below = tiers < tier_inter - 1e-12
        if below.any():
            reserve = np.minimum(claims[below].sum(axis=0), inv)
        else:
            reserve = np.zeros(n)

        # Iterate the (programme -> ration) map to a consistent point.  The programme
        # guarantees the *goods balance*, but the tier structure may still hand a
        # higher-priority final claimant more than the programme reserved for it, which
        # would starve intermediates.  Shrinking the target vector monotonically until
        # intermediates are fully served restores consistency; the loop terminates
        # because x only decreases and is bounded below by zero.  On exit, ``supply``
        # and ``grants`` always correspond to the final ``x``.
        x = ub.copy()
        status = "init"
        n_passes = 0
        for _pass in range(1, self.max_cascade_iter + 1):
            n_passes = _pass
            x_feas, status = self._feasible_output(x, inv, reserve, labour_supply)
            supply = inv + x_feas
            claims[:n, :] = (A * x_feas.reshape(1, -1)).T
            grants = ration_by_priority(supply, claims, tiers)
            need_inter = A @ x_feas
            deficit = need_inter - grants[:n].sum(axis=0)
            if float(np.max(deficit)) <= self.cascade_tol * max(1.0, float(np.max(supply))):
                x = x_feas
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(need_inter > 0, grants[:n].sum(axis=0) / np.maximum(need_inter, 1e-300), 1.0)
            x_new = x_feas * float(np.clip(np.min(ratio), 0.0, 0.999))
            if float(np.max(x_new)) <= 1e-12:
                x = np.zeros(n)
                supply = inv.copy()
                claims[:n, :] = 0.0
                grants = ration_by_priority(supply, claims, tiers)
                break
            x = x_new

        parts = alloc.split_grants(grants)
        Z = A * x.reshape(1, -1)                  # Z[i, j] = A_ij * x_j
        C = np.minimum(parts["final"], demand)
        G = np.minimum(parts["gov"], gov_demand)
        Inv = np.minimum(parts["invest"], invest_demand)
        used = Z.sum(axis=1) + C + G + Inv
        residual = np.maximum(supply - used, 0.0)
        # Storage is finite.  Output that is neither used nor storable is spoilage: a
        # real phenomenon (unsalable output in a shortage economy) that must be
        # recorded rather than silently accumulated, because an unbounded stock would
        # let a compositional mismatch hide inside the inventory series forever.
        stock_cap = float(self.scenario.max_stock_ratio) * np.asarray(
            self.tech.base_output, dtype=float
        )
        S_end = np.minimum(residual, np.maximum(stock_cap - inv, 0.0))
        spoilage = residual - S_end
        unmet = np.maximum(demand - C, 0.0)
        self._last_status = status
        return (x, Z, C, G, Inv, S_end, unmet, n_passes,
                status in ("optimal", "infeasible-ub"), spoilage)

    # -- feasibility sub-problem -------------------------------------------
    def _feasible_output(self, ub, inventory, reserve, labour_supply):
        """Delegate to :func:`planlab.core.feasibility.feasible_output`."""
        return feasible_output(
            self.tech.A, ub, inventory, reserve,
            self.tech.labor, float(labour_supply or 0.0),
        )

    def _audit(self, t: int, *, inventory, supply, Z, C, G, Invest, S_end,
               spoilage=None) -> None:
        lhs = supply
        rhs = Z.sum(axis=1) + C + G + Invest + S_end
        if spoilage is not None:
            rhs = rhs + spoilage
        scale = max(1.0, float(np.max(np.abs(supply))))
        err = float(np.max(np.abs(lhs - rhs))) / scale
        if err > TOL_ACCOUNTING:
            raise AccountingError(
                f"period {t}: balance identity violated by {err:.3e} (relative); "
                "this indicates a bug in the engine or in a mechanism's claims"
            )
        if np.any(~np.isfinite(supply)) or np.any(lhs < -1e-9 * scale):
            raise AccountingError(f"period {t}: negative or non-finite supply")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _validate_plan(plan: Plan, n: int) -> None:
    if not isinstance(plan, Plan):
        raise TypeError("form_plan must return a Plan")
    if plan.targets.shape != (n,):
        raise ValueError(f"plan.targets must be ({n},), got {plan.targets.shape}")
    if plan.final_targets.shape != (n,):
        raise ValueError(f"plan.final_targets must be ({n},), got {plan.final_targets.shape}")


def _validate_allocation(alloc: Allocation, n: int) -> None:
    if not isinstance(alloc, Allocation):
        raise TypeError("allocate must return an Allocation")
    if alloc.claims.shape[1] != n or alloc.claims.shape[0] < n:
        raise ValueError(
            f"allocation.claims must be (m>={n}, {n}), got {alloc.claims.shape}"
        )


def _seed_sequence(scenario_id: str, seed: int) -> int:
    """Deterministic 63-bit seed from (scenario, replicate) for common random numbers."""
    return _stable_hash(scenario_id, int(seed)) % (2**63 - 1)


def _stable_hash(scenario_id: str, seed: int) -> int:
    import hashlib

    payload = f"{scenario_id}|{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def run_simulation(
    technology: Technology,
    mechanism: Any,
    scenario: Scenario,
    seed: int = 0,
    **kwargs: Any,
) -> SimulationResult:
    return Simulation(technology, mechanism, scenario, seed=seed, **kwargs).run()


def run_scenario(
    scenario: Scenario,
    *,
    technology: Technology | None = None,
    mechanism: Any | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> SimulationResult:
    """Convenience wrapper: resolve substrate and mechanism from the scenario."""
    from planlab.mechanisms import get_mechanism

    tech = technology if technology is not None else resolve_technology_spec(scenario.technology)
    mech = (
        mechanism
        if mechanism is not None
        else get_mechanism(scenario.mechanism, tech, scenario.mechanism_params)
    )
    if seed is None:
        seed = int(scenario.seeds[0]) if scenario.seeds else 0
    return run_simulation(tech, mech, scenario, seed=seed, **kwargs)
