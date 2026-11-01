"""Check 2 — an independent one-period implementation, compared with the engine.

I implement the period from the *written* specification only:

* ``docs/ARCHITECTURE.md`` §5 (the ten steps of a period, and the short-side
  programme) and the "Step 6 detail" box;
* the docstring of ``planlab.core.feasibility.feasible_output`` (the exact
  programme: ``max 1'x s.t. Ax - x <= r - reserve, 0 <= x <= ub, labor.x <= L``);
* the docstring of ``Simulation._reconcile`` (reserve = claims that outrank
  intermediate inputs; intermediates are served at their technical requirement;
  the residual is rationed across final claimants by priority tier).

I do NOT import ``planlab.engine``, ``planlab.core.feasibility`` or
``planlab.core.metrics``.  The only thing taken from the project is the *input*
of a period (technology, targets, claims, tiers, inventory, demand, capacity,
labour) and the *output arrays* it recorded, captured by wrapping
``Simulation._reconcile`` at run time.

My implementation
-----------------
1. ``ub = min(targets, capacity)``, ``r = inventory``, ``reserve`` = the sum of
   claims of any row whose tier is strictly below the intermediate tier, clipped
   to ``r``.
2. Solve the goods-balance programme with a from-scratch formulation:
   ``min -1'x  s.t. (A - I)x <= r - reserve, labor'x <= L, 0 <= x <= ub``.
3. Ration: good by good, ascending tier, pro-rata inside a tier (my own
   implementation of the rule in the docstring).
4. If intermediates are not fully served, shrink the target vector by the
   served-requirement ratio and repeat (the documented consistency iteration).
5. Account: ``Z = A diag(x)``, ``C = min(final grants, demand)``,
   ``G = min(gov grants, gov_demand)``, ``I = min(invest grants, invest_demand)``,
   ``S_end = supply - (Z 1 + C + G + I)``; ``unmet = max(demand - C, 0)``.

Regimes (the Lead's request: calm is too easy to be informative)
--------------------------------------------------------------------------
Every captured period is classified from the *inputs* by my own code:

* ``calm``      — the target vector is feasible as is, and no claimant is capped;
* ``capacity``  — the target vector is infeasible, so the programme binds
                  (some sector produces below its target);
* ``tier``      — at least one good is rationed *decidedly*: some claimant gets
                  strictly less than it asked for while a lower-priority claimant
                  asks for that same good (this is where a handout/ration bug hides).

I engineer the scenarios so that all three classes are actually populated, and I
report the worst relative difference *per class*.  I also verify the ladder
invariant itself (a capped tier implies zero deliveries to every later tier), the
per-period accounting identity and ``Z = A diag(x)``, and certify that the
engine's ``x`` is the optimum of the documented programme by re-solving it with
two different HiGHS algorithms.

An honest limitation: because I implement the *published* algorithm, agreement
shows that the code does what the specification says; it does not show that the
specification is the right model.  Where the code and the specification disagree,
the disagreement is reported as a finding.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .common import CheckResult, run_guarded

REL_TOL = 1e-9


# ---------------------------------------------------------------------------
# my own LP
# ---------------------------------------------------------------------------
def solve_goods_balance(A, ub, r, reserve, labor, labour_supply, method="highs"):
    """``max 1'x s.t. (A-I)x <= r - reserve, labor'x <= L, 0 <= x <= ub``."""
    n = A.shape[0]
    A_ub = [A - np.eye(n)]
    b_ub = [np.asarray(r, dtype=float) - np.asarray(reserve, dtype=float)]
    if labor is not None and labour_supply > 0:
        A_ub.append(np.asarray(labor, dtype=float).reshape(1, -1))
        b_ub.append(np.array([float(labour_supply)]))
    res = linprog(
        -np.ones(n),
        A_ub=np.vstack(A_ub),
        b_ub=np.concatenate(b_ub),
        bounds=[(0.0, float(max(ub[i], 0.0))) for i in range(n)],
        method=method,
    )
    if not res.success:                                  # pragma: no cover
        raise RuntimeError(f"independent LP failed ({method}): {res.message}")
    return np.clip(res.x, 0.0, np.maximum(ub, 0.0)), float(-res.fun)


# ---------------------------------------------------------------------------
# my own tier rationing
# ---------------------------------------------------------------------------
def ration_by_tier(supply, claims, tiers):
    """Fill claims good-by-good, ascending tier, pro-rata within a tier."""
    supply = np.asarray(supply, dtype=float)
    claims = np.asarray(claims, dtype=float)
    tiers = np.asarray(tiers, dtype=float)
    grants = np.zeros_like(claims)
    left = np.maximum(supply, 0.0).copy()
    for tier in np.unique(tiers):
        rows = np.where(tiers == tier)[0]
        block = claims[rows, :]
        want = block.sum(axis=0)
        give = np.minimum(want, left)
        frac = np.where(want > 0.0, give / np.where(want > 0.0, want, 1.0), 0.0)
        grants[rows, :] = block * frac[None, :]
        left = np.maximum(left - give, 0.0)
    return grants


# ---------------------------------------------------------------------------
# my own one-period reconcile + accounting
# ---------------------------------------------------------------------------
def independent_period(*, A, labor, targets, capacity, inventory, labour_supply,
                       claims, tiers, labels, demand, gov_demand, invest_demand,
                       base_output=None, max_stock_ratio=None,
                       max_passes=400, tol=1e-9):
    n = A.shape[0]
    ub = np.minimum(np.maximum(targets, 0.0), np.maximum(capacity, 0.0))
    r = np.maximum(np.asarray(inventory, dtype=float), 0.0)
    claims = np.array(claims, dtype=float, copy=True)
    tiers = np.asarray(tiers, dtype=float)

    tier_inter = float(np.min(tiers[:n]))
    below = tiers < tier_inter - 1e-12
    reserve = np.minimum(claims[below].sum(axis=0), r) if below.any() else np.zeros(n)

    x = ub.copy()
    supply = r + x
    passes = 0
    for passes in range(1, max_passes + 1):
        x_feas, _obj = solve_goods_balance(A, x, r, reserve, labor, labour_supply)
        supply = r + x_feas
        claims[:n, :] = (A * x_feas.reshape(1, -1)).T
        grants = ration_by_tier(supply, claims, tiers)
        need = A @ x_feas
        got = grants[:n, :].sum(axis=0)
        deficit = need - got
        if float(np.max(deficit)) <= tol * max(1.0, float(np.max(supply))):
            x = x_feas
            break
        ratio = np.where(need > 0, got / np.where(need > 0, need, 1.0), 1.0)
        x = x_feas * float(np.clip(np.min(ratio), 0.0, 0.999))
        if float(np.max(x)) <= 1e-12:
            x = np.zeros(n)
            supply = r.copy()
            claims[:n, :] = 0.0
            grants = ration_by_tier(supply, claims, tiers)
            break

    parts = {"final": np.zeros(n), "gov": np.zeros(n), "invest": np.zeros(n),
             "stock": np.zeros(n)}
    for prefix in parts:
        mask = np.array([str(lab).startswith(prefix + ":") for lab in labels], dtype=bool)
        if mask.any():
            parts[prefix] = grants[mask].sum(axis=0)
    Z = A * x.reshape(1, -1)
    C = np.minimum(parts["final"], demand)
    G = np.minimum(parts["gov"], gov_demand)
    Inv = np.minimum(parts["invest"], invest_demand)
    used = Z.sum(axis=1) + C + G + Inv
    residual = np.maximum(supply - used, 0.0)
    if base_output is None or max_stock_ratio is None:
        S_end = residual
        spoilage = np.zeros(n)
    else:
        # bounded storage: output that is neither used nor storable is spoilage
        stock_cap = float(max_stock_ratio) * np.asarray(base_output, dtype=float)
        S_end = np.minimum(residual, np.maximum(stock_cap - r, 0.0))
        spoilage = residual - S_end
    unmet = np.maximum(demand - C, 0.0)
    return {
        "x": x, "Z": Z, "C": C, "G": G, "invest": Inv, "inventory_end": S_end,
        "unmet": unmet, "supply": supply, "iterations": passes, "spoilage": spoilage,
        "grants": grants, "claims_used": claims.copy(), "parts": parts,
        "reserve": reserve, "ub": ub,
    }


# ---------------------------------------------------------------------------
# capture the engine's inputs / outputs by wrapping _reconcile
# ---------------------------------------------------------------------------
def capture_runs(mechanisms, *, periods=24, seed=0):
    """Run the engine and record (inputs, outputs) of every reconcile call."""
    from planlab.core.engine import Simulation
    import planlab.core.scenarios as scenarios

    captured = []
    original = Simulation._reconcile

    def wrapper(self, **kwargs):
        snapshot = {
            "targets": np.array(kwargs["plan"].targets, dtype=float, copy=True),
            "capacity": np.array(kwargs["capacity"], dtype=float, copy=True),
            "inventory": np.array(kwargs["inventory"], dtype=float, copy=True),
            "labour_supply": float(kwargs["labour_supply"]),
            "claims": np.array(kwargs["alloc"].claims, dtype=float, copy=True),
            "tiers": np.array(kwargs["alloc"].tiers, dtype=float, copy=True),
            "labels": tuple(kwargs["alloc"].labels),
            "demand": np.array(kwargs["demand"], dtype=float, copy=True),
            "gov_demand": np.array(kwargs["gov_demand"], dtype=float, copy=True),
            "invest_demand": np.array(kwargs["invest_demand"], dtype=float, copy=True),
        }
        out = original(self, **kwargs)
        captured.append({"inputs": snapshot, "outputs": out,
                         "status": str(getattr(self, "_last_status", "")),
                         "A": np.array(self.tech.A, dtype=float, copy=True),
                         "base_output": np.array(self.tech.base_output, dtype=float, copy=True),
                         "max_stock_ratio": float(self.scenario.max_stock_ratio),
                         "labor": np.array(self.tech.labor, dtype=float, copy=True),
                         "mechanism": getattr(self.mechanism, "name", "?")})
        return out

    Simulation._reconcile = wrapper
    try:
        from planlab.core.engine import run_scenario
        from planlab.data import load_technology

        tech = load_technology("eurostat", 2020, 20)
        base = {
            "technology": {"source": "eurostat", "year": 2020, "n_sectors": 20},
            "periods": periods, "seeds": [seed], "burn_in": 2,
        }
        specs = []
        for mech, params, shocks in mechanisms:
            specs.append({
                **base, "id": f"cap_{mech}_{len(specs)}", "mechanism": mech,
                "mechanism_params": params, "shocks": shocks,
            })
        for spec in specs:
            sc = scenarios.scenario_from_dict(spec)
            run_scenario(sc, technology=tech)
    finally:
        Simulation._reconcile = original
    return captured


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------
def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b))) / scale


def classify_flags(inp, mine, A, labor):
    """Non-exclusive regime flags for a captured period, from inputs + reconstruction.

    ``calm``     nothing binds: the target vector is feasible and nobody is capped;
    ``capacity`` the goods-balance programme binds (output below the target vector);
    ``tier``     at least one good is rationed and a *strictly later* tier still asks
                 for that good — the case where a handout/rationing bug would hide.
    """
    n = A.shape[0]
    r = np.maximum(np.asarray(inp["inventory"], dtype=float), 0.0)
    ub = np.minimum(np.maximum(inp["targets"], 0.0), np.maximum(inp["capacity"], 0.0))
    reserve = mine["reserve"]
    scale = max(1.0, float(np.max(np.abs(r) + np.abs(ub))))
    feasible_ub = bool(
        np.all((A @ ub - ub) <= (r - reserve) + 1e-9 * scale)
        and (inp["labour_supply"] <= 0
             or float(np.dot(labor, ub)) <= inp["labour_supply"] * (1 + 1e-9))
    )
    capacity = (not feasible_ub) or bool(np.any(mine["x"] < ub - 1e-9 * np.maximum(1.0, ub)))

    claims = mine["claims_used"]
    grants = mine["grants"]
    tiers = np.asarray(inp["tiers"], dtype=float)
    gscale = np.maximum(1.0, np.max(np.abs(claims), axis=0))
    tier = False
    for i in range(n):
        capped_tiers = []
        for t in np.unique(tiers):
            rows = tiers == t
            claimed = float(claims[rows, i].sum())
            got = float(grants[rows, i].sum())
            if claimed - got > 1e-9 * gscale[i]:
                capped_tiers.append(t)
        for t in capped_tiers:
            later = tiers > t + 1e-12
            if later.any() and np.any(claims[later, i] > 0.0):
                tier = True
                break
        if tier:
            break
    return {"calm": not capacity and not tier, "capacity": capacity, "tier": tier}


def ladder_violations(grants, claims, tiers):
    """Count goods where a capped tier is followed by a served later tier.

    Tiers are compared as tiers, not as rows: within one tier every claimant is
    served pro-rata, so a partially served row inside a tier is correct behaviour.
    """
    tiers = np.asarray(tiers, dtype=float)
    n = claims.shape[1]
    gscale = np.maximum(1.0, np.max(np.abs(claims), axis=0))
    uniq = np.unique(tiers)
    bad = 0
    for i in range(n):
        capped = False
        for t in uniq:
            rows = tiers == t
            claimed = float(claims[rows, i].sum())
            got = float(grants[rows, i].sum())
            if capped and got > 1e-9 * gscale[i]:
                bad += 1
                break
            if claimed - got > 1e-9 * gscale[i]:
                capped = True
    return bad


def collapse_is_forced(inp, A, n):
    """Is x = 0 the only consistent point because tier-0 claims exhaust inventory?

    Returns (forced, n_blocked_goods).  A good is *blocked* when the claims of the
    rows strictly below the intermediate tier already exceed beginning inventory —
    then those rows take all of that good's supply, no intermediate can be served,
    and any positive output would starve its own inputs.
    """
    tiers = np.asarray(inp["tiers"], dtype=float)
    claims = np.asarray(inp["claims"], dtype=float)
    r = np.maximum(np.asarray(inp["inventory"], dtype=float), 0.0)
    tier_inter = float(np.min(tiers[:n]))
    below = tiers < tier_inter - 1e-12
    if not below.any():
        return False, 0
    essential = claims[below].sum(axis=0)
    blocked = (essential >= r) & (A.sum(axis=1) > 0)
    return bool(blocked.any()), int(blocked.sum())


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def _body(res: CheckResult) -> None:
    from planlab.mechanisms import available_mechanisms

    mechs = available_mechanisms()
    shock_cap = [{"type": "capacity", "sector": "c03", "start": 4, "duration": 3, "size": 0.55}]
    shock_cap_hard = [{"type": "capacity", "sector": "c05", "start": 4, "duration": 4, "size": 0.20}]
    shock_dem = [{"type": "demand", "sector": "all", "start": 3, "duration": 4, "size": 2.2}]

    specs = [("material_balance", {}, [])]                                   # calm
    specs.append(("lp_central", {}, []))                                     # calm, other claims
    for mech in mechs:
        specs.append((mech, {}, shock_cap))                                  # capacity / tier
    specs.append(("material_balance", {}, shock_cap_hard))                   # hard capacity
    specs.append(("material_balance", {}, shock_dem))                        # final-side rationing
    specs.append(("material_balance", {"ladder": "consumption_first"}, shock_cap_hard))
    specs.append(("material_balance", {"ladder": "consumption_first"}, []))
    captured = capture_runs(specs, periods=16)
    res.numbers["periods_captured"] = len(captured)
    res.check(len(captured) >= 40, f"captured only {len(captured)} periods from the engine")

    by_flag = {"calm": [], "capacity": [], "tier": []}
    worst_overall = 0.0
    worst_where = ""
    ladder_bad = 0
    statuses: dict[str, int] = {}
    n_collapse = 0
    collapses_forced = 0
    first_pass_optimality = 0
    exit_deficits = []

    for i, cap in enumerate(captured):
        inp, out = cap["inputs"], cap["outputs"]
        x, Z, C, G, Inv, S_end, unmet = (np.asarray(v, dtype=float) for v in out[:7])
        mine = independent_period(
            A=cap["A"], labor=cap["labor"], targets=inp["targets"],
            capacity=inp["capacity"], inventory=inp["inventory"],
            labour_supply=inp["labour_supply"], claims=inp["claims"],
            tiers=inp["tiers"], labels=inp["labels"], demand=inp["demand"],
            gov_demand=inp["gov_demand"], invest_demand=inp["invest_demand"],
            base_output=cap["base_output"], max_stock_ratio=cap["max_stock_ratio"],
        )
        spoilage = np.asarray(out[9], dtype=float) if len(out) > 9 else np.zeros_like(x)
        diffs = {
            "x": _rel(x, mine["x"]),
            "Z": _rel(Z, mine["Z"]),
            "C": _rel(C, mine["C"]),
            "G": _rel(G, mine["G"]),
            "invest": _rel(Inv, mine["invest"]),
            "inventory_end": _rel(S_end, mine["inventory_end"]),
            "unmet": _rel(unmet, mine["unmet"]),
            "spoilage": _rel(spoilage, mine["spoilage"]),
            "supply": _rel(inp["inventory"] + x, mine["supply"]),
        }
        flags = classify_flags(inp, mine, cap["A"], cap["labor"])
        tag = f"{cap['mechanism']} period {i} ({'/'.join(k for k, v in flags.items() if v) or 'calm'})"
        statuses[cap["status"]] = statuses.get(cap["status"], 0) + 1
        for name, d in diffs.items():
            res.check(d <= REL_TOL, f"2a: {tag}: {name} differs by {d:.2e} (tol {REL_TOL:.0e})")
        w = max(diffs.values())
        for k, on in flags.items():
            if on:
                by_flag[k].append(w)
        if w > worst_overall:
            worst_overall, worst_where = w, tag

        # ---- structural audit on the engine's own arrays -------------------
        supply = inp["inventory"] + x
        res.check(_rel(supply, Z.sum(axis=1) + C + G + Inv + S_end + spoilage) <= 1e-8,
                  f"2b: {tag}: balance identity")
        res.check(np.all(spoilage >= 0.0), f"2b: {tag}: negative spoilage")
        res.check(np.max(np.abs(Z - cap["A"] * x.reshape(1, -1)))
                  <= 1e-12 * max(1.0, np.max(np.abs(Z))), f"2b: {tag}: Z == A diag(x)")
        res.check(min(np.min(x), np.min(Z), np.min(C), np.min(G), np.min(Inv),
                      np.min(S_end), np.min(unmet)) >= 0.0, f"2b: {tag}: non-negative arrays")
        res.check(np.all(x <= inp["capacity"] * (1 + 1e-12) + 1e-12),
                  f"2b: {tag}: x <= capacity")
        res.check(np.all(x <= inp["targets"] * (1 + 1e-12) + 1e-12),
                  f"2b: {tag}: x <= plan targets")
        res.check(np.all(C <= inp["demand"] + 1e-9 * (1 + np.abs(inp["demand"]))),
                  f"2b: {tag}: C <= demand")
        res.check(np.all(Z.sum(axis=1) <= supply + 1e-8 * max(1.0, np.max(supply))),
                  f"2b: {tag}: intermediate use <= available supply")
        if inp["labour_supply"] > 0:
            used_lab = float(np.dot(cap["labor"], x))
            res.check(used_lab <= inp["labour_supply"] * (1 + 1e-9) + 1e-9,
                      f"2b: {tag}: labour {used_lab:.6f} <= supply {inp['labour_supply']:.6f}")

        # ---- ladder invariant on the reconstructed grant matrix -------------
        bad = ladder_violations(mine["grants"], mine["claims_used"], inp["tiers"])
        ladder_bad += bad
        res.check(bad == 0, f"2b: {tag}: {bad} good(s) where a capped tier is followed by a served one")

        # ---- documented cascade exit condition ------------------------------
        need = cap["A"] @ mine["x"]
        got = mine["grants"][: cap["A"].shape[0], :].sum(axis=0)
        deficit = float(np.max(need - got))
        exit_deficits.append(deficit / max(1.0, float(np.max(mine["supply"]))))
        res.check(deficit <= 1e-9 * max(1.0, float(np.max(mine["supply"]))),
                  f"2b: {tag}: intermediates not fully served at the fixed point "
                  f"(deficit {deficit:.3e})")

        # ---- optimality / collapse semantics --------------------------------
        obj_engine = float(np.sum(x))
        collapsed = obj_engine <= 1e-9 * max(1.0, float(np.max(inp["targets"])))
        if not collapsed and mine["iterations"] == 1:
            # one pass: x must be the optimum of the documented programme
            first_pass_optimality += 1
            for method in ("highs-ds", "highs-ipm"):
                _, obj = solve_goods_balance(cap["A"], mine["ub"],
                                             np.maximum(inp["inventory"], 0.0),
                                             mine["reserve"], cap["labor"],
                                             inp["labour_supply"], method=method)
                res.check(abs(obj - obj_engine) <= 1e-9 * max(1.0, abs(obj_engine)),
                          f"2c: {tag}: engine output not optimal for the documented "
                          f"programme ({method}: {obj:.6f} vs engine {obj_engine:.6f})")
        elif collapsed:
            # the cascade drove output to zero; that is only legitimate if the
            # outranking claims make every positive output inconsistent
            n_collapse += 1
            forced, n_blocked = collapse_is_forced(inp, cap["A"], cap["A"].shape[0])
            collapses_forced += int(forced)
            res.check(forced, f"2c: {tag}: cascade collapsed to x=0 although tier-0 claims "
                              f"do not exhaust inventory (not structurally forced)")
            res.numbers.setdefault("collapse_blocked_goods", []).append(n_blocked)

    counts = {k: len(v) for k, v in by_flag.items()}
    res.numbers["regime_counts"] = counts
    res.numbers["max_rel_diff_by_regime"] = {
        k: (max(v) if v else None) for k, v in by_flag.items()
    }
    res.numbers["max_relative_difference"] = worst_overall
    res.numbers["max_relative_difference_at"] = worst_where
    res.numbers["ladder_violations"] = ladder_bad
    res.numbers["engine_status_counts"] = statuses
    res.numbers["n_collapse_periods"] = n_collapse
    res.numbers["n_collapse_forced"] = collapses_forced
    res.numbers["n_first_pass_optimality_probes"] = first_pass_optimality
    res.numbers["max_cascade_exit_deficit_rel"] = max(exit_deficits) if exit_deficits else 0.0

    res.check(counts["calm"] >= 5, f"only {counts['calm']} calm periods captured (need >=5)")
    res.check(counts["capacity"] >= 5, f"only {counts['capacity']} capacity-binding periods (need >=5)")
    res.check(counts["tier"] >= 5, f"only {counts['tier']} tier-binding periods (need >=5)")
    res.check(ladder_bad == 0, f"priority ladder violated in {ladder_bad} (good, period) cases")

    if res.status == "PASS":
        res.summary = (
            f"My from-scratch period reproduces the engine on {len(captured)} captured periods "
            f"(regimes, non-exclusive: {counts['calm']} calm / {counts['capacity']} "
            f"capacity-binding / {counts['tier']} tier-binding): worst relative difference "
            f"{worst_overall:.2e} over x, Z, C, G, I, S_end, unmet, supply. The balance "
            f"identity, Z = A diag(x), capacity/target/labour bounds and the tier ladder hold; "
            f"single-pass output is certified optimal by two independent HiGHS algorithms; the "
            f"{n_collapse} collapse periods are structurally forced by outranking claims."
        )


def run() -> CheckResult:
    return run_guarded("engine", "Independent one-period implementation vs the engine", _body)


if __name__ == "__main__":                              # pragma: no cover
    import json

    result = run()
    print(json.dumps(result.as_dict(), indent=2, default=str))
    raise SystemExit(0 if result.status == "PASS" else 1)
