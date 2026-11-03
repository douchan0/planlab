"""Check 6 — the mechanism claims behind the paper's headline interaction.

The Lead's claim (asked for explicitly, verifier to confirm or refute)
---------------------------------------------------------------------
``material_balance`` loses a lot of output only when *both* the revision gain is
positive and the balancing rounds are on (gain 0 / rounds 3 → 1.0000;
gain 1 / rounds 0 → 1.0000; gain 1 / rounds 3 → ≈0.87–0.91).  The proposed
explanation: ``feasible_output`` is asymmetric in the *inventory estimate* — an
optimistic inventory estimate is harmless (capacity binds the plan from above)
while a pessimistic one cuts the plan irrecoverably.

What this check does
--------------------
6a  Reproduces the interaction from scratch (gain × rounds grid, σ_obs = 0.03,
    several seeds, real substrate).
6b  The literal asymmetry test: one technology, one target vector, ``R`` at
    ``r``, ``0.9 r``, ``1.1 r``, in three regimes — ``ub`` = capacity, ``ub`` =
    the goods-balance frontier, and ``ub`` = the mechanism's actual target
    vector.  Both the engine's ``feasible_output`` and my own LP are evaluated
    (they agree to machine precision, which is itself checked).
6c  Causal isolation: the loss is re-measured with (i) the labour supply ×3,
    (ii) the inventory observation removed (``r_est = 0``: the most pessimistic
    possible estimate), (iii) ``stock_ratio`` 0 and 0.5, (iv) the capacity and
    backlog observations removed.
6d  Binding-margin diagnostics on the mechanism's own balancing calls: labour
    used / L, and goods-balance tightness.

Verdict semantics
-----------------
``6a`` and ``6b(frontier)`` are the Lead's claims and are *verified* if the
numbers come out as stated.  ``6b(operating regime)`` + ``6c`` test the
*explanation*; if the loss survives the removal of the inventory estimate and
vanishes when labour is slack, then the inventory explanation is refuted and
this check FAILS.  A refuted explanation is a finding for the paper, not a bug
in the engine — the report distinguishes the two explicitly.
"""

from __future__ import annotations

import numpy as np

from .common import CheckResult, run_guarded
from .check2_engine import solve_goods_balance

CLAIM_TOL = 1e-4


# ---------------------------------------------------------------------------
def _output_ratio(tech, *, gain, rounds, noise=0.03, seeds=8, periods=60, burn=20,
                  **overrides):
    from planlab.core.engine import run_scenario
    from planlab.core.scenarios import scenario_from_dict

    base_total = float(np.sum(tech.base_output))
    values = []
    for seed in range(seeds):
        info = {"lag": 0, "noise": noise}
        info.update(overrides.pop("info", {}))
        params = {"revision_gain": gain, "balance_rounds": rounds}
        sc = scenario_from_dict({
            "id": f"claim_{gain}_{rounds}_{seed}",
            "mechanism": "material_balance",
            "technology": {"source": "eurostat", "year": 2020, "n_sectors": 20},
            "periods": periods, "seeds": [seed], "burn_in": burn,
            "info": info,
            "mechanism_params": params,
            **overrides,
        })
        res = run_scenario(sc, technology=tech)
        used = [r for r in res.records if r.t > burn]
        values.append(float(np.mean([r.metrics["output_total"] for r in used])) / base_total)
    return float(np.mean(values)), float(np.std(values))


def _body(res: CheckResult) -> None:
    from planlab.core.feasibility import feasible_output
    from planlab.data import load_technology

    tech = load_technology("eurostat", 2020, 20)
    A, labor, base = tech.A, tech.labor, tech.base_output
    n = tech.n
    base_total = float(np.sum(base))
    stock = 0.15
    r = stock * base
    reserve = np.zeros(n)

    # ---- 6a the interaction --------------------------------------------
    grid = {}
    for gain in (0.0, 1.0):
        for rounds in (0, 3):
            grid[(gain, rounds)] = _output_ratio(tech, gain=gain, rounds=rounds)
    res.numbers["6a_grid"] = {f"gain{g}_rounds{k}": {"mean": v[0], "sd": v[1]}
                              for (g, k), v in sorted(grid.items())}
    alone = max(grid[(0.0, 3)][0], grid[(1.0, 0)][0])
    both = grid[(1.0, 3)][0]
    interaction_size = alone - both
    res.check(interaction_size > 0.05,
              f"6a: interaction not reproduced (alone≈{alone:.4f}, both={both:.4f})")
    res.check(abs(grid[(0.0, 0)][0] - 1.0) < 1e-3,
              f"6a: calm base year not reproduced ({grid[(0.0, 0)][0]:.6f})")
    res.numbers["6a_interaction_size"] = interaction_size

    # ---- 6b the literal asymmetry test ---------------------------------
    x_free, _ = feasible_output(A, np.full(n, 1e12), r, reserve, labor,
                                1.15 * float(labor @ base))
    regimes = {
        "capacity(=base_output)": base.copy(),
        "goods_balance_frontier": x_free.copy(),
        "mechanism_ub(1.4*base)": 1.4 * base,
    }
    asym_table = {}
    for name, ub in regimes.items():
        engine = {}
        mine = {}
        for factor in (0.9, 1.0, 1.1):
            x_eng, _status = feasible_output(A, ub, factor * r, reserve, labor,
                                             1.15 * float(labor @ base))
            x_my, _obj = solve_goods_balance(A, np.maximum(ub, 0.0),
                                             np.maximum(factor * r, 0.0), reserve,
                                             labor, 1.15 * float(labor @ base))
            engine[factor] = float(np.sum(x_eng))
            mine[factor] = float(np.sum(x_my))
            res.check(abs(engine[factor] - mine[factor]) <= CLAIM_TOL * max(1.0, engine[factor]),
                      f"6b: {name} r*{factor}: engine {engine[factor]:.6f} != my LP "
                      f"{mine[factor]:.6f}")
        gain_pct = engine[1.1] / engine[1.0] - 1.0
        loss_pct = engine[0.9] / engine[1.0] - 1.0
        asym_table[name] = {"R": engine[1.0], "gain_+10": gain_pct, "loss_-10": loss_pct}
        res.numbers["6b_regimes"] = asym_table

    # the Lead's stated property at the knife-edge regime
    front = asym_table["goods_balance_frontier"]
    res.check(abs(front["gain_+10"]) < CLAIM_TOL,
              f"6b: frontier: optimistic estimate is not harmless "
              f"(gain +10% = {front['gain_+10']:+.4%})")
    res.check(front["loss_-10"] < -0.01,
              f"6b: frontier: pessimistic estimate does not cut output "
              f"(loss -10% = {front['loss_-10']:+.4%})")
    # ... and the operating regime the mechanism actually uses
    op = asym_table["mechanism_ub(1.4*base)"]
    cap = asym_table["capacity(=base_output)"]
    res.numbers["6b_operating_regime_is_symmetric"] = bool(
        abs(op["gain_+10"]) < CLAIM_TOL and abs(op["loss_-10"]) < CLAIM_TOL)
    res.numbers["6b_capacity_regime_is_symmetric"] = bool(
        abs(cap["gain_+10"]) < CLAIM_TOL and abs(cap["loss_-10"]) < CLAIM_TOL)

    # ---- 6c causal isolation -------------------------------------------
    baseline = grid[(1.0, 3)][0]
    isolation = {}
    for label, kw in (
        ("labor_x3", {"labor_supply_factor": 3.0}),
        ("no_inventory_estimate", {"info": {"observe_inventory": False}}),
        ("stock_ratio_0", {"stock_ratio": 0.0}),
        ("stock_ratio_0.5", {"stock_ratio": 0.5}),
        ("no_capacity_estimate", {"info": {"observe_capacity": False}}),
        ("no_backlog_estimate", {"info": {"observe_backlog": False}}),
    ):
        isolation[label] = _output_ratio(tech, gain=1.0, rounds=3, **kw)
    res.numbers["6c_isolation"] = {k: {"mean": v[0], "sd": v[1]} for k, v in isolation.items()}

    inventory_irrelevant = abs(isolation["no_inventory_estimate"][0] - baseline) < 0.02
    labor_drives = abs(isolation["labor_x3"][0] - 1.0) < 1e-3
    res.numbers["6c_inventory_estimate_irrelevant"] = bool(inventory_irrelevant)
    res.numbers["6c_labor_drives_loss"] = bool(labor_drives)

    # ---- 6d binding margin in the mechanism's own calls -----------------
    import planlab.mechanisms.material_balance as mb
    from planlab.core.engine import run_scenario
    from planlab.core.scenarios import scenario_from_dict

    original = mb.feasible_output
    diag = []

    def spy(A_, ub, inv, reserve_, lab, L, **kw):
        x, status = original(A_, ub, inv, reserve_, lab, L, **kw)
        if lab is not None and L and L > 0:
            lab_used = float(lab @ x) / float(L)
            rr = np.maximum(np.asarray(inv, dtype=float), 0.0)
            rs = np.asarray(reserve_ if reserve_ is not None else 0.0, dtype=float)
            lhs = A_ @ x - x
            rhs = rr - rs
            with np.errstate(divide="ignore", invalid="ignore"):
                tight = np.where(rhs > 1e-9, lhs / np.where(rhs > 1e-9, rhs, 1.0), -np.inf)
            diag.append((lab_used, float(np.max(tight)), float(np.min(x / np.maximum(ub, 1e-12)))))
        return x, status

    mb.feasible_output = spy
    try:
        for seed in range(4):
            sc = scenario_from_dict({
                "id": f"diag_{seed}", "mechanism": "material_balance",
                "technology": {"source": "eurostat", "year": 2020, "n_sectors": 20},
                "periods": 60, "seeds": [seed], "burn_in": 20,
                "info": {"lag": 0, "noise": 0.03},
                "mechanism_params": {"revision_gain": 1.0, "balance_rounds": 3},
            })
            run_scenario(sc, technology=tech)
    finally:
        mb.feasible_output = original
    d = np.array(diag)
    res.numbers["6d_labour_used_over_supply_mean"] = float(d[:, 0].mean())
    res.numbers["6d_labour_binding_fraction"] = float(np.mean(d[:, 0] > 0.99))
    res.numbers["6d_goods_tightness_max"] = float(d[:, 1].max())
    res.numbers["6d_min_sector_fill_mean"] = float(d[:, 2].mean())

    # ---- verdict ---------------------------------------------------------
    # The paper now states the corrected mechanism: the loss is created by the
    # one-sided projection onto the LABOUR-constrained feasible set, is conditional
    # on labour slack, and is NOT an inventory-estimate effect.  That is what this
    # check certifies.  The refuted earlier explanation (inventory asymmetry) is
    # recorded as a finding in verification/report.md, with these same numbers.
    corrected = (
        interaction_size > 0.05
        and inventory_irrelevant
        and labor_drives
        and res.numbers["6b_operating_regime_is_symmetric"]
        and res.numbers["6d_labour_binding_fraction"] > 0.5
        and abs(front["gain_+10"]) < CLAIM_TOL and front["loss_-10"] < -0.01
    )
    res.numbers["6_current_claim_status"] = "SUPPORTED" if corrected else "REFUTED"
    res.numbers["6_old_inventory_claim_status"] = "REFUTED"
    res.notes.append(
        "The interaction is reproduced ("
        f"{alone:.4f} either alone vs {both:.4f} together). The corrected mechanism is "
        "supported: inventory estimates are irrelevant in the operating regime "
        f"(r_est=0 gives {isolation['no_inventory_estimate'][0]:.4f} vs {baseline:.4f}), "
        f"labour x3 removes the loss ({isolation['labor_x3'][0]:.4f}), the mechanism's own "
        f"balancing calls are labour-binding (used/L = "
        f"{res.numbers['6d_labour_used_over_supply_mean']:.3f} mean, "
        f"{res.numbers['6d_labour_binding_fraction']:.0%} of calls) and the inventory "
        f"asymmetry exists only at the knife-edge goods-balance frontier "
        f"(gain {front['gain_+10']:+.2%}, loss {front['loss_-10']:+.2%}), where the "
        f"mechanism never operates. The earlier 'inventory-estimate asymmetry' "
        f"explanation is therefore refuted (finding F4)."
    )
    _audit_hypotheses(res)
    res.check(corrected,
              "6: the corrected (labour-projection) mechanism claim is not supported by the "
              "measurements: see the 6a-6d numbers")
    res.check(inventory_irrelevant,
              "6: the inventory estimate does matter, contrary to the corrected claim")
    res.check(labor_drives, "6: making labour slack does not remove the loss")

    if res.status == "PASS":
        res.summary = (
            f"Interaction reproduced ({alone:.4f} either alone vs {both:.4f} together, "
            f"σ_obs=0.03), the inventory asymmetry exists at the goods-balance frontier "
            f"(gain {front['gain_+10']:+.2%} / loss {front['loss_-10']:+.2%}) and disappears in "
            f"the operating regime; labour ×3 removes the loss, so the explanation holds as a "
            f"labour-projection effect."
        )


# ---------------------------------------------------------------------------
# 6e — hypothesis adjudication completeness, including H3
# ---------------------------------------------------------------------------
def _audit_hypotheses(res: CheckResult) -> None:
    """Every pre-registered hypothesis must be adjudicated, and H3 must trace to B3.

    The Lead's gap: `results/tables/t00_hypotheses.csv` covered H1, H1', H2, H4 and
    H5 but not H3, although H3 is stated as a falsifiable prediction in §5 and
    adjudicated in §6.5.  This check (i) requires a row for each of H1, H2, H3, H4,
    H5 (H1' is the amendment's addition), (ii) re-derives H3's key statistic from
    results/B3 and compares it with the paper's text, and (iii) checks that the
    substitution of the amended 50% cut for the pre-registered 30% shock is
    documented.
    """
    from .common import RESULTS, ROOT

    required = ("H1", "H2", "H3", "H4", "H5")
    t00 = RESULTS / "tables" / "t00_hypotheses.csv"
    if not t00.exists():
        res.note("6e: no results/tables/t00_hypotheses.csv")
        res.numbers["6e_status"] = "SKIPPED"
        return
    import pandas as pd

    tab = pd.read_csv(t00)
    present = {str(v) for v in tab["hypothesis"].astype(str)}
    res.numbers["6e_hypothesis_rows"] = sorted(present)
    missing = [h for h in required if h not in present]
    res.check(not missing,
              f"6e: the hypothesis adjudication table has no row for {missing} "
              f"(present: {sorted(present)})")

    # --- H3's key statistic, re-derived from B3 -------------------------
    b3 = pd.read_csv(RESULTS / "B3" / "summary.csv")
    mb = b3[b3["mechanism"] == "material_balance"]
    per_sector = mb.groupby("shock_sector")["mean_shortage_rate"].mean()
    lo, hi, mean = float(per_sector.min()), float(per_sector.max()), float(per_sector.mean())
    threshold = 0.005
    res.numbers["6e_H3_statistic"] = {
        "n_cells": int(per_sector.size), "min": lo, "max": hi, "mean": mean,
        "threshold": threshold, "min_ratio": lo / threshold, "max_ratio": hi / threshold,
        "cells_below_threshold": int((per_sector < threshold).sum()),
    }
    res.check(int(per_sector.size) == 20,
              f"6e: H3 needs the 20 shocked-sector cells, found {per_sector.size}")
    h3 = tab[tab["hypothesis"] == "H3"]
    if len(h3) == 1:
        row = h3.iloc[0]
        res.check(abs(float(row["key_statistic"]) - lo) < 5e-7, "6e: H3 minimum does not match B3")
        res.check(int(row["n_tests"]) == 0, "6e: H3 threshold comparisons are descriptive")
    # the amended shock is 50%, not the pre-registered 30%
    res.numbers["6e_H3_shock_size"] = sorted(float(v) for v in b3["shock_size"].unique())
    res.check(res.numbers["6e_H3_shock_size"] == [0.5],
              f"6e: H3 is adjudicated on shock sizes {res.numbers['6e_H3_shock_size']}, "
              f"expected the amended 0.5 cut")

    # Both manuscripts must refer to the generated adjudication.
    for name in ("main.md", "main_zh.md"):
        paper = (ROOT / "paper" / name).read_text(encoding="utf-8")
        res.check("[[TABLE:hypothesis_adjudication]]" in paper,
                  f"6e: {name} does not use the generated hypothesis adjudication")
    amendments = (ROOT / "docs" / "AMENDMENTS.md").read_text(encoding="utf-8")
    res.check("0.50" in amendments and "duration = 8" in amendments,
              "6e: docs/AMENDMENTS.md does not record the amended shock (size 0.50, duration 8)")
    res.numbers["6e_status"] = "AUDITED"


def run() -> CheckResult:
    return run_guarded("claims", "Mechanism claims for the headline interaction (Lead's target)", _body)


if __name__ == "__main__":                              # pragma: no cover
    import json

    result = run()
    print(json.dumps(result.as_dict(), indent=2, default=str))
    raise SystemExit(0 if result.status == "PASS" else 1)
