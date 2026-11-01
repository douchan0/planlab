"""Re-run a results cell from its recorded inputs — the verifier's own reconstruction.

The experiments layer records, for every cell: the full scenario dict, the
authoritative technology spec (``cell["tech"]``: substrate, A-perturbation and its
seed), and the optional *probe* (an instrument offset used only to measure the
local gain).  This module reconstructs all three from the record and re-runs the
cell, so that checks 3 and 4 can ask "does the stored result follow from the
recorded inputs?" without importing the runner.

The A-perturbation is re-implemented here from its *documented* definition
(``A' = A * exp(magnitude * z)``, columns rescaled to stay productive,
``base_final = max(x - A' x, 1e-3 x)``), so a mismatch with the project's own
perturbation is a finding rather than a tautology.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cell_tech_spec(cell: dict) -> dict:
    """The cell's authoritative technology spec (cell["tech"] if present)."""
    tech = cell.get("tech")
    if isinstance(tech, dict) and tech:
        return dict(tech)
    return dict(cell["scenario"]["technology"])


def resolve_cell_technology(cell: dict) -> Any:
    """Resolve the substrate and apply any recorded A-perturbation myself."""
    from planlab.core.scenarios import resolve_technology_spec
    from planlab.core.state import Technology

    spec = cell_tech_spec(cell)
    base_spec = {k: v for k, v in spec.items() if k not in {"a_perturb", "perturb_seed"}}
    tech = resolve_technology_spec(base_spec)
    magnitude = float(spec.get("a_perturb", 0.0) or 0.0)
    if magnitude <= 0:
        return tech
    seed = int(spec.get("perturb_seed", 0))
    rng = np.random.default_rng(seed)
    A = np.asarray(tech.A, dtype=float)
    A2 = A * np.exp(magnitude * rng.standard_normal(A.shape))
    colsum = A2.sum(axis=0)
    A2 = A2 * np.where(colsum >= 0.98, 0.98 / np.maximum(colsum, 1e-12), 1.0).reshape(1, -1)
    base = np.asarray(tech.base_output, dtype=float)
    final2 = np.maximum(base - A2 @ base, 1e-3 * np.maximum(base, 1e-9))
    meta = dict(getattr(tech, "meta", {}) or {})
    meta["a_perturbation_verifier"] = {"magnitude": magnitude, "seed": seed}
    return Technology(
        sectors=tech.sectors, A=A2, labor=tech.labor, kappa=tech.kappa,
        consumption_weights=tech.consumption_weights, essentials=tech.essentials,
        base_output=base, base_final=final2, meta=meta,
    )


def apply_probe(mech: Any, sim: Any, probe: dict | None, tech: Any) -> None:
    """Re-implement the probe contract: target_offset / price_offset."""
    if not probe:
        return
    kind = str(probe.get("kind", ""))
    sectors = list(getattr(tech, "sectors", ()))
    sector = probe.get("sector")
    delta = float(probe.get("delta", 0.0))
    if kind == "target_offset":
        targets = np.asarray(mech.targets, dtype=float)
        if sector is None:
            targets = targets * (1.0 + delta)
        else:
            idx = sectors.index(str(sector)) if isinstance(sector, str) else int(sector)
            targets = targets.copy()
            targets[idx] *= 1.0 + delta
        mech.targets = targets
    elif kind == "price_offset":
        price = np.asarray(sim.state.price, dtype=float).copy()
        if sector is None:
            price = price * (1.0 + delta)
        else:
            idx = sectors.index(str(sector)) if isinstance(sector, str) else int(sector)
            price[idx] *= 1.0 + delta
        sim.state.price = price
    else:
        raise ValueError(f"unknown probe kind {kind!r}")


def rerun_cell(cell: dict, seed: int):
    """Run one recorded cell with the verifier's reconstructor."""
    from planlab.core.engine import Simulation, run_scenario
    from planlab.core.scenarios import scenario_from_dict
    from planlab.mechanisms import get_mechanism

    scenario = scenario_from_dict(dict(cell["scenario"]))
    tech = resolve_cell_technology(cell)
    probe = cell.get("probe")
    if probe:
        mech = get_mechanism(cell["mechanism"], tech, scenario.mechanism_params)
        sim = Simulation(tech, mech, scenario, seed=int(seed))
        apply_probe(mech, sim, probe, tech)
        return sim.run(), tech
    return run_scenario(scenario, technology=tech, seed=int(seed)), tech


def capture_cell(cell: dict, seed: int):
    """Re-run a cell and capture the engine's reconcile inputs/outputs per period.

    Returns ``(result, captured)`` where ``captured[t]`` holds the claims, tiers,
    inventory, demand and returns of the engine's one-period routine at period t.
    """
    from planlab.core.engine import Simulation, run_scenario
    from planlab.core.scenarios import scenario_from_dict
    from planlab.mechanisms import get_mechanism

    scenario = scenario_from_dict(dict(cell["scenario"]))
    tech = resolve_cell_technology(cell)
    probe = cell.get("probe")
    captured: dict[int, dict] = {}
    original = Simulation._reconcile

    def wrapper(self, **kwargs):
        alloc = kwargs["alloc"]
        snapshot = {
            "claims": np.array(alloc.claims, dtype=float, copy=True),
            "tiers": np.array(alloc.tiers, dtype=float, copy=True),
            "labels": tuple(alloc.labels),
            "inventory": np.array(kwargs["inventory"], dtype=float, copy=True),
            "demand": np.array(kwargs["demand"], dtype=float, copy=True),
            "targets": np.array(kwargs["plan"].targets, dtype=float, copy=True),
            "capacity": np.array(kwargs["capacity"], dtype=float, copy=True),
            "gov_demand": np.array(kwargs["gov_demand"], dtype=float, copy=True),
            "invest_demand": np.array(kwargs["invest_demand"], dtype=float, copy=True),
            "labour_supply": float(kwargs["labour_supply"]),
            "t": int(self.state.t),
        }
        out = original(self, **kwargs)
        snapshot["outputs"] = out
        captured[snapshot["t"]] = snapshot
        return out

    Simulation._reconcile = wrapper
    try:
        if probe:
            mech = get_mechanism(cell["mechanism"], tech, scenario.mechanism_params)
            sim = Simulation(tech, mech, scenario, seed=int(seed))
            apply_probe(mech, sim, probe, tech)
            result = sim.run()
        else:
            result = run_scenario(scenario, technology=tech, seed=int(seed))
    finally:
        Simulation._reconcile = original
    return result, captured
