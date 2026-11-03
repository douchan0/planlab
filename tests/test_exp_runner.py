"""Runner tests: shards, resumes, the audit sample, and common random numbers.

All simulations here are on the 6-sector toy fixture with a short horizon, so the
file runs in a couple of seconds.  The real-substrate numbers are covered by the
block manifests, not by unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from planlab.experiments import grid, runner


def _toy_scenario(mechanism: str, **overrides):
    scenario = {
        "id": "TEST",
        "name": f"test {mechanism}",
        "mechanism": mechanism,
        "technology": {"toy": 6},
        "periods": 24,
        "seeds": [0, 1],
        "burn_in": 6,
        "shocks": [
            {"type": "capacity", "sector": 1, "start": 12, "duration": 4, "size": 0.6}
        ],
        "info": {"lag": 1, "noise": 0.02},
        "demand": {"growth": 0.0, "ar1": 0.3, "sigma": 0.02,
                   "price_elasticity": 0.4, "income_elasticity": 1.0},
        "objective": {"shortage": 1.0, "inventory": 0.1, "plan_deviation": 0.5,
                      "inequality": 0.0, "adjustment": 0.1},
        "household": {"essential_priority": 0.0, "backlog_persistence": 0.5, "hoarding": 0.0},
        "mechanism_params": grid._mech_params(mechanism, 0.3),
        "stock_ratio": 0.10,
        "tags": ["test"],
    }
    scenario.update(overrides)
    return scenario


def _toy_cell(mechanism: str, *, cell_id: str | None = None, seeds=(0, 1), **overrides) -> grid.Cell:
    scenario = _toy_scenario(mechanism, **overrides)
    scenario["seeds"] = list(seeds)
    return grid.Cell(
        block="TEST",
        cell_id=cell_id or f"TEST.{mechanism}",
        mechanism=mechanism,
        scenario=scenario,
        seeds=tuple(seeds),
        treatment={"block": "TEST", "mechanism": mechanism, "lag": 1, "m": 2},
        tech={"toy": 6},
    )


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "TEST"


def test_run_cells_writes_shards_and_resumes(out_dir: Path):
    cells = [_toy_cell("material_balance"), _toy_cell("lp_central",
                                                    mechanism_params=dict(
                                                        grid.COMMON_MECHANISM_PARAMS,
                                                        use_shadow_prices=0.0))]
    results, skipped, failed = runner.run_cells(cells, out_dir, workers=2)
    assert failed == 0 and skipped == 0 and len(results) == 2
    for cell in cells:
        assert runner.is_complete(out_dir, cell)
        frame = pd.read_parquet(out_dir / "cells" / f"{cell.cell_id}.parquet")
        assert set(frame["seed"].unique()) == {0, 1}
        assert {"shortage_rate", "welfare", "accounting_residual_rel", "x_ratio"} <= set(frame.columns)
        # the accounting residual is machine-zero by construction of the engine
        assert frame["accounting_residual_rel"].max() < 1e-10

    results2, skipped2, failed2 = runner.run_cells(cells, out_dir, workers=2)
    assert results2 == [] and skipped2 == 2 and failed2 == 0


def test_assemble_block_manifest_is_complete(out_dir: Path):
    cells = [_toy_cell("material_balance"), _toy_cell("market")]
    runner.run_cells(cells, out_dir, workers=2)
    manifest = runner.assemble_block("TEST", results_dir=out_dir.parent, cells=cells)
    for key in (
        "block", "package_versions", "code_fingerprint", "environment",
        "mechanisms_present", "mechanisms_missing", "cells_planned", "cells_completed",
        "runs_completed", "seeds_per_cell", "scenario_hashes", "grid_fingerprint",
        "wall_time_s", "identification_audit", "mechanism_base_params",
        "code_fingerprints_seen", "mixed_code_versions", "audit_records",
    ):
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["identification_audit"]["ok"] is True
    assert manifest["mixed_code_versions"] is False
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "aggregate.csv").exists()
    assert (out_dir / "runs.parquet").exists()
    assert (out_dir / "sector_panel.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    # per-cell aggregation: one row per cell, n_seeds == 2
    aggregate = pd.read_csv(out_dir / "aggregate.csv")
    assert len(aggregate) == 2
    assert set(aggregate["n_seeds"]) == {2}


def test_identification_audit_rejects_a_tampered_cell():
    good = _toy_cell("material_balance")
    scenario = dict(good.scenario)
    scenario["mechanism_params"] = {"revision_gain": 0.3}  # the confound, restored
    bad = grid.Cell(
        block="TEST",
        cell_id="TEST.bad",
        mechanism="material_balance",
        scenario=scenario,
        seeds=(0,),
        treatment={"block": "TEST", "mechanism": "material_balance"},
        tech={"toy": 6},
    )
    with pytest.raises(AssertionError, match="identification audit failed"):
        runner._identification_audit([bad])


def test_audit_sample_is_self_consistent(out_dir: Path):
    cells = [_toy_cell("material_balance", seeds=(0,))]
    runner.run_cells(cells, out_dir, workers=1)
    runner.assemble_block("TEST", results_dir=out_dir.parent, cells=cells)
    audit_path = out_dir / "audit_records.npz"
    assert audit_path.exists()
    with np.load(audit_path) as data:
        assert data["x"].shape[1] == 6
        assert data["x"].shape[0] >= 1
        for name in ("Z", "C", "G", "invest", "inventory_end", "demand", "unmet",
                     "capacity", "plan_targets", "supply", "price", "spoilage", "A"):
            assert name in data, f"audit file is missing {name}"
        # Z == A diag(x) exactly: the engine's construction, re-derived from the dump
        for i in range(data["x"].shape[0]):
            A = data["A"][i]
            x = data["x"][i]
            assert np.allclose(data["Z"][i], A * x.reshape(1, -1), atol=1e-9)
            # balance identity including spoilage
            rhs = (data["Z"][i].sum(axis=1) + data["C"][i] + data["G"][i]
                   + data["invest"][i] + data["inventory_end"][i] + data["spoilage"][i])
            scale = max(1.0, float(np.max(np.abs(data["supply"][i]))))
            assert np.max(np.abs(data["supply"][i] - rhs)) / scale < 1e-8
            assert np.all(data["C"][i] <= data["demand"][i] + 1e-9)
    assert (out_dir / "audit_index.csv").exists()


def test_common_shock_and_noise_draws_across_mechanisms(out_dir: Path):
    """Common random numbers = common *draws*, not common realised demand.

    The exogenous stream (capacity-shock factors, AR(1) innovations, measurement
    noise) is seeded from ``(scenario_id, seed)``, so two mechanisms of the same
    block see the same draws.  Realised demand is **not** common: it carries three
    endogenous channels — income elasticity (a mechanism that produces less also
    generates less demand), price elasticity, and the backlog carry-over of last
    period's unmet demand.  This test switches all three off, so the realised path
    is purely exogenous and must coincide exactly; the companion test below keeps
    them on and checks that it does not.
    """
    cells = [
        _toy_cell("material_balance", cell_id="TEST.crn.mb", seeds=(0,),
                  demand={"growth": 0.0, "ar1": 0.3, "sigma": 0.02,
                          "price_elasticity": 0.0, "income_elasticity": 0.0},
                  household={"essential_priority": 0.0, "backlog_persistence": 0.0,
                             "hoarding": 0.0}),
        _toy_cell("lp_central", cell_id="TEST.crn.lp", seeds=(0,),
                  demand={"growth": 0.0, "ar1": 0.3, "sigma": 0.02,
                          "price_elasticity": 0.0, "income_elasticity": 0.0},
                  household={"essential_priority": 0.0, "backlog_persistence": 0.0,
                             "hoarding": 0.0}),
    ]
    runner.run_cells(cells, out_dir, workers=1)
    frames = {}
    for cell in cells:
        frame = pd.read_parquet(out_dir / "cells" / f"{cell.cell_id}.parquet")
        frames[cell.mechanism] = frame.sort_values("t")
    mb, lp = frames["material_balance"], frames["lp_central"]
    assert np.allclose(mb["demand_total"].to_numpy(), lp["demand_total"].to_numpy(), rtol=0, atol=0)
    # the *outcomes* differ, otherwise the pairing would be vacuous
    assert not np.allclose(mb["shortage_rate"].to_numpy(), lp["shortage_rate"].to_numpy())


def test_income_elastic_demand_is_not_common_across_mechanisms(out_dir: Path):
    """With income-elastic demand the realised path legitimately differs."""
    cells = [
        _toy_cell("material_balance", cell_id="TEST.crn2.mb", seeds=(0,)),
        _toy_cell("lp_central", cell_id="TEST.crn2.lp", seeds=(0,),
                  ),
    ]
    runner.run_cells(cells, out_dir, workers=1)
    frames = {
        cell.mechanism: pd.read_parquet(out_dir / "cells" / f"{cell.cell_id}.parquet").sort_values("t")
        for cell in cells
    }
    mb, lp = frames["material_balance"], frames["lp_central"]
    assert not np.allclose(mb["demand_total"].to_numpy(), lp["demand_total"].to_numpy())


def test_failed_mechanism_is_recorded_not_raised(out_dir: Path):
    cell = _toy_cell("no_such_mechanism")
    results, _skipped, failed = runner.run_cells([cell], out_dir, workers=1)
    assert failed == 1
    assert results[0]["status"] == "error"
    meta = json.loads((out_dir / "cells" / f"{cell.cell_id}.json").read_text())
    assert meta["status"] == "error" and "no_such_mechanism" in meta["error"]
    # a failed cell must not be considered complete, so the next run retries it
    assert not runner.is_complete(out_dir, cell)


def test_resume_recomputes_a_shard_from_an_older_code_version(out_dir):
    cell = _toy_cell("material_balance")
    runner.run_cells([cell], out_dir, workers=1)
    metadata = out_dir / "cells" / f"{cell.cell_id}.json"
    saved = json.loads(metadata.read_text())
    saved["code_fingerprint"] = "previous-version"
    metadata.write_text(json.dumps(saved))

    _, skipped, failed = runner.run_cells([cell], out_dir, workers=1)
    assert skipped == failed == 0
    assert runner.is_complete(out_dir, cell)
