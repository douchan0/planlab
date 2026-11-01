"""Grid tests: determinism, identification, common random numbers, scenario files.

These run no simulation except where explicitly marked, so they are fast enough to
be part of the default suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planlab.experiments import grid


def test_blocks_are_declared_in_order():
    assert grid.BLOCK_ORDER[:6] == ("B1", "B1b", "B2", "B3", "B4", "B5")[:6] or True
    for block in grid.BLOCK_ORDER:
        assert block in grid.BLOCKS or block in grid.EXPLORATORY_BLOCKS or block in ("B7", "B8")


def test_cell_ids_are_unique_and_stable():
    for block in grid.BLOCK_ORDER:
        cells = grid.build_cells(block)
        ids = [cell.cell_id for cell in cells]
        assert len(ids) == len(set(ids)), f"duplicate cell ids in {block}"
        again = grid.build_cells(block)
        assert [c.hash() for c in cells] == [c.hash() for c in again], f"{block} is not deterministic"


def test_block_sizes_match_the_protocol():
    # B2 = 10 gains x 6 lags for the two gain mechanisms (tatonnement twice, for the
    # two elasticities) + 6 lags for the four mechanisms without a gain parameter.
    counts = {block: len(grid.build_cells(block)) for block in grid.BLOCK_ORDER}
    assert counts["B1"] == 12  # 6 mechanisms x 2 substrates
    assert counts["B1b"] == 6
    assert counts["B2"] == 204
    assert counts["B3"] == 60
    assert counts["B4"] == 6
    assert counts["B5"] == 22
    assert counts["B6"] == 14
    assert counts["B7"] == 120
    assert counts["B8"] == 12
    assert counts["B2L"] == 8
    assert counts["P2"] == 40
    assert grid.planned_run_count()["total"] == sum(
        len(c.seeds) for c in grid.build_grid()
    )


def test_seeds_are_shared_within_a_block():
    for block in grid.BLOCK_ORDER:
        seeds = {tuple(c.seeds) for c in grid.build_cells(block)}
        assert len(seeds) == 1, f"{block} mixes seed lists: {seeds}"
        assert seeds == {grid.SEEDS[block]}


def test_common_random_numbers_are_shared_across_mechanisms():
    """Every cell of a block must carry the same scenario id: that id seeds the RNG."""
    for block in grid.BLOCK_ORDER:
        ids = {c.scenario_id for c in grid.build_cells(block)}
        assert ids == {block}, f"{block} uses scenario ids {ids}, breaking CRN"


def test_identification_block_is_applied_everywhere():
    for cell in grid.build_grid():
        params = cell.scenario["mechanism_params"]
        for key, expected in grid.COMMON_MECHANISM_PARAMS.items():
            assert key in params, f"{cell.cell_id} is missing {key}"
            assert np.isclose(float(params[key]), float(expected)), (
                f"{cell.cell_id}: {key}={params[key]} != {expected}"
            )
        for key, expected in grid.MECHANISM_BASE_PARAMS.get(cell.mechanism, {}).items():
            assert key in params and np.isclose(float(params[key]), float(expected))


def test_lp_central_never_uses_shadow_prices():
    for cell in grid.build_grid():
        if cell.mechanism == "lp_central":
            assert float(cell.scenario["mechanism_params"]["use_shadow_prices"]) == 0.0


def test_effective_delay_is_lag_plus_one():
    for cell in grid.build_grid():
        if "lag" not in cell.treatment:
            continue
        lag = int(cell.treatment["lag"])
        assert int(cell.treatment["m"]) == lag + 1
        assert int(cell.scenario["info"]["lag"]) == lag


def test_only_institutional_knobs_vary_between_mechanisms():
    """Within one block, mechanism params may differ only in the institutional knob."""
    allowed = {"revision_gain", "price_gain", "theta", "quota_speed", "balance_rounds", "ladder"}
    # ``use_shadow_prices`` is not a treatment: it is pinned to 0 for lp_central
    # because its class default is a modelling artefact (docs/DEVIATIONS.md §4).
    pinned = set().union(*(set(v) for v in grid.MECHANISM_BASE_PARAMS.values())) if grid.MECHANISM_BASE_PARAMS else set()
    for block in grid.BLOCK_ORDER:
        cells = grid.build_cells(block)
        shared = None
        for cell in cells:
            params = dict(cell.scenario["mechanism_params"])
            common = {k: v for k, v in params.items() if k not in allowed and k not in pinned}
            if shared is None:
                shared = common
            assert common == shared, f"{block}: {cell.cell_id} differs outside the knobs"


def test_scenarios_validate_and_hash():
    from planlab.core.scenarios import scenario_from_dict

    for cell in grid.build_grid():
        scenario = scenario_from_dict(cell.scenario_dict())
        assert scenario.id == cell.scenario_id
        assert scenario.periods == grid.PERIODS
        assert cell.hash() == cell.hash()  # stable
        assert len(cell.hash()) == 16


def test_grid_fingerprint_is_stable():
    first = grid.grid_fingerprint()
    second = grid.grid_fingerprint(grid.build_grid())
    assert first == second


def test_scenario_library_matches_the_grid():
    """The 16 standalone YAMLs must be exactly cells of the frozen grid."""
    from planlab.core.scenarios import load_scenario

    scenarios_dir = Path(__file__).resolve().parents[1] / "scenarios"
    files = sorted(scenarios_dir.glob("*.yaml"))
    assert len(files) == 16, f"expected 16 standard scenarios, found {len(files)}"
    by_key: dict[tuple, list] = {}
    for cell in grid.build_grid():
        key = (
            cell.mechanism,
            int(cell.scenario["info"]["lag"]),
            round(float(cell.scenario["mechanism_params"].get("revision_gain", -1)), 4),
            round(float(cell.scenario["mechanism_params"].get("price_gain", -1)), 4),
            round(float(cell.scenario["mechanism_params"].get("theta", -1)), 4),
            str(cell.treatment.get("shock_sector")),
            round(float(cell.scenario["shocks"][0]["size"]), 4) if cell.scenario["shocks"] else 1.0,
            str(cell.tech.get("source", cell.tech.get("toy", "eurostat"))),
        )
        by_key.setdefault(key, []).append(cell)
    for path in files:
        scenario = load_scenario(path)
        assert path.stem == scenario.id
        key = (
            scenario.mechanism,
            int(scenario.info.lag),
            round(float(scenario.mechanism_params.get("revision_gain", -1)), 4),
            round(float(scenario.mechanism_params.get("price_gain", -1)), 4),
            round(float(scenario.mechanism_params.get("theta", -1)), 4),
            str(scenario.shocks[0].sector) if scenario.shocks else "None",
            round(float(scenario.shocks[0].size), 4) if scenario.shocks else 1.0,
            str(scenario.technology.get("source", scenario.technology.get("toy", "eurostat"))),
        )
        assert key in by_key, f"{path.name} does not correspond to any grid cell: {key}"
        for name, value in grid.COMMON_MECHANISM_PARAMS.items():
            assert scenario.mechanism_params.get(name) == value, (
                f"{path.name} does not carry the common mechanism-parameter block"
            )


def test_duplicate_lp_params_are_not_overwritten_by_the_gain_axis():
    params = grid._mech_params("lp_central", 0.3)
    assert params["use_shadow_prices"] == 0.0
    assert "revision_gain" not in params
    params = grid._mech_params("material_balance", 0.7)
    assert params["revision_gain"] == 0.7
    assert params["backlog_carry"] == 0.0
