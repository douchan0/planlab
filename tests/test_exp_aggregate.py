"""Aggregation tests: the statistical layer on synthetic frames, no simulation.

The point is that ``build_tables`` works on any frame with the run-level schema,
including one seed per cell, so the table code can be exercised before (or without)
the full grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from planlab.experiments import aggregate as A


def _summary_rows(block: str, mechanisms, n_seeds: int, *, base: float = 1000.0, **extra) -> pd.DataFrame:
    rows = []
    for mechanism in mechanisms:
        for seed in range(n_seeds):
            rng = np.random.default_rng(abs(hash((block, mechanism, seed))) % (2**32))
            rows.append(
                {
                    "block": block,
                    "cell_id": f"{block}.{mechanism}",
                    "mechanism": mechanism,
                    "seed": seed,
                    "lag": extra.get("lag", 0),
                    "m": extra.get("m", 1),
                    "gain": extra.get("gain", 0.3),
                    "price_elasticity": extra.get("price_elasticity", 0.4),
                    "theta": extra.get("theta", np.nan),
                    "mean_shortage_rate": float(0.02 + 0.001 * seed + 0.01 * (mechanism == "material_balance")),
                    "max_shortage_rate": float(0.2 + rng.normal(0, 0.01)),
                    "cum_welfare": float(base - 5 * seed - 10 * (mechanism == "material_balance")),
                    "mean_welfare": float((base - 5 * seed) / 50),
                    "mean_plan_fulfilment": float(0.95 + rng.normal(0, 0.005)),
                    "mean_iterations": 1.0,
                    "oscillation_amplitude": float(0.01 + 0.002 * extra.get("lag", 0) + rng.normal(0, 1e-4)),
                    "acf_lag1": float(0.4 + 0.05 * extra.get("lag", 0)),
                    "oscillation_period": 10.0,
                    "output_ratio_mean": float(0.99 + rng.normal(0, 0.002)),
                    "shock_output_loss": float(0.05 + 0.01 * seed),
                    "shock_output_amplification": float(2.0 + 0.1 * seed),
                    "shock_amplification": 5.0,
                    "shock_peak_shortage": 0.19,
                    "recovery_half_life": 2.0,
                    "mean_spoilage_rate": 0.01,
                    "mean_labour_utilisation": 0.99,
                    "mean_investment_ratio": 0.043,
                    "capital_ratio_final": 0.95,
                    "plan_lowering_periods": 30.0,
                    "accounting_residual_max": 1e-16,
                    "accounting_checks_passed": 50.0,
                    "n_periods_used": 50.0,
                    "variant": extra.get("variant", ""),
                    "substrate": "eurostat",
                    "shock_sector": extra.get("shock_sector", "c12"),
                    "shock_size": extra.get("shock_size", 0.5),
                    "stock_ratio": 0.1,
                    "delta": np.nan,
                    "sector": np.nan,
                    "ladder": extra.get("ladder", "production_first"),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _bundle(tmp_path: Path) -> A.ResultsBundle:
    b1 = _summary_rows("B1", A.MECH_ORDER, 1, base=1000.0)
    b1["cell_id"] = b1["cell_id"] + ".eurostat"
    b1.loc[b1["mechanism"] == "lp_central", "cell_id"] = "baseline_lp_k0"
    b1b = _summary_rows("B1b", A.MECH_ORDER, 3)
    b2 = pd.concat(
        [
            _summary_rows("B2", A.MECH_ORDER, 3, lag=lag, m=lag + 1)
            for lag in (0, 1, 3)
        ],
        ignore_index=True,
    )
    b2["gain_axis"] = "swept"
    b3 = pd.concat(
        [_summary_rows("B3", ["material_balance"], 2, shock_sector=s) for s in ("c01", "c12")],
        ignore_index=True,
    )
    b3["shock_output_loss"] = [0.03, 0.05, 0.10, 0.16]
    sectors = pd.DataFrame(
        [
            {
                "block": "B3",
                "cell_id": row["cell_id"],
                "mechanism": row["mechanism"],
                "seed": row["seed"],
                "sector": "c12",
                "is_shocked": True,
                "household_unmet_rate_mean": 0.3,
                "household_unmet_rate_peak": 0.85,
                "unmet_share_of_total": 0.2,
                "final_share_of_rationed": 0.8,
                "other_share_of_rationed": 0.2,
                "intermediate_service_rate": 0.78,
                "plan_gap_inter_mean": 1.0,
                "unmet_final_mean": 10.0,
                "unmet_other_mean": 2.0,
                "x_ratio_mean": 0.95,
                "x_ratio_min": 0.49,
                "capacity_ratio_mean": 0.99,
                "inventory_ratio_mean": 0.1,
                "demand_share": 0.05,
                "demand_mean": 100.0,
            }
            for _, row in b3.iterrows()
        ]
    )
    b5 = pd.concat(
        [
            _summary_rows("B5", ["hybrid_dual_track"], 2, theta=theta, lag=lag)
            for theta in (0.0, 0.5, 1.0)
            for lag in (0, 3)
        ],
        ignore_index=True,
    )
    b6 = pd.concat(
        [
            _summary_rows("B6", ["material_balance"], 2, variant=v)
            for v in ("reference", "synthetic", "balance_rounds_0", "labour115_taut")
        ],
        ignore_index=True,
    )
    b6["labor_supply_factor"] = 1.15
    b6["sigma_obs"] = 0.02
    b6["a_perturb"] = 0.0
    b8 = pd.concat(
        [
            _summary_rows("B8", ["material_balance"], 2, gain=gain)
            for gain in (0.0, 0.3)
        ],
        ignore_index=True,
    )
    b8["revision_gain"] = b8["gain"]
    b8["balance_rounds"] = 3
    b8["output_ratio_mean"] = 1.0 - 0.05 * b8["revision_gain"]
    return A.ResultsBundle(
        root=tmp_path,
        summaries={
            "B1": b1,
            "B1b": b1b,
            "B2": b2,
            "B3": b3,
            "B5": b5,
            "B6": b6,
            "B8": b8,
        },
        sectors={"B3": sectors},
        manifests={"B2": {"grid_fingerprint": "abc", "git_commit": "deadbeef"}},
    )


def test_build_tables_writes_csv_tex_and_tests(tmp_path: Path):
    bundle = _bundle(tmp_path)
    out = tmp_path / "tables"
    result = A.build_tables(bundle, out)
    files = {p.name for p in out.iterdir()}
    assert "tests.csv" in files and "manifest.json" in files
    assert any(name.endswith(".tex") for name in files)
    assert any(name.startswith("t02_noise_only") for name in files)
    assert any(name.startswith("t03_lag_mechanism") for name in files)
    assert any(name.startswith("t13_material_balance") for name in files)
    tests = result["tests"]
    assert not tests.empty
    for column in ("block", "hypothesis", "test", "p_value", "p_holm", "effect_value"):
        assert column in tests.columns
    # Holm correction is within block and never below the raw p-value
    valid = tests.dropna(subset=["p_value", "p_holm"])
    assert (valid["p_holm"] >= valid["p_value"] - 1e-12).all()
    # the noise-only table has the columns the Lead asked for
    noise = pd.read_csv(out / "t02_noise_only_information_rent.csv")
    for column in (
        "mechanism", "x_over_base_mean", "mean_shortage_rate_mean",
        "mean_welfare_mean", "mean_spoilage_rate", "n_seeds",
    ):
        assert column in noise.columns


def test_single_seed_cells_do_not_break_aggregation(tmp_path: Path):
    bundle = _bundle(tmp_path)
    bundle.summaries["B1b"] = bundle.summaries["B1b"].groupby("mechanism").head(1)
    result = A.build_tables(bundle, tmp_path / "tables-single")
    assert not result["tests"].empty
    table = pd.read_csv(tmp_path / "tables-single" / "t02_noise_only_information_rent.csv")
    assert set(table["n_seeds"]) == {1}


def test_holm_is_monotone_and_conservative():
    p = np.array([0.001, 0.02, 0.03, 0.2])
    adjusted = A.holm(p)
    assert np.all(adjusted >= p - 1e-12)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_paired_test_handles_identical_series():
    a = np.linspace(0.1, 0.2, 20)
    res = A.paired_test(a, a)
    assert res["mean_diff"] == 0.0
    assert res["p_value"] == 1.0
    assert res["n_pairs"] == 20


def test_cliffs_delta_bounds():
    a = np.arange(20.0)
    b = np.arange(20.0) + 1.0
    delta = A.cliffs_delta(a, b)
    assert -1.0 <= delta <= 1.0
    assert delta < 0


def test_ols_cluster_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    n = 200
    lag = rng.integers(0, 5, n).astype(float)
    y = 1.0 + 0.5 * lag + rng.normal(0, 0.01, n)
    frame = pd.DataFrame({"lag": lag, "y": y, "seed": rng.integers(0, 20, n)})
    fit = A.ols_cluster(frame, "y", ["lag"], cluster="seed")
    assert fit["params"]["lag"] == pytest.approx(0.5, abs=0.01)
    assert fit["p"]["lag"] < 0.01


def test_tex_is_booktabs_and_escaped():
    frame = pd.DataFrame({"mechanism": ["material_balance"], "value": [0.5]})
    tex = A.to_tex(frame, caption="A \\emph{test}", label="tab:t", notes="note")
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "material\\_balance" in tex
    assert "tab:t" in tex


def test_g_star_matches_the_notes():
    assert A.G_STAR(1) == pytest.approx(2.0)
    assert A.G_STAR(2) == pytest.approx(1.0)
    assert A.G_STAR(3) == pytest.approx(0.6180339887, abs=1e-9)
    assert A.G_STAR(4) == pytest.approx(0.4450418679, abs=1e-9)
    assert A.PERIOD_STAR(4) == 18
