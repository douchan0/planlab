"""Figure tests: every figure is produced by code, deterministically, and indexed.

The tests build one figure from a synthetic bundle (no simulation) and check that
the PNG and PDF exist, that the index row is written, and that two consecutive
builds of the same figure are byte-identical — the property a notebook needs to be
able to regenerate a figure and compare it with the stored artefact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from planlab.experiments import aggregate as A
from planlab.experiments import figures as F


def _noise_only_frame(n_seeds: int = 4) -> pd.DataFrame:
    rows = []
    for mechanism in A.MECH_ORDER:
        for seed in range(n_seeds):
            rows.append(
                {
                    "block": "B1b",
                    "cell_id": f"B1b.{mechanism}",
                    "mechanism": mechanism,
                    "seed": seed,
                    "substrate": "eurostat",
                    "lag": 0,
                    "m": 1,
                    "mean_shortage_rate": 0.016 + 0.001 * seed,
                    "output_ratio_mean": 0.99 - 0.01 * (mechanism == "material_balance"),
                    "mean_welfare": 250.0 + seed,
                    "mean_spoilage_rate": 0.0,
                    "cum_welfare": 12500.0,
                    "mean_iterations": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def bundle(tmp_path: Path) -> A.ResultsBundle:
    frame = _noise_only_frame()
    baseline = _noise_only_frame(n_seeds=1)
    baseline = baseline[baseline["mechanism"] == "lp_central"].copy()
    baseline["block"] = "B1"
    baseline["cell_id"] = "baseline_lp_k0"
    return A.ResultsBundle(root=tmp_path, summaries={"B1": baseline, "B1b": frame})


def test_noise_only_figure_is_written_and_indexed(bundle, tmp_path: Path):
    out = tmp_path / "figures"
    written: list[dict] = []
    F._style()
    F.make_noise_only(bundle, out, written)
    assert (out / "fig_noise_only.png").exists()
    assert (out / "fig_noise_only.pdf").exists()
    assert written and written[0]["figure"] == "fig_noise_only"
    assert (out / "fig_noise_only.png").stat().st_size > 1000


def test_figure_is_byte_stable(bundle, tmp_path: Path):
    F._style()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    w1: list[dict] = []
    w2: list[dict] = []
    F.make_noise_only(bundle, first, w1)
    F.make_noise_only(bundle, second, w2)
    assert (first / "fig_noise_only.png").read_bytes() == (second / "fig_noise_only.png").read_bytes()
    assert (first / "fig_noise_only.pdf").read_bytes() == (second / "fig_noise_only.pdf").read_bytes()


def test_make_all_tolerates_missing_blocks(tmp_path: Path):
    """A bundle with one block must still produce its figures and a full index."""
    baseline = _noise_only_frame(n_seeds=1)
    baseline = baseline[baseline["mechanism"] == "lp_central"].copy()
    baseline["block"] = "B1"
    baseline["cell_id"] = "baseline_lp_k0"
    bundle = A.ResultsBundle(root=tmp_path, summaries={"B1": baseline, "B1b": _noise_only_frame()})
    out = tmp_path / "figs"
    result = F.make_all(results_dir=tmp_path, out_dir=out, only=None, verbose=False)
    # every figure that needs a missing block is skipped, not an exception
    assert isinstance(result["missing"], list)
    index = pd.read_csv(tmp_path / "figures_index.csv")
    assert not index.empty
    for column in ("figure", "block", "png", "pdf", "what", "generated_by"):
        assert column in index.columns
    for _, row in index.iterrows():
        assert Path(row["png"]).exists()
        assert Path(row["pdf"]).exists()


def test_unknown_figure_name_is_reported(tmp_path: Path):
    result = F.make_all(results_dir=tmp_path, out_dir=tmp_path / "f", only=["not_a_figure"], verbose=False)
    assert result["figures"] == []
    assert result["missing"] and "not_a_figure" in result["missing"][0]
