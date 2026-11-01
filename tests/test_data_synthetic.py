"""Tests for the synthetic substrate generator.

Two things matter scientifically: (i) the synthetic table must have the **same
schema** as the real one so experiments can swap substrates without special
cases, and (ii) it must be **calibrated to the real aggregate totals** while
having a genuinely different internal structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planlab.data.aggregate import AggregatedSIOT, load_sector_map
from planlab.data.synthetic import DEFAULT_SEED, generate_synthetic

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
MAPPING = REPO / "data" / "metadata" / "sector_map_eurostat64_to_20.csv"


def _reference() -> AggregatedSIOT:
    """Rebuild the real 20-sector aggregate from the committed artifacts."""
    meta = json.loads((PROCESSED / "eurostat_2020_n20_meta.json").read_text())
    import csv

    with (PROCESSED / "eurostat_2020_n20_A.csv").open() as handle:
        rows = list(csv.reader(handle))
    A = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    x = np.array(meta["gross_output"], dtype=float)
    vectors: dict[str, list[float]] = {}
    with (PROCESSED / "eurostat_2020_n20_vectors.csv").open() as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key != "sector":
                    vectors.setdefault(key, []).append(float(value))
    return AggregatedSIOT(
        sectors=tuple(meta["sectors"]),
        Z=A * x[None, :],
        x=x,
        y_total=np.array(vectors["final_demand_total"]),
        household=np.array(vectors["household"]),
        government=np.array(vectors["government"]),
        gfcf=np.array(vectors["gfcf"]),
        inventories=np.array(vectors["inventories"]),
        exports=np.array(vectors["exports"]),
        imports=np.array(vectors["imports"]),
        compensation=np.array(vectors["compensation"]),
        gva=np.array(vectors["gva"]),
        intermediate_consumption=np.array(vectors["intermediate_consumption"]),
        n_source=65,
    )


pytestmark = pytest.mark.skipif(
    not (PROCESSED / "eurostat_2020_n20_meta.json").is_file()
    or not MAPPING.is_file(),
    reason="eurostat substrate not built",
)


@pytest.fixture(scope="module")
def reference() -> AggregatedSIOT:
    return _reference()


@pytest.fixture(scope="module")
def synthetic(reference):
    return generate_synthetic(reference, seed=DEFAULT_SEED)


# ----------------------------------------------------------------- schema parity
def test_schema_is_identical_to_the_real_table(reference, synthetic):
    syn = synthetic.table
    assert syn.sectors == reference.sectors
    assert syn.n == reference.n == 20
    for name in (
        "Z",
        "x",
        "y_total",
        "household",
        "government",
        "gfcf",
        "inventories",
        "exports",
        "imports",
        "compensation",
        "gva",
        "intermediate_consumption",
    ):
        real = np.asarray(getattr(reference, name), dtype=float)
        drew = np.asarray(getattr(syn, name), dtype=float)
        assert drew.shape == real.shape, name
        assert drew.dtype == np.float64, name
        assert np.isfinite(drew).all(), name


def test_coefficients_are_well_formed(synthetic):
    table = synthetic.table
    A = table.A()
    assert A.shape == (20, 20)
    assert (A >= 0).all() and (A <= 1).all()
    assert (A.sum(axis=0) < 1.0).all()
    residual = A @ table.x + table.y_total - table.x
    assert np.max(np.abs(residual)) <= 1e-6 * np.max(np.abs(table.x))


def test_final_demand_components_sum_to_total(synthetic):
    table = synthetic.table
    components = (
        table.household + table.government + table.gfcf + table.inventories + table.exports
    )
    np.testing.assert_allclose(components, table.y_total, atol=1e-6 * table.y_total.max())


# --------------------------------------------------------------- calibration etc.
def test_calibrated_to_the_real_aggregate_totals(reference, synthetic):
    syn = synthetic.table
    # Gross output is copied verbatim.
    np.testing.assert_allclose(syn.x, reference.x, rtol=0, atol=0)
    # Sectoral intermediate budgets (hence GVA shares) are copied verbatim.
    np.testing.assert_allclose(syn.Z.sum(axis=0), reference.Z.sum(axis=0), rtol=1e-10)
    np.testing.assert_allclose(syn.A().sum(axis=0), reference.A().sum(axis=0), rtol=1e-10)
    # Aggregate final demand matches the real total to within the published
    # table's own rounding (x - Z*1 differs from the reported y by ~2 MIO EUR on
    # 14 trillion, i.e. 1.4e-7 relative).  Household consumption matches exactly.
    assert syn.y_total.sum() == pytest.approx(reference.y_total.sum(), rel=1e-6)
    assert syn.household.sum() == pytest.approx(reference.household.sum(), rel=1e-9)
    # Compensation and GVA vectors are carried over unchanged.
    np.testing.assert_allclose(syn.compensation, reference.compensation)
    np.testing.assert_allclose(syn.gva, reference.gva)


def test_structure_is_genuinely_different_from_the_real_table(reference, synthetic):
    A_real = reference.A()
    A_syn = synthetic.table.A()
    relative_distance = np.linalg.norm(A_real - A_syn) / np.linalg.norm(A_real)
    assert relative_distance > 0.2, "synthetic substrate is too close to the real one"
    # Not a mere rescaling of the real matrix.
    assert not np.allclose(A_syn, A_real, atol=1e-3)


def test_generator_is_deterministic_given_the_seed(reference):
    a = generate_synthetic(reference, seed=11).table
    b = generate_synthetic(reference, seed=11).table
    np.testing.assert_array_equal(a.Z, b.Z)
    np.testing.assert_array_equal(a.y_total, b.y_total)
    c = generate_synthetic(reference, seed=12).table
    assert not np.allclose(a.Z, c.Z)


def test_calibration_record_is_honest(synthetic):
    calibration = dict(synthetic.calibration)
    assert calibration["generator"] == "planlab.data.synthetic"
    assert calibration["seed"] == DEFAULT_SEED
    assert "drawn" in calibration and "copied_from_reference" in calibration
    achieved = calibration["achieved_totals"]
    assert achieved["total_intermediate_uses"] == pytest.approx(
        float(synthetic.table.Z.sum()), rel=1e-9
    )
    scale = max(float(synthetic.table.x.max()), 1.0)
    assert calibration["identity_residual_max_abs"] / scale <= 1e-10


def test_negative_margins_are_rejected():
    rng = np.random.default_rng(0)
    n = 5
    table = AggregatedSIOT(
        sectors=tuple(f"c{i:02d}" for i in range(n)),
        Z=rng.uniform(1, 2, size=(n, n)),
        x=np.full(n, 10.0),
        y_total=np.full(n, 9.0),
        household=np.full(n, 9.0),
        government=np.zeros(n),
        gfcf=np.zeros(n),
        inventories=np.zeros(n),
        exports=np.zeros(n),
        imports=np.zeros(n),
        compensation=np.zeros(n),
        gva=np.zeros(n),
        intermediate_consumption=np.full(n, 25.0),
        n_source=n,
    )
    with pytest.raises(ValueError):
        generate_synthetic(table, seed=1)
