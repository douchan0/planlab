"""Tests for the 65 -> 20 value-space aggregation.

The scientific requirement is that aggregation happens in **value space**: sum
the transactions, then divide.  :func:`test_value_space_aggregation_matches_a_manual_sum`
and :func:`test_aggregating_coefficients_would_give_a_different_answer` pin that
down so a future refactor cannot quietly switch to averaging coefficients.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from planlab.data.aggregate import (
    ESSENTIAL_AGGREGATES,
    GROUP_DEFINITIONS,
    aggregate_siot,
    consumption_weights,
    essentials,
    group_vector,
    grouping_matrix,
    load_sector_map,
    sector_labels,
)
from planlab.data.eurostat import SIOT, to_coefficients

REPO = Path(__file__).resolve().parents[1]
MAPPING = REPO / "data" / "metadata" / "sector_map_eurostat64_to_20.csv"

pytestmark = pytest.mark.skipif(
    not MAPPING.is_file(), reason="sector mapping not built"
)


@pytest.fixture(scope="module")
def smap():
    return load_sector_map(MAPPING)


@pytest.fixture(scope="module")
def fake_siot(smap):
    """A random but internally consistent 65-branch table on the real codes."""
    rng = np.random.default_rng(7)
    n = smap.n_source
    x = rng.uniform(1000.0, 50000.0, size=n)
    shares = rng.uniform(0.05, 0.55, size=n)
    raw = rng.uniform(0.1, 1.0, size=(n, n))
    Z = raw * (shares[None, :] * x[None, :]) / raw.sum(axis=0, keepdims=True)
    y = x - Z.sum(axis=0)  # row sums of Z + y = x  (per product)
    y = np.maximum(y, 0.0)
    # Rebuild Z so the identity holds exactly with the clipped y.
    row_target = x - y
    Z = raw * row_target[:, None] / raw.sum(axis=1, keepdims=True)
    y = x - Z.sum(axis=1)
    codes = tuple(smap.code64)
    return SIOT(
        codes=codes,
        names={c: c for c in codes},
        Z=Z,
        x=x,
        y_total=y,
        household=y * 0.6,
        government=y * 0.2,
        gfcf=y * 0.1,
        inventories=np.zeros(n),
        exports=y * 0.1,
        imports=np.zeros(n),
        compensation=x * 0.25,
        gva=x - Z.sum(axis=0),
        intermediate_consumption=Z.sum(axis=0),
        unit="MIO_EUR",
        price_basis="basic",
        stk_flow="DOM",
        provenance={},
    )


# ------------------------------------------------------------------- mapping IO
def test_mapping_covers_every_source_code_exactly_once(smap):
    assert smap.n_source == 65
    assert len(set(smap.code64)) == smap.n_source
    assert all(code.startswith("CPA_") for code in smap.code64)


def test_mapping_has_exactly_twenty_groups(smap):
    assert smap.n_target == 20
    assert len(set(smap.sectors)) == 20
    assert set(smap.sectors) == {g[0] for g in GROUP_DEFINITIONS}
    counts = smap.counts()
    assert sum(counts.values()) == 65
    assert all(count > 0 for count in counts.values()), "every group needs a source code"


def test_mapping_identifiers_are_ascii(smap):
    for code in smap.code64:
        assert code.isascii()
    for sector in smap.sectors:
        assert sector.isascii()


def test_every_group_definition_is_used_once(smap):
    codes = [g[0] for g in GROUP_DEFINITIONS]
    assert len(codes) == len(set(codes)) == 20
    for code, name_en, name_zh, aggregate in GROUP_DEFINITIONS:
        assert code.isascii() and name_en and name_zh and aggregate
        assert smap.counts()[code] >= 1


def test_grouping_matrix_is_one_hot_per_source_column(smap, fake_siot):
    S = grouping_matrix(smap, fake_siot.codes)
    assert S.shape == (20, 65)
    np.testing.assert_allclose(S.sum(axis=0), 1.0)
    assert set(np.unique(S)) <= {0.0, 1.0}
    # Each target sector has at least one source column.
    np.testing.assert_allclose((S.sum(axis=1) > 0), np.ones(20, dtype=bool))


def test_group_vector_sums_values(smap, fake_siot):
    values = np.arange(65, dtype=float)
    grouped = group_vector(smap, values, fake_siot.codes)
    assert grouped.shape == (20,)
    np.testing.assert_allclose(grouped.sum(), values.sum())
    first = smap.sectors[0]
    idx = [i for i, c in enumerate(smap.code64) if smap.sector20[i] == first]
    np.testing.assert_allclose(grouped[0], values[idx].sum())


def test_unknown_target_sector_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code64", "name_en", "sector20_code", "mapping_note"])
        writer.writerow(["CPA_A01", "x", "c99", ""])
    with pytest.raises(ValueError, match="unknown target sector"):
        load_sector_map(bad)


def test_incomplete_mapping_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code64", "name_en", "sector20_code", "mapping_note"])
        writer.writerow(["CPA_A01", "x", "c01", ""])
    with pytest.raises(ValueError, match="no source codes|target sectors"):
        load_sector_map(bad)


# ----------------------------------------------------------------- aggregation
def test_value_space_aggregation_matches_a_manual_sum(smap, fake_siot):
    agg = aggregate_siot(fake_siot, smap)
    S = grouping_matrix(smap, fake_siot.codes)
    np.testing.assert_allclose(agg.Z, S @ fake_siot.Z @ S.T, rtol=0, atol=0)
    np.testing.assert_allclose(agg.x, S @ fake_siot.x)
    np.testing.assert_allclose(agg.y_total, S @ fake_siot.y_total)
    # Coefficients are derived *after* summing, not summed.
    np.testing.assert_allclose(agg.A(), (S @ fake_siot.Z @ S.T) / agg.x[None, :])


def test_aggregation_preserves_value_totals(smap, fake_siot):
    agg = aggregate_siot(fake_siot, smap)
    np.testing.assert_allclose(agg.Z.sum(), fake_siot.Z.sum(), rtol=1e-12)
    np.testing.assert_allclose(agg.x.sum(), fake_siot.x.sum(), rtol=1e-12)
    np.testing.assert_allclose(agg.y_total.sum(), fake_siot.y_total.sum(), rtol=1e-12)
    np.testing.assert_allclose(agg.household.sum(), fake_siot.household.sum(), rtol=1e-12)
    np.testing.assert_allclose(agg.compensation.sum(), fake_siot.compensation.sum(), rtol=1e-12)


def test_aggregating_coefficients_would_give_a_different_answer(smap, fake_siot):
    """Averaging coefficients is wrong; the test fixes the correct behaviour."""
    agg = aggregate_siot(fake_siot, smap)
    S = grouping_matrix(smap, fake_siot.codes)
    A_source = to_coefficients(fake_siot.Z, fake_siot.x)
    # Naive (incorrect) aggregation: the plain average of the source coefficients
    # inside each group pair.  It ignores how much output each source branch has,
    # which is exactly the error value-space aggregation avoids.
    counts = np.array([smap.counts()[s] for s in smap.sectors], dtype=float)
    naive = (S @ A_source @ S.T) / np.outer(counts, counts)
    assert naive.shape == agg.A().shape
    assert not np.allclose(agg.A(), naive, atol=1e-6)


def test_aggregated_coefficients_are_well_formed(smap, fake_siot):
    agg = aggregate_siot(fake_siot, smap)
    A = agg.A()
    assert A.shape == (20, 20)
    assert np.isfinite(A).all()
    assert (A >= 0).all() and (A <= 1).all()
    assert (A.sum(axis=0) < 1.0).all()
    # The identity that the simulation relies on: A x + y = x.
    np.testing.assert_allclose(A @ agg.x + agg.y_total, agg.x, atol=1e-9 * agg.x.max())


def test_aggregated_identity_holds_on_the_real_table():
    """A x + y ~= x for the committed 20-sector substrate."""
    processed = REPO / "data" / "processed"
    meta_path = processed / "eurostat_2020_n20_meta.json"
    if not meta_path.is_file():
        pytest.skip("eurostat substrate not built")
    import json

    meta = json.loads(meta_path.read_text())
    with (processed / "eurostat_2020_n20_A.csv").open() as handle:
        rows = list(csv.reader(handle))
    A = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    x = np.array(meta["gross_output"], dtype=float)
    y = np.array(meta["base_final"], dtype=float)
    residual = A @ x + y - x
    assert np.max(np.abs(residual)) <= 1e-6 * np.max(np.abs(x))
    assert (A.sum(axis=0) < 1.0).all()


# ------------------------------------------------------------------- attributes
def test_essentials_and_weights_follow_the_aggregate_tags(smap):
    w = consumption_weights(smap)
    e = essentials(smap)
    tags = {code: aggregate for code, _, _, aggregate in GROUP_DEFINITIONS}
    assert w.shape == e.shape == (20,)
    for i, sector in enumerate(smap.sectors):
        expected = 1.0 if tags[sector] in ESSENTIAL_AGGREGATES else 0.0
        assert e[i] == expected
        assert w[i] == (1.2 if expected else 1.0)
    assert 0 < e.sum() < 20, "the essentials tier must be a strict subset"


def test_labels_are_complete(smap):
    rows = sector_labels(smap)
    assert len(rows) == 20
    assert [r["sector_index"] for r in rows] == list(range(20))
    assert [r["code"] for r in rows] == list(smap.sectors)
    for row in rows:
        assert row["name_en"] and row["name_zh"]
        assert row["essential"] in (0, 1)
        assert int(row["n_source"]) >= 1
    assert sum(int(r["n_source"]) for r in rows) == 65


def test_unmapped_source_code_raises(smap):
    smap_short = type(smap)(
        code64=smap.code64[:-1],
        sector20=smap.sector20[:-1],
        sectors=smap.sectors,
        notes=smap.notes,
        names=smap.names,
    )
    codes = (*smap.code64,)
    with pytest.raises(ValueError, match="no mapping entry"):
        grouping_matrix(smap_short, codes)
