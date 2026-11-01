"""Tests for the Eurostat SIOT parser: orientation, identities, real artifacts.

The decisive test is :func:`test_use_rows_products_columns_orientation`: it feeds
a hand-built 2x2 table in the *actual* Eurostat layout (users as rows, products
as columns) and asserts that the parser returns the Leontief orientation
(``Z[product, industry]``) with the documented accounting identity intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from planlab.data import eurostat as E
from planlab.data.eurostat import EurostatError, parse_siot, to_coefficients

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"

# ---------------------------------------------------------------------------
# Hand-built payload in the exact Eurostat layout.
#
#   products: CPA_A, CPA_B
#   Z (product x industry):  [[10, 20],
#                             [30, 40]]      -> row sums (intermediate use
#                                               of each product) = [30, 70]
#   x = [100, 200]                            -> y_total must be [70, 130]
#   final use: household [55, 85], government [5, 20], gfcf [5, 10],
#              inventories 0, exports [5, 15]
#   column sums of Z = [40, 60] -> intermediate shares [0.40, 0.30]
#
#   In the payload's orientation X[prd_use=user, prd_ava=product] = Z[product, user]
# ---------------------------------------------------------------------------
USE_CODES = [
    "TOTAL",
    "CPA_A",
    "CPA_B",
    "P3",
    "P3_S13",
    "P3_S14",
    "P5",
    "P51G",
    "P6",
    "TU",
]
AVA_CODES = ["CPA_TOTAL", "CPA_A", "CPA_B", "P1", "P2_ADJ", "B1G", "D1", "P7", "IMP"]

CELLS: dict[tuple[str, str], float] = {
    # intermediate block, X[user, product]
    ("CPA_A", "CPA_A"): 10.0,
    ("CPA_A", "CPA_B"): 30.0,
    ("CPA_B", "CPA_A"): 20.0,
    ("CPA_B", "CPA_B"): 40.0,
    # output stored along the use axis (observed Eurostat convention)
    ("CPA_A", "P1"): 100.0,
    ("CPA_B", "P1"): 200.0,
    # value added / intermediate consumption
    ("CPA_A", "P2_ADJ"): 40.0,
    ("CPA_B", "P2_ADJ"): 60.0,
    ("CPA_A", "B1G"): 60.0,
    ("CPA_B", "B1G"): 140.0,
    ("CPA_A", "D1"): 45.0,
    ("CPA_B", "D1"): 60.0,
    # final use rows, indexed by product
    ("P3_S14", "CPA_A"): 55.0,
    ("P3_S14", "CPA_B"): 85.0,
    ("P3_S13", "CPA_A"): 5.0,
    ("P3_S13", "CPA_B"): 20.0,
    ("P3", "CPA_A"): 60.0,
    ("P3", "CPA_B"): 105.0,
    ("P5", "CPA_A"): 5.0,
    ("P5", "CPA_B"): 10.0,
    ("P51G", "CPA_A"): 5.0,
    ("P51G", "CPA_B"): 10.0,
    ("P6", "CPA_A"): 5.0,
    ("P6", "CPA_B"): 15.0,
    # totals row/column (not used by the parser, kept consistent)
    ("TOTAL", "CPA_A"): 30.0,
    ("TOTAL", "CPA_B"): 60.0,
}


def _toy_payload(use_codes=None, ava_codes=None) -> dict:
    """Build the toy payload; strides are recomputed from the category lists."""
    use_codes = list(use_codes or USE_CODES)
    ava_codes = list(ava_codes or AVA_CODES)
    sizes = [1, 1, len(use_codes), len(ava_codes)]
    strides = [sizes[1] * sizes[2] * sizes[3], sizes[2] * sizes[3], sizes[3], 1]
    values: dict[str, float] = {}
    for (use_code, ava_code), amount in CELLS.items():
        flat = (
            0 * strides[0]
            + 0 * strides[1]
            + use_codes.index(use_code) * strides[2]
            + ava_codes.index(ava_code) * strides[3]
        )
        values[str(flat)] = amount
    return {
        "version": "2.0",
        "class": "dataset",
        "label": "toy symmetric input-output table",
        "source": "ESTAT",
        "updated": "2026-01-01T00:00:00+0100",
        "id": ["unit", "stk_flow", "prd_use", "prd_ava"],
        "size": sizes,
        "dimension": {
            "unit": {"category": {"index": {"MIO_EUR": 0}}},
            "stk_flow": {"category": {"index": {"DOM": 0}}},
            "prd_use": {"category": {"index": {c: i for i, c in enumerate(use_codes)}}},
            "prd_ava": {"category": {"index": {c: i for i, c in enumerate(ava_codes)}}},
        },
        "value": values,
        "extension": {"datastructure": {"id": "NAIO_10_CP1700", "version": "61.0"}},
    }


# ----------------------------------------------------------------- orientation
def test_use_rows_products_columns_orientation():
    """Z must be [product, industry], i.e. the transpose of the payload slice."""
    siot = parse_siot(_toy_payload())
    assert siot.codes == ("CPA_A", "CPA_B")
    assert siot.Z.shape == (2, 2)
    np.testing.assert_allclose(siot.Z, [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(siot.x, [100.0, 200.0])


def test_final_use_and_value_added_are_read_on_the_right_axes():
    siot = parse_siot(_toy_payload())
    np.testing.assert_allclose(siot.household, [55.0, 85.0])
    np.testing.assert_allclose(siot.government, [5.0, 20.0])
    np.testing.assert_allclose(siot.gfcf, [5.0, 10.0])
    np.testing.assert_allclose(siot.inventories, [0.0, 0.0])  # P5 - P51G
    np.testing.assert_allclose(siot.exports, [5.0, 15.0])
    np.testing.assert_allclose(siot.y_total, [70.0, 130.0])
    np.testing.assert_allclose(siot.compensation, [45.0, 60.0])
    np.testing.assert_allclose(siot.gva, [60.0, 140.0])


def test_toy_table_satisfies_the_accounting_identity_exactly():
    siot = parse_siot(_toy_payload())
    residual = siot.output_residual()
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    A = siot.A()
    assert np.all(A.sum(axis=0) < 1.0)
    np.testing.assert_allclose(A @ siot.x + siot.y_total, siot.x)


def test_final_demand_components_sum_to_total_final_use():
    siot = parse_siot(_toy_payload())
    components = (
        siot.household + siot.government + siot.gfcf + siot.inventories + siot.exports
    )
    np.testing.assert_allclose(components, siot.y_total, atol=1e-12)


def test_unpublished_product_codes_are_dropped_without_losing_values():
    # Eurostat lists a finer CPA sub-code that it does not publish any value for.
    payload = _toy_payload(
        use_codes=[*USE_CODES, "CPA_A01"], ava_codes=[*AVA_CODES, "CPA_A01"]
    )
    siot = parse_siot(payload)
    assert "CPA_A01" not in siot.codes
    assert siot.provenance["n_unpublished_codes"] == 1
    assert siot.provenance["unpublished_codes"] == ["CPA_A01"]


def test_wrong_unit_filter_raises():
    with pytest.raises(EurostatError, match="unit"):
        parse_siot(_toy_payload(), unit="MIO_NAC")


def test_missing_structural_dimension_raises():
    payload = _toy_payload()
    payload["id"] = ["unit", "stk_flow", "prd_use", "something_else"]
    payload["dimension"]["something_else"] = payload["dimension"].pop("prd_ava")
    with pytest.raises(EurostatError, match="prd_ava"):
        parse_siot(payload)


def test_negative_intermediate_flow_is_rejected():
    payload = _toy_payload()
    # Corrupt one intermediate cell to a negative value.
    key = None
    for flat, amount in payload["value"].items():
        if amount == 10.0:
            key = flat
            break
    assert key is not None
    payload["value"][key] = -5.0
    with pytest.raises(EurostatError, match="negative"):
        parse_siot(payload)


def test_to_coefficients_zero_column_for_zero_output():
    Z = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = to_coefficients(Z, np.array([2.0, 0.0]))
    np.testing.assert_allclose(A, [[0.5, 0.0], [1.5, 0.0]])


def test_default_url_records_every_filter():
    url = E.default_url(year=2020)
    assert "naio_10_cp1700" in url
    for fragment in ("unit=MIO_EUR", "stk_flow=DOM", "geo=EU27_2020", "time=2020", "format=JSON"):
        assert fragment in url, fragment


# ------------------------------------------------------- committed real artifacts
pytestmark_real = pytest.mark.skipif(
    not (PROCESSED / "eurostat_2020_n65_Z.csv").is_file(),
    reason="65-sector audit artifacts not built (run python -m planlab.data.build)",
)


@pytestmark_real
def test_real_table_identity_holds_at_full_resolution():
    """``A x + y ~= x`` at 65 sectors within 1e-6 relative (pre-aggregation)."""
    import csv

    with (PROCESSED / "eurostat_2020_n65_Z.csv").open() as handle:
        rows = list(csv.reader(handle))
    codes = rows[0][1:]
    Z = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    vectors = {
        row["sector"]: row
        for row in csv.DictReader((PROCESSED / "eurostat_2020_n65_vectors.csv").open())
    }
    x = np.array([float(vectors[c]["gross_output"]) for c in codes])
    y = np.array([float(vectors[c]["final_demand_total"]) for c in codes])

    assert len(codes) == 65
    assert Z.shape == (65, 65)
    A = to_coefficients(Z, x)
    residual = A @ x + y - x
    scale = max(float(np.max(np.abs(x))), 1.0)
    assert np.max(np.abs(residual)) / scale <= 1e-6
    assert np.isfinite(A).all()
    assert (A >= 0).all() and (A <= 1).all()
    positive = A.sum(axis=0)
    assert (positive < 1.0).all()


@pytestmark_real
def test_real_table_value_added_matches_published_eu27_gva():
    """The implied GVA total must be within 1% of Eurostat's published EU27 GVA."""
    import json

    meta = json.loads((PROCESSED / "eurostat_2020_n20_meta.json").read_text())
    manifest_path = REPO / "data" / "metadata" / "manifest.json"
    published = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        published = manifest.get("validation", {}).get("published_gva_total_mio_eur")
    implied = meta["checks"]["totals"]["gross_value_added"]
    assert implied > 0
    if published is None:
        pytest.skip("published GVA cross-check unavailable in the manifest")
    assert abs(implied - published) / published < 0.01
