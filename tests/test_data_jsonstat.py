"""Unit tests for the JSON-stat reader (offline, dependency-free)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from planlab.data.jsonstat import JsonStat, JsonStatError

# A tiny dataset: id=[unit, a, b], size=[1, 2, 3]; the right-most dimension
# varies fastest, so flat = i*6 + j*3 + k.
SPARSE = {
    "version": "2.0",
    "class": "dataset",
    "label": "toy dataset",
    "updated": "2026-01-01T00:00:00+0100",
    "source": "ESTAT",
    "id": ["unit", "a", "b"],
    "size": [1, 2, 3],
    "dimension": {
        "unit": {"category": {"index": {"EUR": 0}, "label": {"EUR": "Euro"}}},
        "a": {"category": {"index": {"p": 0, "q": 1}, "label": {"p": "P", "q": "Q"}}},
        "b": {"category": {"index": {"r": 0, "s": 1, "t": 2}}},
    },
    # (j, k) = (0,0) -> 1.0 ; (0,1) -> 2.0 ; (1,0) -> 4.0 ; everything else absent
    "value": {"0": 1.0, "1": 2.0, "3": 4.0},
    "status": {"5": "c"},
}


def _payload(**overrides):
    out = json.loads(json.dumps(SPARSE))
    out.update(overrides)
    return out


def test_parse_reads_dimension_order_and_sizes():
    js = JsonStat.parse(json.dumps(SPARSE))
    assert js.ids == ("unit", "a", "b")
    assert js.sizes == (1, 2, 3)
    assert js.ndim == 3
    assert js.n_cells == 6
    assert js.defined() == 3
    assert js.code("a", 1) == "q"
    assert js.position("b", "t") == 2
    assert js.single("unit") == "EUR"
    assert js.single("a") is None
    assert js.status_counts() == {"c": 1}


def test_dimension_major_flat_index_is_decoded():
    """(j, k) = (1, 0) is flat index 3 and must land at [0, 1, 0]."""
    dense = JsonStat.parse(SPARSE).dense()
    assert dense.shape == (1, 2, 3)
    assert dense[0, 0, 0] == 1.0
    assert dense[0, 0, 1] == 2.0
    assert dense[0, 1, 0] == 4.0
    assert np.isnan(dense[0, 1, 1])


def test_sparse_and_dense_encodings_agree():
    sparse = JsonStat.parse(SPARSE)
    dense_payload = _payload(value=[1.0, 2.0, None, 4.0, None, None])
    dense_payload.pop("status", None)
    dense = JsonStat.parse(dense_payload)
    np.testing.assert_array_equal(sparse.dense(), dense.dense())
    assert dense.statuses == {}
    assert sparse.status_counts() == {"c": 1}


def test_category_index_may_be_a_position_list():
    payload = _payload()
    payload["dimension"]["b"]["category"]["index"] = ["r", "s", "t"]
    js = JsonStat.parse(payload)
    assert js.codes["b"] == ("r", "s", "t")
    assert js.position("b", "s") == 1


def test_undefined_cells_are_nan_not_zero():
    js = JsonStat.parse(SPARSE)
    dense = js.dense()
    mask = np.isnan(dense)
    assert mask.sum() == 3
    assert not np.any(dense[~mask] == 0.0)


def test_unknown_label_lookup_raises():
    js = JsonStat.parse(SPARSE)
    with pytest.raises(KeyError):
        js.position("a", "nope")
    with pytest.raises(KeyError):
        js.position("nope", "p")


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"id": None}, "id"),
        ({"size": None}, "size"),
        ({"dimension": None}, "dimension"),
    ],
)
def test_missing_members_raise(mutation, message):
    with pytest.raises(JsonStatError, match=message):
        JsonStat.parse(_payload(**mutation))


def test_inconsistent_category_count_raises():
    payload = _payload()
    payload["dimension"]["b"]["category"]["index"] = {"r": 0, "s": 1}
    with pytest.raises(JsonStatError, match="categories"):
        JsonStat.parse(payload)


def test_out_of_range_flat_index_raises():
    payload = _payload(value={"99": 1.0})
    with pytest.raises(JsonStatError, match="outside"):
        JsonStat.parse(payload)


def test_invalid_json_raises():
    with pytest.raises(JsonStatError, match="not valid JSON"):
        JsonStat.parse(b"{not json")


def test_singleton_axes_are_squeezed_by_slice():
    """Pinning every axis reproduces the raw payload as a scalar."""
    js = JsonStat.parse(SPARSE)
    cube = js.dense()
    assert cube.shape == (1, 2, 3)
    # Pinning the size-1 axis gives the ("a", "b") matrix.
    pinned = cube[0, :, :]
    assert pinned.shape == (2, 3)
