"""Tests for the public loader: schema conformance, offline guarantee, idempotency.

These are the contract tests that ``docs/DATA_SCHEMA.md`` §3–§5 demand, plus the
guarantee that ``load_technology`` never reaches the network and that whatever it
returns is accepted unchanged by the simulation core's own validator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from planlab.data import (
    DataError,
    artifact_key,
    available_substrates,
    build_eurostat,
    load_technology,
)
from planlab.data.loader import KAPPA_DEFAULT, make_technology, paths

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
METADATA = REPO / "data" / "metadata"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "eurostat_2020_n20_meta.json").is_file(),
    reason="substrates not built (run: python -m planlab.data.build --source both)",
)

SUBSTRATES = sorted(key for key in available_substrates() if key.endswith("_n20"))


# --------------------------------------------------------------------- fixtures
@pytest.fixture(params=SUBSTRATES)
def substrate(request) -> str:
    return request.param


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if anything in these tests opens a socket."""

    def _blocked(*args, **kwargs):
        raise AssertionError("load_technology must never touch the network")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    yield
    monkeypatch.undo()


# -------------------------------------------------------------- schema conformance
def test_available_substrates_lists_both(substrate):
    assert "eurostat_2020_n20" in SUBSTRATES
    assert "synthetic_2020_n20" in SUBSTRATES


def test_schema_conformance(substrate):
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    n = 20
    assert tech.n == n
    assert len(tech.sectors) == n
    assert len(set(tech.sectors)) == n
    assert all(s.isascii() for s in tech.sectors)
    assert tech.A.shape == (n, n)
    for name in ("labor", "kappa", "consumption_weights", "essentials", "base_output", "base_final"):
        assert np.asarray(getattr(tech, name)).shape == (n,), name
    for name in ("A", "labor", "kappa", "consumption_weights", "essentials", "base_output", "base_final"):
        assert np.isfinite(np.asarray(getattr(tech, name))).all(), f"{name} has NaN/inf"
    assert isinstance(tech.meta, dict)


def test_coefficients_are_admissible(substrate):
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    A = np.asarray(tech.A)
    assert (A >= 0).all() and (A <= 1).all()
    column_sums = A.sum(axis=0)
    assert (column_sums < 1.0).all(), f"column sums must all be < 1, got {column_sums.max()}"
    assert column_sums.min() >= 0.0


def test_the_table_is_productive(substrate):
    """A productive Leontief matrix (spectral radius < 1) keeps the engine finite."""
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    A = np.asarray(tech.A)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    assert spectral_radius < 1.0, f"spectral radius {spectral_radius} >= 1"
    inverse = np.linalg.inv(np.eye(A.shape[0]) - A)
    assert np.isfinite(inverse).all()
    assert (inverse >= 0).all(), "the Leontief inverse must be non-negative"
    assert tech.meta["checks"]["productive"] is True
    assert tech.meta["checks"]["spectral_radius"] == pytest.approx(spectral_radius, rel=1e-9)


def test_base_year_balance_holds(substrate):
    """A @ x + y = x, which the core requires of base_output/base_final."""
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    residual = np.asarray(tech.A) @ np.asarray(tech.base_output) + np.asarray(tech.base_final) - np.asarray(tech.base_output)
    assert np.max(np.abs(residual)) <= 1e-6 * np.max(np.abs(tech.base_output))


def test_documented_assumptions_are_flagged(substrate):
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    meta = tech.meta
    assert meta["kappa_fallback"] is True
    np.testing.assert_allclose(tech.kappa, KAPPA_DEFAULT)
    assert meta["labor_fallback"] is False
    assert set(np.unique(tech.essentials)) <= {0.0, 1.0}
    assert set(np.unique(tech.consumption_weights)) <= {1.0, 1.2}
    # Every essentials sector is also weighted 1.2 and vice versa.
    np.testing.assert_array_equal(
        tech.essentials > 0, np.isclose(tech.consumption_weights, 1.2)
    )
    assert meta["provenance"]["aggregation"].startswith("64->20")
    assert meta["provenance"]["transform_version"] == "1"


def test_meta_has_every_required_key(substrate):
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    required = {
        "source",
        "year",
        "n_sectors",
        "unit",
        "price_basis",
        "table_type",
        "sectors",
        "gross_output",
        "final_consumption",
        "labor",
        "kappa",
        "consumption_weights",
        "essentials",
        "provenance",
    }
    missing = required - set(tech.meta)
    assert not missing, f"meta is missing {missing}"
    provenance = tech.meta["provenance"]
    for key in ("url", "downloaded_utc", "raw_sha256", "aggregation", "transform_version"):
        assert key in provenance, key
    assert tech.meta["unit"].startswith("million EUR")


# ------------------------------------------------------------------- Technology glue
def test_returned_object_is_a_core_technology(substrate):
    try:
        from planlab.core.state import Technology
    except Exception:  # pragma: no cover - core not written yet
        pytest.skip("planlab.core.state.Technology not available yet")
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    assert isinstance(tech, Technology)


def test_core_validator_accepts_the_arrays_unchanged(substrate):
    """``io_model.build_technology`` must not need to rescale our table."""
    try:
        from planlab.core import io_model
    except Exception:  # pragma: no cover
        pytest.skip("simulation core not available yet")
    builder = getattr(io_model, "build_technology", None)
    if builder is None:  # pragma: no cover
        pytest.skip("core exposes no build_technology")
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    rebuilt = builder(
        sectors=tech.sectors,
        A=tech.A,
        labor=tech.labor,
        kappa=tech.kappa,
        base_output=tech.base_output,
        base_final=tech.base_final,
        consumption_weights=tech.consumption_weights,
        essentials=tech.essentials,
        meta=dict(tech.meta),
    )
    np.testing.assert_allclose(rebuilt.A, tech.A, rtol=0, atol=0)
    np.testing.assert_allclose(rebuilt.base_final, tech.base_final)


# --------------------------------------------------------------------- offline path
def test_load_technology_is_offline(substrate):
    """The autouse fixture blocks sockets; loading must still succeed."""
    tech = load_technology(substrate.split("_")[0], 2020, 20)
    assert tech.n == 20


def test_missing_substrate_falls_back_to_synthetic_with_a_flag(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "metadata").mkdir(parents=True)
    for suffix in ("_A.csv", "_meta.json", "_labels.csv", "_vectors.csv"):
        shutil.copy(
            PROCESSED / f"synthetic_2020_n20{suffix}",
            root / "data" / "processed" / f"synthetic_2020_n20{suffix}",
        )
    tech = load_technology("eurostat", 2020, 20, offline_ok=True, root=root)
    assert tech.meta["substrate_fallback"] == "synthetic_2020_n20"
    assert tech.meta["requested_source"] == "eurostat"
    # The fallback must be loud, not silent.
    assert tech.meta["requested_key"] == "eurostat_2020_n20"


def test_missing_everything_raises_with_the_exact_command(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "processed").mkdir(parents=True)
    with pytest.raises(DataError, match="planlab.data.build"):
        load_technology("eurostat", 2020, 20, offline_ok=True, root=root)


def test_corrupt_matrix_is_rejected(tmp_path):
    root = tmp_path / "repo"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    for suffix in ("_A.csv", "_meta.json", "_labels.csv", "_vectors.csv"):
        shutil.copy(
            PROCESSED / f"eurostat_2020_n20{suffix}",
            processed / f"eurostat_2020_n20{suffix}",
        )
    # Break the accounting identity by halving one gross-output entry.
    meta = json.loads((processed / "eurostat_2020_n20_meta.json").read_text())
    meta["gross_output"][0] *= 0.5
    (processed / "eurostat_2020_n20_meta.json").write_text(json.dumps(meta))
    with pytest.raises(DataError, match="balance|column sums|violat"):
        load_technology("eurostat", 2020, 20, root=root)


# -------------------------------------------------------------------- idempotency
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_second_build_reuses_the_cached_raw_payload(tmp_path, monkeypatch):
    """A rebuild must read the cache, not the network, and be byte-identical."""
    root = tmp_path / "repo"
    raw_dir = root / "data" / "raw"
    meta_dir = root / "data" / "metadata"
    raw_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    shutil.copy(
        METADATA / "sector_map_eurostat64_to_20.csv",
        meta_dir / "sector_map_eurostat64_to_20.csv",
    )
    cached = next((REPO / "data" / "raw").glob("eurostat_naio_10_cp1700_2020_*_MIO_EUR_DOM.jsonstat"), None)
    if cached is None:
        pytest.skip("raw payload not in the local cache")
    copied = raw_dir / cached.name
    shutil.copy(cached, copied)
    shutil.copy(
        cached.with_suffix(cached.suffix + ".meta.json"),
        copied.with_suffix(copied.suffix + ".meta.json"),
    )
    raw_sha_before = _sha(copied)

    def _no_download(*args, **kwargs):
        raise AssertionError("the build must read the cache, not the network")

    monkeypatch.setattr(urllib.request, "urlopen", _no_download)

    first = build_eurostat(root=root)
    a_first = _sha(root / "data" / "processed" / "eurostat_2020_n20_A.csv")
    second = build_eurostat(root=root)
    a_second = _sha(root / "data" / "processed" / "eurostat_2020_n20_A.csv")

    assert first["meta"]["checks"]["totals"] == second["meta"]["checks"]["totals"]
    assert a_first == a_second
    assert _sha(copied) == raw_sha_before
    assert (root / "data" / "processed" / "eurostat_2020_n65_Z.csv").is_file()
    assert (root / "data" / "processed" / "eurostat_2020_n20_labels.csv").is_file()
    assert (root / "data" / "processed" / "eurostat_2020_n20_meta.json").is_file()


def test_build_without_any_cache_or_network_fails_loudly(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "data" / "raw").mkdir(parents=True)
    meta_dir = root / "data" / "metadata"
    meta_dir.mkdir(parents=True)
    shutil.copy(
        METADATA / "sector_map_eurostat64_to_20.csv",
        meta_dir / "sector_map_eurostat64_to_20.csv",
    )

    def _offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", _offline)
    from planlab.data import EurostatError

    with pytest.raises(EurostatError, match="could not download"):
        build_eurostat(root=root)


# ------------------------------------------------------------------- artifact shape
def test_processed_csv_layout_matches_the_schema():
    for key in SUBSTRATES:
        with (PROCESSED / f"{key}_A.csv").open() as handle:
            rows = list(csv.reader(handle))
        assert rows[0][0] == "sector"
        codes = rows[0][1:]
        meta = json.loads((PROCESSED / f"{key}_meta.json").read_text())
        assert codes == meta["sectors"]
        assert len(rows) - 1 == len(codes) == 20
        labels = list(csv.DictReader((PROCESSED / f"{key}_labels.csv").open()))
        assert [row["code"] for row in labels] == codes
        assert [int(row["sector_index"]) for row in labels] == list(range(20))


def test_artifact_key_matches_the_schema_naming():
    assert artifact_key("eurostat", 2020, 20) == "eurostat_2020_n20"
    assert artifact_key("synthetic", 2020, 20) == "synthetic_2020_n20"


def test_manifest_records_real_recorded_values():
    manifest_path = METADATA / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("manifest not written")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["eurostat"]["raw_sha256"]
    assert len(manifest["eurostat"]["raw_sha256"]) == 64
    assert manifest["aggregation"]["n_groups"] == 20
    assert manifest["identity_check_n65"]["passed"] is True
    assert manifest["checks_n20"]["all_column_sums_below_one"] is True
    assert manifest["substrates"], "the manifest must list the substrates"
    for key, entry in manifest["substrates"].items():
        assert (REPO / entry["meta_file"]).is_file(), key
        assert len(entry["a_sha256"]) == 64


def test_paths_points_into_the_repository():
    store = paths()
    assert (store.root / "src" / "planlab").is_dir()
    assert store.processed == store.root / "data" / "processed"
