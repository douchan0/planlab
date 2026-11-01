"""Artifact I/O, the public ``Technology`` loader, and the build orchestrator.

Two layers, deliberately separated (``docs/DATA_SCHEMA.md`` §4):

* :func:`load_technology` is **cheap and offline** — it reads the committed
  artefacts under ``data/processed/`` and never touches the network;
* :func:`build_all` / :func:`build_eurostat` do the network fetch, aggregation
  and manifest writing.  They are only reachable through the CLI
  (``python -m planlab.data.build``) or from tests.

``planlab.core.state.Technology`` is imported **lazily** inside
:func:`make_technology`, so the whole data layer — including the build and its
tests — works before the simulation core exists.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .aggregate import (
    GROUP_DEFINITIONS,
    AggregatedSIOT,
    SectorMap,
    aggregate_siot,
    consumption_weights,
    essentials,
    load_sector_map,
    sector_labels,
)
from .eurostat import (
    DATASET,
    DEFAULT_GEO,
    DEFAULT_STK_FLOW,
    DEFAULT_UNIT,
    EurostatError,
    SIOT,
    default_url,
    fetch_published_gva,
    load_siot,
    raw_cache_path,
    to_coefficients,
)
from .synthetic import DEFAULT_SEED, DEFAULT_SIGMA, generate_synthetic

__all__ = [
    "TRANSFORM_VERSION",
    "KAPPA_DEFAULT",
    "DataError",
    "paths",
    "available_substrates",
    "load_technology",
    "make_technology",
    "artifact_key",
    "read_substrate_meta",
    "build_eurostat",
    "build_synthetic",
    "build_all",
    "write_manifest",
]


class DataError(RuntimeError):
    """Raised for missing/corrupt artifacts or an unusable simulation core."""


#: Version of the numeric transform (bump when the construction changes).
TRANSFORM_VERSION = "1"

#: Flat capital-output ratio used when no by-industry capital stock is available
#: (``docs/DATA_SCHEMA.md`` §3.4 fallback; ``docs/PROTOCOL.md`` §4).
KAPPA_DEFAULT = 2.5


# ----------------------------------------------------------------------- paths
@dataclass(frozen=True)
class _Paths:
    """Repository paths used by the data layer."""

    root: Path
    src: Path
    data: Path
    raw: Path
    processed: Path
    metadata: Path

    def ensure(self) -> None:
        """Create the writable directories."""
        for directory in (self.raw, self.processed, self.metadata):
            directory.mkdir(parents=True, exist_ok=True)


def _repo_root() -> Path:
    """Locate the repository root, preferring a future ``planlab.paths`` helper."""
    try:  # pragma: no cover - depends on a not-yet-existing module
        from planlab import paths as core_paths  # type: ignore[attr-defined]

        for attribute in ("ROOT", "REPO_ROOT", "PROJECT_ROOT"):
            value = getattr(core_paths, attribute, None)
            if value:
                return Path(value)
    except Exception:  # noqa: BLE001 - optional module
        pass
    return Path(__file__).resolve().parents[3]


def _paths(root: Path | None = None) -> _Paths:
    root = Path(root) if root is not None else _repo_root()
    return _Paths(
        root=root,
        src=root / "src",
        data=root / "data",
        raw=root / "data" / "raw",
        processed=root / "data" / "processed",
        metadata=root / "data" / "metadata",
    )


paths = _paths


# ------------------------------------------------------------------- artifacts
def artifact_key(source: str, year: int, n_sectors: int) -> str:
    """Return the artifact stem, e.g. ``eurostat_2020_n20``."""
    return f"{source}_{int(year)}_n{int(n_sectors)}"


def available_substrates(root: Path | None = None) -> list[str]:
    """List the substrates that have a complete artifact set on disk.

    Returns:
        Sorted keys such as ``['eurostat_2020_n20', 'synthetic_2020_n20']``.
    """
    processed = paths(root).processed
    if not processed.is_dir():
        return []
    found: list[str] = []
    for meta_file in sorted(processed.glob("*_n*_meta.json")):
        key = meta_file.name[: -len("_meta.json")]
        if (processed / f"{key}_A.csv").is_file() and (processed / f"{key}_labels.csv").is_file():
            found.append(key)
    return found


def read_substrate_meta(key: str, root: Path | None = None) -> dict[str, Any]:
    """Read ``<key>_meta.json``.

    Raises:
        DataError: if the file is missing or is not valid JSON.
    """
    path = paths(root).processed / f"{key}_meta.json"
    if not path.is_file():
        raise DataError(f"missing artifact {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON: {exc}") from exc


def _write_matrix_csv(path: Path, codes: Sequence[str], matrix: np.ndarray) -> None:
    """Write a labelled square matrix CSV (first column = row sector code)."""
    import csv

    matrix = np.asarray(matrix, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sector", *codes])
        for code, row in zip(codes, matrix):
            writer.writerow([code, *[f"{value:.10g}" for value in row]])


def _read_matrix_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a labelled square matrix CSV written by :func:`_write_matrix_csv`."""
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        codes = header[1:]
        rows: list[list[float]] = []
        row_codes: list[str] = []
        for row in reader:
            if not row:
                continue
            row_codes.append(row[0])
            rows.append([float(value) for value in row[1:]])
    if row_codes != codes:
        raise DataError(f"{path}: row codes {row_codes[:3]}... != column codes {codes[:3]}...")
    matrix = np.array(rows, dtype=float)
    if matrix.shape != (len(codes), len(codes)):
        raise DataError(f"{path}: expected a square matrix, got {matrix.shape}")
    return codes, matrix


def _write_vectors_csv(path: Path, codes: Sequence[str], **vectors: np.ndarray) -> None:
    """Write named length-n vectors as CSV columns."""
    import csv

    names = list(vectors)
    columns = [np.asarray(vectors[name], dtype=float) for name in names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sector", *names])
        for i, code in enumerate(codes):
            writer.writerow([code, *[f"{col[i]:.10g}" for col in columns]])


def _read_vectors_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a vectors CSV written by :func:`_write_vectors_csv`."""
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise DataError(f"{path} is empty")
    out: dict[str, np.ndarray] = {}
    for name in rows[0]:
        if name == "sector":
            continue
        out[name] = np.array([float(row[name]) for row in rows], dtype=float)
    return out


def _write_labels_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ``labels.csv``."""
    import csv

    fieldnames = ["sector_index", "code", "name_en", "name_zh", "aggregate", "essential", "n_source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def _read_labels_csv(path: Path) -> list[dict[str, Any]]:
    """Read ``labels.csv`` back into a list of dicts."""
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------- Technology glue
def _technology_class() -> type:
    """Import ``planlab.core.state.Technology`` lazily.

    Raises:
        DataError: if the simulation core has not been written yet, or does not
            expose ``Technology``.
    """
    try:
        from planlab.core.state import Technology  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - core may legitimately be absent
        raise DataError(
            "planlab.core.state.Technology is not importable; the data layer can build "
            "and validate artifacts without it, but load_technology needs the real class. "
            f"Underlying error: {exc}"
        ) from exc
    return Technology


def make_technology(
    *,
    sectors: Sequence[str],
    A: np.ndarray,
    labor: np.ndarray,
    kappa: np.ndarray,
    consumption_weights_: np.ndarray,
    essentials_: np.ndarray,
    base_output: np.ndarray,
    base_final: np.ndarray,
    meta: Mapping[str, Any],
) -> Any:
    """Build a ``planlab.core.state.Technology`` from validated arrays.

    Returns:
        An instance of the real ``Technology`` dataclass.

    Raises:
        DataError: if the real class is unavailable or its field names differ
            from ``docs/ARCHITECTURE.md`` §3.
    """
    Technology = _technology_class()
    payload = {
        "sectors": tuple(str(s) for s in sectors),
        "A": np.asarray(A, dtype=float),
        "labor": np.asarray(labor, dtype=float),
        "kappa": np.asarray(kappa, dtype=float),
        "consumption_weights": np.asarray(consumption_weights_, dtype=float),
        "essentials": np.asarray(essentials_, dtype=float),
        "base_output": np.asarray(base_output, dtype=float),
        "base_final": np.asarray(base_final, dtype=float),
        "meta": dict(meta),
    }
    try:
        return Technology(**payload)
    except TypeError as exc:
        raise DataError(
            "could not construct Technology with the documented fields "
            f"({sorted(payload)}); the core contract may have changed: {exc}"
        ) from exc


def _validate_arrays(
    sectors: Sequence[str],
    A: np.ndarray,
    labor: np.ndarray,
    kappa: np.ndarray,
    base_output: np.ndarray,
    base_final: np.ndarray,
    *,
    expected_n: int,
) -> None:
    """Assert the invariants of ``docs/DATA_SCHEMA.md`` §3.8.

    Raises:
        DataError: on any violation.
    """
    n = len(sectors)
    if n != expected_n:
        raise DataError(f"expected n={expected_n} sectors, found {n}")
    if A.shape != (n, n):
        raise DataError(f"A has shape {A.shape}, expected {(n, n)}")
    for name, array in (
        ("labor", labor),
        ("kappa", kappa),
        ("base_output", base_output),
        ("base_final", base_final),
    ):
        if np.asarray(array).shape != (n,):
            raise DataError(f"{name} has shape {np.asarray(array).shape}, expected {(n,)}")
    for name, array in (("A", A), ("labor", labor), ("base_output", base_output), ("base_final", base_final)):
        if not np.all(np.isfinite(np.asarray(array, dtype=float))):
            raise DataError(f"{name} contains NaN or infinity")
    if (A < -1e-12).any() or (A > 1.0 + 1e-12).any():
        raise DataError(f"A has entries outside [0, 1]: min={A.min()}, max={A.max()}")
    residual = A @ np.asarray(base_output, dtype=float) + np.asarray(base_final, dtype=float) - np.asarray(base_output, dtype=float)
    scale = max(float(np.max(np.abs(base_output))), 1.0)
    if float(np.max(np.abs(residual))) > 1e-6 * scale:
        raise DataError(
            "base-year balance A @ base_output + base_final - base_output is violated "
            f"(max |residual| = {np.max(np.abs(residual)):.6g} MIO EUR); base_final must be "
            "total final demand (x - A x), as planlab.core.io_model.build_technology requires"
        )
    column_sums = A.sum(axis=0)
    if (column_sums >= 1.0).any():
        bad = sectors[int(np.argmax(column_sums))]
        raise DataError(
            f"sum_i A[{bad}] = {column_sums.max():.6f} >= 1; the table is not value-additive"
        )


# ------------------------------------------------------------------- public API
def load_technology(
    source: str = "eurostat",
    year: int = 2020,
    n_sectors: int = 20,
    *,
    offline_ok: bool = True,
    root: Path | None = None,
) -> Any:
    """Load a substrate from ``data/processed/`` and return a ``Technology``.

    This function is **offline by construction** — it never performs a network
    request.  If the requested substrate has not been built and ``offline_ok``
    is true, it falls back to the committed ``synthetic`` substrate of the same
    year/size and flags the substitution in ``Technology.meta``.

    Args:
        source: ``"eurostat"`` or ``"synthetic"``.
        year: Reference year.
        n_sectors: Number of sectors.
        offline_ok: Allow the documented synthetic fallback.
        root: Repository root override (tests only).

    Returns:
        ``planlab.core.state.Technology``.

    Raises:
        DataError: if neither the requested nor the fallback artifacts exist,
            or if the loaded artifacts violate the schema invariants.
    """
    store = paths(root)
    key = artifact_key(source, year, n_sectors)
    fallback_key = artifact_key("synthetic", year, n_sectors)
    meta: dict[str, Any]
    if (store.processed / f"{key}_meta.json").is_file() and (store.processed / f"{key}_A.csv").is_file():
        meta = read_substrate_meta(key, root)
    elif offline_ok and source != "synthetic" and (
        (store.processed / f"{fallback_key}_meta.json").is_file()
        and (store.processed / f"{fallback_key}_A.csv").is_file()
    ):
        meta = read_substrate_meta(fallback_key, root)
        meta = dict(meta)
        meta["requested_source"] = source
        meta["requested_key"] = key
        meta["substrate_fallback"] = fallback_key
        key = fallback_key
    else:
        raise DataError(
            f"no artifacts for {key!r} under {store.processed}. "
            f"Run: python -m planlab.data.build --source {source}"
        )

    codes, A = _read_matrix_csv(store.processed / f"{key}_A.csv")
    sectors = [str(s) for s in meta["sectors"]]
    if codes != sectors:
        raise DataError(f"{key}: A.csv column order {codes[:3]}... != meta sectors {sectors[:3]}...")

    labor = np.asarray(meta["labor"], dtype=float)
    kappa = np.asarray(meta["kappa"], dtype=float)
    weights = np.asarray(meta["consumption_weights"], dtype=float)
    essential = np.asarray(meta["essentials"], dtype=float)
    base_output = np.asarray(meta["gross_output"], dtype=float)
    # Precedence: explicit base_final -> total final demand -> household column.
    # The first two are equivalent by construction; the third applies only to a
    # legacy artifact that predates the field.
    for key in ("base_final", "final_demand_total"):
        if key in meta:
            base_final = np.asarray(meta[key], dtype=float)
            break
    else:
        base_final = np.asarray(meta["final_consumption"], dtype=float)

    _validate_arrays(sectors, A, labor, kappa, base_output, base_final, expected_n=int(n_sectors))
    return make_technology(
        sectors=sectors,
        A=A,
        labor=labor,
        kappa=kappa,
        consumption_weights_=weights,
        essentials_=essential,
        base_output=base_output,
        base_final=base_final,
        meta=meta,
    )


# ----------------------------------------------------------------------- build
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _versions() -> dict[str, str]:
    import pandas

    return {
        "planlab": "0.1.0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sys_platform": sys.platform,
    }


def build_eurostat(
    *,
    year: int = 2020,
    n_sectors: int = 20,
    geo: str = DEFAULT_GEO,
    unit: str = DEFAULT_UNIT,
    stk_flow: str = DEFAULT_STK_FLOW,
    force: bool = False,
    root: Path | None = None,
    mapping_path: Path | None = None,
) -> dict[str, Any]:
    """Fetch/parse the SIOT, aggregate it, and write both artifact sets.

    Writes ``<key>_n65`` (audit trail: the raw 65-branch value block) and
    ``<key>_n20`` (the substrate the simulation consumes).

    Returns:
        A summary dict used by :func:`write_manifest`.

    Raises:
        DataError: on a mapping/table mismatch or a violated invariant.
    """
    store = paths(root)
    store.ensure()
    mapping_file = mapping_path or (store.metadata / "sector_map_eurostat64_to_20.csv")
    if not mapping_file.is_file():
        raise DataError(f"missing sector mapping {mapping_file}")

    smap: SectorMap = load_sector_map(mapping_file)
    siot: SIOT = load_siot(store.raw, year=year, geo=geo, unit=unit, stk_flow=stk_flow, force=force)
    missing = [c for c in siot.codes if c not in set(smap.code64)]
    if missing:
        raise DataError(
            f"{len(missing)} source sectors have no mapping entry: {missing[:8]}. "
            f"Update {mapping_file}."
        )

    # --- 65-branch audit artifacts -----------------------------------------
    key65 = artifact_key("eurostat", year, siot.n)
    _write_matrix_csv(store.processed / f"{key65}_Z.csv", siot.codes, siot.Z)
    A65 = to_coefficients(siot.Z, siot.x)
    residual = A65 @ siot.x + siot.y_total - siot.x
    scale = max(float(np.max(np.abs(siot.x))), 1.0)
    _write_vectors_csv(
        store.processed / f"{key65}_vectors.csv",
        siot.codes,
        gross_output=siot.x,
        final_demand_total=siot.y_total,
        household=siot.household,
        government=siot.government,
        gfcf=siot.gfcf,
        inventories=siot.inventories,
        exports=siot.exports,
        imports=siot.imports,
        compensation=siot.compensation,
        gva=siot.gva,
        intermediate_consumption=siot.intermediate_consumption,
    )
    meta65 = {
        "source": "eurostat",
        "dataset": DATASET,
        "year": int(year),
        "n_sectors": int(siot.n),
        "unit": "million EUR, basic prices, current prices",
        "price_basis": siot.price_basis,
        "stk_flow": siot.stk_flow,
        "geo": geo,
        "table_type": "symmetric",
        "sectors": list(siot.codes),
        "names": {c: siot.names.get(c, "") for c in siot.codes},
        "checks": {
            "identity_max_abs_residual": float(np.max(np.abs(residual))),
            "identity_max_rel_residual": float(np.max(np.abs(residual)) / scale),
            "a_column_sums_max": float(A65.sum(axis=0).max()),
            "a_column_sums_min": float(A65.sum(axis=0).min()),
            "n_zero_output_sectors": int((siot.x <= 0).sum()),
            "zero_output_sectors": [c for c, out in zip(siot.codes, siot.x) if out <= 0],
            "defined_cells": siot.provenance.get("defined_cells"),
            "total_cells": siot.provenance.get("total_cells"),
        },
        "provenance": dict(siot.provenance),
    }
    (store.processed / f"{key65}_meta.json").write_text(
        json.dumps(meta65, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- 20-sector substrate ------------------------------------------------
    agg = aggregate_siot(siot, smap)
    key20 = artifact_key("eurostat", year, n_sectors)
    meta20 = _write_substrate(
        store=store,
        key=key20,
        source="eurostat",
        year=year,
        geo=geo,
        agg=agg,
        smap=smap,
        siot_provenance=dict(siot.provenance),
        checks={
            "identity_max_abs_residual_n65": float(np.max(np.abs(residual))),
            "identity_max_rel_residual_n65": float(np.max(np.abs(residual)) / scale),
            "n65_a_column_sums_max": float(A65.sum(axis=0).max()),
        },
    )
    return {
        "key": key20,
        "key65": key65,
        "meta": meta20,
        "aggregated": agg,
        "siot": siot,
        "mapping_file": str(mapping_file),
        "mapping_sha256": _sha256(mapping_file),
    }


def build_synthetic(
    *,
    reference: AggregatedSIOT | None = None,
    year: int = 2020,
    n_sectors: int = 20,
    seed: int = DEFAULT_SEED,
    sigma: float = DEFAULT_SIGMA,
    force: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Build the synthetic substrate calibrated to the Eurostat aggregate.

    Args:
        reference: The real aggregated table; when ``None`` it is re-read from
            the committed ``eurostat_<year>_n<n>`` artifacts.
        year: Reference year.
        n_sectors: Number of sectors.
        seed: RNG seed.
        sigma: Log-noise scale of the random topology.
        force: Rebuild even if artifacts exist.
        root: Repository root override (tests only).

    Returns:
        A summary dict used by :func:`write_manifest`.

    Raises:
        DataError: if no calibration reference is available.
    """
    store = paths(root)
    store.ensure()
    key = artifact_key("synthetic", year, n_sectors)
    mapping_file = store.metadata / "sector_map_eurostat64_to_20.csv"
    smap = load_sector_map(mapping_file)

    if reference is None:
        reference = _read_aggregated(store=store, source="eurostat", year=year, n_sectors=n_sectors)

    result = generate_synthetic(reference, seed=seed, sigma=sigma)
    meta = _write_substrate(
        store=store,
        key=key,
        source="synthetic",
        year=year,
        geo="EU27_2020 (calibration reference only)",
        agg=result.table,
        smap=smap,
        siot_provenance={},
        checks={
            "identity_max_abs_residual_n20": float(
                np.max(np.abs(result.table.A() @ result.table.x + result.table.y_total - result.table.x))
            ),
        },
        extra={
            "synthetic": dict(result.calibration),
            "calibration_reference": artifact_key("eurostat", year, n_sectors),
        },
    )
    return {"key": key, "meta": meta, "aggregated": result.table}


def _read_aggregated(*, store: _Paths, source: str, year: int, n_sectors: int) -> AggregatedSIOT:
    """Reconstruct an :class:`AggregatedSIOT` from committed artifacts."""
    key = artifact_key(source, year, n_sectors)
    meta = read_substrate_meta(key, store.root)
    codes, A = _read_matrix_csv(store.processed / f"{key}_A.csv")
    if codes != list(meta["sectors"]):
        raise DataError(f"{key}: A.csv and meta sectors disagree")
    x = np.asarray(meta["gross_output"], dtype=float)
    Z = A * x[None, :]
    vectors = _read_vectors_csv(store.processed / f"{key}_vectors.csv")
    return AggregatedSIOT(
        sectors=tuple(codes),
        Z=Z,
        x=x,
        y_total=vectors["final_demand_total"],
        household=vectors["household"],
        government=vectors["government"],
        gfcf=vectors["gfcf"],
        inventories=vectors["inventories"],
        exports=vectors["exports"],
        imports=vectors["imports"],
        compensation=vectors["compensation"],
        gva=vectors["gva"],
        intermediate_consumption=vectors["intermediate_consumption"],
        n_source=int(meta.get("n_source_sectors", 0)),
    )


def _write_substrate(
    *,
    store: _Paths,
    key: str,
    source: str,
    year: int,
    geo: str,
    agg: AggregatedSIOT,
    smap: SectorMap,
    siot_provenance: Mapping[str, Any],
    checks: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``A.csv`` / ``meta.json`` / ``labels.csv`` / ``<key>_vectors.csv``."""
    sectors = list(agg.sectors)
    A = agg.A()
    labor = np.divide(agg.compensation, agg.x, out=np.zeros(agg.n), where=agg.x > 0)
    kappa = np.full(agg.n, KAPPA_DEFAULT, dtype=float)
    weights = consumption_weights(smap)
    essential = essentials(smap)

    _write_matrix_csv(store.processed / f"{key}_A.csv", sectors, A)
    _write_vectors_csv(
        store.processed / f"{key}_vectors.csv",
        sectors,
        gross_output=agg.x,
        final_demand_total=agg.y_total,
        household=agg.household,
        government=agg.government,
        gfcf=agg.gfcf,
        inventories=agg.inventories,
        exports=agg.exports,
        imports=agg.imports,
        compensation=agg.compensation,
        gva=agg.gva,
        intermediate_consumption=agg.intermediate_consumption,
    )
    _write_labels_csv(store.processed / f"{key}_labels.csv", sector_labels(smap))

    column_sums = A.sum(axis=0)
    # A productive Leontief matrix (spectral radius < 1) is the precondition for
    # the engine's short-side fixed point to have a finite solution.
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A)))) if agg.n else 0.0
    meta: dict[str, Any] = {
        "source": source,
        "year": int(year),
        "n_sectors": int(agg.n),
        "unit": "million EUR, basic prices, current prices",
        "price_basis": "basic",
        "table_type": "symmetric",
        "sectors": sectors,
        "gross_output": [float(v) for v in agg.x],
        # DATA_SCHEMA 2 requires the household final consumption column here.
        "final_consumption": [float(v) for v in agg.household],
        # Technology.base_final must satisfy A x + y = x (core contract, see
        # planlab/core/io_model.py and docs/ARCHITECTURE.md 3), which is total
        # final demand, not household consumption alone.  Both are recorded.
        "base_final": [float(v) for v in agg.y_total],
        "base_final_basis": "total final demand (x - A x)",
        "labor": [float(v) for v in labor],
        "kappa": [float(v) for v in kappa],
        "consumption_weights": [float(v) for v in weights],
        "essentials": [float(v) for v in essential],
        "provenance": {
            "url": siot_provenance.get("url")
            or default_url(year=year, geo=geo if geo.startswith("EU") else DEFAULT_GEO),
            "downloaded_utc": siot_provenance.get("downloaded_utc"),
            "raw_sha256": siot_provenance.get("sha256"),
            "aggregation": "64->20 per data/metadata/sector_map_eurostat64_to_20.csv",
            "transform_version": TRANSFORM_VERSION,
            "geo": geo,
            "n_source_sectors": int(agg.n_source),
            "mapping_sha256": _sha256(store.metadata / "sector_map_eurostat64_to_20.csv"),
        },
        "final_demand_total": [float(v) for v in agg.y_total],
        "compensation_of_employees": [float(v) for v in agg.compensation],
        "gross_value_added": [float(v) for v in agg.gva],
        "intermediate_consumption": [float(v) for v in agg.intermediate_consumption],
        "imports": [float(v) for v in agg.imports],
        "gfcf": [float(v) for v in agg.gfcf],
        "exports": [float(v) for v in agg.exports],
        "labor_fallback": False,
        "kappa_fallback": True,
        "kappa_fallback_note": (
            "naio_10_cp1700 has no by-industry capital stock or GFCF by industry "
            "(P51C is empty in this release), so kappa is the documented flat 2.5 "
            "from docs/DATA_SCHEMA.md 3.4 / docs/PROTOCOL.md 4."
        ),
        "checks": {
            **dict(checks),
            "a_column_sums_max": float(column_sums.max()),
            "a_column_sums_min": float(column_sums.min()),
            "all_column_sums_below_one": bool((column_sums < 1.0).all()),
            "spectral_radius": spectral_radius,
            "productive": bool(spectral_radius < 1.0),
            "aggregate_intermediate_share": float((column_sums @ agg.x) / agg.x.sum()),
            "finite": bool(np.isfinite(A).all()),
            "n_sectors_ok": bool(agg.n == 20),
            "totals": {
                "gross_output": float(agg.x.sum()),
                "intermediate_uses": float(agg.Z.sum()),
                "final_demand": float(agg.y_total.sum()),
                "household_consumption": float(agg.household.sum()),
                "compensation_of_employees": float(agg.compensation.sum()),
                "gross_value_added": float(agg.gva.sum()),
            },
        },
    }
    if extra:
        meta.update(dict(extra))
    _validate_arrays(sectors, A, labor, kappa, agg.x, agg.y_total, expected_n=agg.n)
    (store.processed / f"{key}_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def write_manifest(
    *,
    year: int = 2020,
    n_sectors: int = 20,
    root: Path | None = None,
) -> dict[str, Any]:
    """Assemble ``data/metadata/manifest.json`` from the artifacts on disk.

    Every recorded number is read back from a file written earlier in the same
    build; nothing is recomputed in a way that could silently disagree.

    Returns:
        The manifest dict that was written.
    """
    store = paths(root)
    store.ensure()
    key20 = artifact_key("eurostat", year, n_sectors)
    key65 = artifact_key("eurostat", year, 65)
    raw = raw_cache_path(store.raw, year=year, geo=DEFAULT_GEO, unit=DEFAULT_UNIT, stk_flow=DEFAULT_STK_FLOW)
    mapping_file = store.metadata / "sector_map_eurostat64_to_20.csv"

    meta20 = read_substrate_meta(key20, root)
    meta65_path = store.processed / f"{key65}_meta.json"
    meta65 = json.loads(meta65_path.read_text()) if meta65_path.is_file() else {}
    synthetic_meta_path = store.processed / f"{artifact_key('synthetic', year, n_sectors)}_meta.json"
    synthetic_meta = (
        json.loads(synthetic_meta_path.read_text()) if synthetic_meta_path.is_file() else {}
    )
    raw_meta_path = raw.with_suffix(raw.suffix + ".meta.json")
    raw_meta = json.loads(raw_meta_path.read_text()) if raw_meta_path.is_file() else {}

    gva_implied = float(meta20["checks"]["totals"]["gross_value_added"])
    crosscheck = fetch_published_gva(year=year)
    validation: dict[str, Any] = {"published_gva": crosscheck}
    if crosscheck.get("status") == "ok":
        published = float(crosscheck["value"])
        validation["implied_gva_total_mio_eur"] = gva_implied
        validation["published_gva_total_mio_eur"] = published
        validation["relative_deviation"] = (gva_implied - published) / published
        validation["verdict"] = (
            "consistent" if abs((gva_implied - published) / published) < 0.01 else "check needed"
        )
    else:
        validation["verdict"] = "cross-check unavailable (offline?)"

    manifest: dict[str, Any] = {
        "transform_version": TRANSFORM_VERSION,
        "generated_utc": _now(),
        "versions": _versions(),
        "dataset": DATASET,
        "year": int(year),
        "n_sectors": int(n_sectors),
        "n_source_sectors": int(meta20["provenance"]["n_source_sectors"]),
        "eurostat": {
            "url": raw_meta.get("url") or meta20["provenance"]["url"],
            "raw_sha256": raw_meta.get("sha256") or meta20["provenance"]["raw_sha256"],
            "raw_bytes": raw_meta.get("bytes"),
            "downloaded_utc": raw_meta.get("downloaded_utc"),
            "dataset_label": meta65.get("provenance", {}).get("label"),
            "dataset_updated": meta65.get("provenance", {}).get("updated"),
            "datastructure": meta65.get("provenance", {}).get("datastructure"),
            "id": meta65.get("provenance", {}).get("id"),
            "size": meta65.get("provenance", {}).get("size"),
            "defined_cells": meta65.get("provenance", {}).get("defined_cells"),
            "total_cells": meta65.get("provenance", {}).get("total_cells"),
            "status_counts": meta65.get("provenance", {}).get("status_counts"),
            "filters": {
                "unit": DEFAULT_UNIT,
                "stk_flow": DEFAULT_STK_FLOW,
                "geo": DEFAULT_GEO,
                "time": str(year),
                "freq": "A",
            },
        },
        "aggregation": {
            "mapping_file": "data/metadata/sector_map_eurostat64_to_20.csv",
            "mapping_sha256": _sha256(mapping_file),
            "n_groups": len(GROUP_DEFINITIONS),
            "n_source_sectors": int(meta20["provenance"]["n_source_sectors"]),
            "groups": [
                {"code": row["code"], "n_source": int(row["n_source"]), "aggregate": row["aggregate"]}
                for row in _read_labels_csv(store.processed / f"{key20}_labels.csv")
            ],
        },
        "identity_check_n65": {
            "max_abs_residual": meta65.get("checks", {}).get("identity_max_abs_residual"),
            "max_rel_residual": meta65.get("checks", {}).get("identity_max_rel_residual"),
            "tolerance_rel": 1e-6,
            "passed": bool(
                (meta65.get("checks", {}).get("identity_max_rel_residual") or 1.0) <= 1e-6
            ),
        },
        "checks_n20": meta20["checks"],
        "validation": validation,
        "substrates": {
            key: {
                "meta_file": f"data/processed/{key}_meta.json",
                "a_sha256": _sha256(store.processed / f"{key}_A.csv"),
                "labels_sha256": _sha256(store.processed / f"{key}_labels.csv"),
            }
            for key in available_substrates(root)
        },
        "synthetic": synthetic_meta.get("synthetic", {}),
        "files": sorted(
            str(p.relative_to(store.root)) for p in store.processed.glob("*") if p.is_file()
        ),
    }
    (store.metadata / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_all(
    *,
    year: int = 2020,
    n_sectors: int = 20,
    force: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Fetch, aggregate, generate the synthetic substrate, and write the manifest.

    Args:
        year: Reference year.
        n_sectors: Target sector count.
        force: Re-download / rebuild even if artifacts and cache exist.
        root: Repository root override (tests only).

    Returns:
        ``{"eurostat": {...}, "synthetic": {...}, "manifest": {...}}``

    Raises:
        DataError: if any stage violates the documented invariants.
    """
    eurostat_summary = build_eurostat(year=year, n_sectors=n_sectors, force=force, root=root)
    synthetic_summary = build_synthetic(
        reference=eurostat_summary["aggregated"],
        year=year,
        n_sectors=n_sectors,
        force=force,
        root=root,
    )
    manifest = write_manifest(year=year, n_sectors=n_sectors, root=root)
    return {"eurostat": eurostat_summary, "synthetic": synthetic_summary, "manifest": manifest}
