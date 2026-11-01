"""Eurostat symmetric input-output table ingestion (``naio_10_cp1700``).

Dataset
-------
``naio_10_cp1700`` = *Symmetric input-output table at basic prices (product by
product)*.  For ``geo=EU27_2020`` and ``time=2020`` the API returns a single
table in million euro.

Observed JSON-stat layout (read from the response, not assumed)
--------------------------------------------------------------
``id = [freq, unit, stk_flow, prd_use, prd_ava, geo, time]`` and
``size = [1, 1, 1, 121, 123, 1, 1]`` for the filtered query.  Only 5 712 of the
14 883 cells are defined (JSON-stat sparse encoding); every remaining cell is
genuinely undefined rather than zero.

The two structural dimensions are *not* oriented the way the member names
suggest.  Empirically (see ``docs/DATA_NOTES.md`` for the evidence):

* ``prd_use`` (121 categories) enumerates the **using** side: the 65
  production branches ("products" in a product-by-product table) plus the
  final-use aggregates ``P3``/``P5``/``P6``/``TU``/``TFU``.  Value-added items
  are *not* here.
* ``prd_ava`` (123 categories) enumerates the **supplied** side: the same 65
  products plus the value-added items (``B1G``, ``D1``, ...), the adjustment
  columns (``IMP``, ``P2_ADJ``) and the supply total ``TS_BP``.

So a cell ``X[prd_use=r, prd_ava=c]`` reads "amount of ``c`` used by ``r``".
Consequently the Leontief use matrix is the *transpose* of the slice:
``Z[product i, industry j] = X[prd_use=j, prd_ava=i]``.  Three independent
checks confirm this orientation:

1. ``sum_users X[r, c] + final_use(c) == P1(c)`` for every product, to 0.09 MIO
   EUR on a 25 143 704 MIO EUR base (publication rounding, relative 3.6e-9).
   The alternative orientation is off by up to 4.8e5 MIO EUR per sector.
2. The implied gross value added totals 12 147 083 MIO EUR, i.e. 0.18 % below
   Eurostat's published EU27 2020 GVA (``nama_10_gdp``, ``na_item=B1G``,
   12 168 846 MIO EUR).
3. The implied sectoral labour shares are economically plausible (coke and
   refined petroleum 4.8 %, imputed rents 0.1 %, employment services 70.8 %,
   education 66.9 %).

Public API
----------
``default_url(year, ...)``      canonical API URL (recorded in the manifest).
``fetch_raw(...)``              download / re-use the cached raw payload.
``parse_siot(...)``             JSON-stat -> :class:`SIOT` (65 x 65 value block).
``load_siot(...)``              fetch-or-cache + parse in one call.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .jsonstat import JsonStat, JsonStatError

__all__ = [
    "DATASET",
    "DEFAULT_GEO",
    "DEFAULT_UNIT",
    "DEFAULT_STK_FLOW",
    "SIOT",
    "EurostatError",
    "default_url",
    "fetch_raw",
    "parse_siot",
    "load_siot",
    "raw_cache_path",
]

DATASET = "naio_10_cp1700"
API_ROOT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
DEFAULT_GEO = "EU27_2020"
DEFAULT_UNIT = "MIO_EUR"
DEFAULT_STK_FLOW = "DOM"
REQUEST_TIMEOUT = 180

#: Dimensions whose categories are *not* production branches or products.
_NON_PRODUCT_CODES = {
    "TOTAL",
    "CPA_TOTAL",
    "B1G",
    "B2A3N",
    "B2A3G",
    "B3G",
    "D1",
    "D11",
    "D21X31",
    "D29X39",
    "P1",
    "P2",
    "P2_ADJ",
    "P7",
    "P7_B0",
    "P7_D0",
    "P7_U2",
    "P7_U3",
    "P51C",
    "IMP",
    "TS_BP",
}

class EurostatError(RuntimeError):
    """Raised when the API payload cannot be turned into a SIOT."""


@dataclass(frozen=True)
class SIOT:
    """A symmetric input-output table in constant-price value units.

    All arrays are in million euro at basic prices and share the same sector
    order (:attr:`codes`).

    Attributes:
        codes: Product/industry codes, e.g. ``("CPA_A01", ...)`` (n,).
        names: ``{code: English label}`` from the payload.
        Z: Intermediate uses, ``Z[i, j]`` = product *i* used by industry *j*
            (n, n), million euro.
        x: Gross output by industry (= ``P1``), (n,).
        y_total: Total final use by product (= household + government + gross
            capital formation + changes in inventories + exports), (n,).
        household: Household final consumption (``P3_S14``) by product, (n,).
        government: Government final consumption (``P3_S13``), (n,).
        gfcf: Gross fixed capital formation (``P51G``), (n,).
        inventories: Changes in inventories and valuables (``P52`` + ``P53``),
            (n,).
        exports: Exports of goods and services (``P6``), (n,).
        imports: Imports of goods and services (``P7``), (n,).
        compensation: Compensation of employees (``D1``) by industry, (n,).
        gva: Gross value added at basic prices (``B1G``) by industry, (n,).
        intermediate_consumption: ``P2_ADJ`` by industry, (n,).
        unit: Unit string of the values.
        price_basis: ``"basic"``.
        stk_flow: ``"DOM"`` (domestic uses) or ``"TOTAL"``.
        provenance: Free-form provenance dict (URL, sha256, timestamps, ...).
    """

    codes: tuple[str, ...]
    names: Mapping[str, str]
    Z: np.ndarray
    x: np.ndarray
    y_total: np.ndarray
    household: np.ndarray
    government: np.ndarray
    gfcf: np.ndarray
    inventories: np.ndarray
    exports: np.ndarray
    imports: np.ndarray
    compensation: np.ndarray
    gva: np.ndarray
    intermediate_consumption: np.ndarray
    unit: str
    price_basis: str
    stk_flow: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of production branches in the table."""
        return len(self.codes)

    def A(self) -> np.ndarray:
        """Direct input coefficients ``a[i, j] = Z[i, j] / x[j]``.

        Branches with zero gross output get a zero column (documented rule in
        ``docs/DATA_SCHEMA.md`` §3.1) — in this table that is only
        ``CPA_U`` (extraterritorial organisations), which has no output.
        """
        return to_coefficients(self.Z, self.x)

    def output_residual(self) -> np.ndarray:
        """``A x + y - x`` per product; zero in exact accounting."""
        return self.A() @ self.x + self.y_total - self.x


# --------------------------------------------------------------------- network
def default_url(
    *,
    year: int = 2020,
    geo: str = DEFAULT_GEO,
    unit: str = DEFAULT_UNIT,
    stk_flow: str = DEFAULT_STK_FLOW,
) -> str:
    """Return the canonical Eurostat dissemination API URL for the SIOT.

    Args:
        year: Reference year (``time`` filter).
        geo: Reporting geo code; ``EU27_2020`` is the EU-27 aggregate.
        unit: ``MIO_EUR`` (million euro) or ``MIO_NAC``.
        stk_flow: ``DOM`` (domestic uses) or ``TOTAL`` (domestic + imported).

    Returns:
        Fully-formed URL with an explicit, documented query string.
    """
    query = {
        "format": "JSON",
        "lang": "EN",
        "freq": "A",
        "geo": geo,
        "time": str(year),
        "unit": unit,
        "stk_flow": stk_flow,
    }
    return f"{API_ROOT}/{DATASET}?{urllib.parse.urlencode(query)}"


def raw_cache_path(raw_dir: Path, *, year: int, geo: str, unit: str, stk_flow: str) -> Path:
    """Return the deterministic cache path for a raw JSON-stat payload."""
    return Path(raw_dir) / f"eurostat_{DATASET}_{year}_{geo}_{unit}_{stk_flow}.jsonstat"


def fetch_raw(
    url: str,
    cache_path: Path,
    *,
    force: bool = False,
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Download the payload, or reuse the cached copy.

    Args:
        url: URL returned by :func:`default_url`.
        cache_path: Destination for the raw bytes (``data/raw/...jsonstat``).
        force: Re-download even if a cached file exists.
        timeout: Socket timeout in seconds.

    Returns:
        Provenance dict with ``sha256``, ``bytes``, ``downloaded_utc``,
        ``from_cache`` and ``url``.

    Raises:
        EurostatError: on network failure when no cache is available.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        payload = cache_path.read_bytes()
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
        downloaded = None
        if meta_path.exists():
            try:
                downloaded = json.loads(meta_path.read_text())["downloaded_utc"]
            except (OSError, ValueError, KeyError):  # pragma: no cover - defensive
                downloaded = None
        return {
            "url": url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "downloaded_utc": downloaded,
            "from_cache": True,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "planlab/0.1 (+data layer)", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EurostatError(f"could not download {url}: {exc}") from exc
    if not payload:
        raise EurostatError(f"empty response from {url}")

    stamp = _utc_now()
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(cache_path)
    meta = {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "downloaded_utc": stamp,
        "from_cache": False,
    }
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    return meta


# ----------------------------------------------------------------------- parse
def parse_siot(
    payload: str | bytes | Mapping[str, Any],
    *,
    unit: str = DEFAULT_UNIT,
    stk_flow: str = DEFAULT_STK_FLOW,
    provenance: Mapping[str, Any] | None = None,
) -> SIOT:
    """Turn a Eurostat ``naio_10_cp1700`` payload into a :class:`SIOT`.

    Args:
        payload: Raw JSON-stat response (bytes/str) or decoded dict.
        unit: Unit filter applied to the request; recorded in the result.
        stk_flow: Stock/flow filter applied to the request.
        provenance: Extra provenance keys merged into ``SIOT.provenance``.

    Returns:
        Parsed symmetric input-output table on the domestic/stated basis.

    Raises:
        EurostatError: if the payload does not have the expected structure.
    """
    try:
        js = JsonStat.parse(payload)
    except JsonStatError as exc:
        raise EurostatError(f"not a usable JSON-stat payload: {exc}") from exc

    for dim in ("prd_use", "prd_ava"):
        if dim not in js.ids:
            raise EurostatError(
                f"expected structural dimension {dim!r}; got {list(js.ids)} from {DATASET}"
            )
    # Pin the filters that are meant to be singletons.
    fixed: dict[str, str] = {}
    for dim, wanted in (("unit", unit), ("stk_flow", stk_flow)):
        if dim in js.ids:
            available = js.codes[dim]
            if wanted not in available:
                raise EurostatError(
                    f"{dim}={wanted!r} not in response categories {available}; "
                    "check the query filters"
                )
            fixed[dim] = wanted

    try:
        dense, _, axis = _slice(js, fixed)
    except JsonStatError as exc:  # pragma: no cover - defensive
        raise EurostatError(str(exc)) from exc

    use_codes = axis["prd_use"]
    ava_codes = axis["prd_ava"]
    use_index = {c: i for i, c in enumerate(use_codes)}
    ava_index = {c: i for i, c in enumerate(ava_codes)}

    # Eurostat lists the *full* CPA tree on both axes (104 codes) but publishes
    # values only for the A*64-level branches plus CPA_T / CPA_U (65 codes).  A
    # code counts as published when the payload defines at least one cell for
    # it; the no-data codes are strictly finer subdivisions of a published code,
    # so dropping them loses nothing — which is asserted, not assumed.
    candidates = [c for c in ava_codes if c in use_index and c not in _NON_PRODUCT_CODES]
    if not candidates:
        raise EurostatError("no product/industry codes shared by prd_use and prd_ava")
    cand_rows = [use_index[c] for c in candidates]
    cand_cols = [ava_index[c] for c in candidates]
    raw_block = dense[np.ix_(cand_rows, cand_cols)]  # [using branch, product]
    finite = np.isfinite(raw_block)
    p1_all = (
        np.array([dense[r, ava_index["P1"]] if "P1" in ava_index else np.nan for r in cand_rows])
    )
    published = finite.any(axis=0) | finite.any(axis=1) | np.isfinite(p1_all)
    products = [code for code, keep in zip(candidates, published) if keep]
    dropped = [code for code, keep in zip(candidates, published) if not keep]
    if not products:
        raise EurostatError("the payload defines no product cells")

    rows = [use_index[c] for c in products]  # using branches, canonical order
    cols = [ava_index[c] for c in products]  # supplied products, same order
    # X[r, c] = amount of product c used by r  ->  Z[product, industry] = X.T
    block = dense[np.ix_(rows, cols)]
    kept_total = float(np.nansum(block))
    all_total = float(np.nansum(raw_block))
    if abs(kept_total - all_total) > 1e-6 * max(abs(all_total), 1.0):
        raise EurostatError(
            "dropping unpublished product codes would discard defined values: "
            f"kept {kept_total:.6f} vs all {all_total:.6f}"
        )
    Z = np.asarray(block.T, dtype=float)
    Z = np.nan_to_num(Z, nan=0.0)

    def _attribute(item: str, *, default: float = 0.0) -> np.ndarray:
        """Read a column-total attribute stored along the ``prd_use`` axis."""
        if item not in ava_index:
            return np.full(len(products), default, dtype=float)
        column = ava_index[item]
        values = np.array([dense[row, column] for row in rows], dtype=float)
        return np.nan_to_num(values, nan=default)

    def _final_use(item: str) -> np.ndarray:
        """Read a final-use row (``prd_use``) indexed by product."""
        if item not in use_index:
            return np.zeros(len(products), dtype=float)
        row = use_index[item]
        values = np.array([dense[row, column] for column in cols], dtype=float)
        return np.nan_to_num(values, nan=0.0)

    x = _attribute("P1")
    y_total = _final_use("P3") + _final_use("P5") + _final_use("P6")
    household = _final_use("P3_S14")
    # P3 = P3_S13 (government) + P3_S14 (households) + P3_S15 (NPISH).  NPISH
    # collective consumption is grouped with government so that the components
    # sum to P3 exactly (asserted below).
    government = _final_use("P3_S13") + _final_use("P3_S15")
    gfcf = _final_use("P51G")
    # P5 = P51G + everything else (inventories, valuables); taking the residual
    # keeps the split exact instead of double counting the P5M aggregate.
    inventories = _final_use("P5") - gfcf
    exports = _final_use("P6")
    if not np.any(y_total):
        y_total = household + government + gfcf + inventories + exports

    names: dict[str, str] = {}
    for dim in ("prd_ava", "prd_use"):
        for code, label in js.labels.get(dim, {}).items():
            names.setdefault(code, label)

    prov: dict[str, Any] = {
        "dataset": DATASET,
        "label": js.label,
        "updated": js.updated,
        "source": js.source_label,
        "id": list(js.ids),
        "size": list(js.sizes),
        "defined_cells": js.defined(),
        "total_cells": js.n_cells,
        "status_counts": js.status_counts(),
        "unit_filter": unit,
        "stk_flow_filter": stk_flow,
        "datastructure": _datastructure(js),
        "n_source_sectors": len(products),
        "n_candidate_codes": len(candidates),
        "n_unpublished_codes": len(dropped),
        "unpublished_codes": dropped,
        "block_total_value": kept_total,
    }
    if provenance:
        prov.update(provenance)

    siot = SIOT(
        codes=tuple(products),
        names=names,
        Z=Z,
        x=x,
        y_total=y_total,
        household=household,
        government=government,
        gfcf=gfcf,
        inventories=inventories,
        exports=exports,
        imports=_attribute("P7"),
        compensation=_attribute("D1"),
        gva=_attribute("B1G"),
        intermediate_consumption=_attribute("P2_ADJ"),
        unit=unit,
        price_basis="basic",
        stk_flow=stk_flow,
        provenance=prov,
    )
    _validate(siot)
    return siot


def load_siot(
    raw_dir: Path,
    *,
    year: int = 2020,
    geo: str = DEFAULT_GEO,
    unit: str = DEFAULT_UNIT,
    stk_flow: str = DEFAULT_STK_FLOW,
    force: bool = False,
) -> SIOT:
    """Fetch (or reuse) the raw payload and parse it into a :class:`SIOT`."""
    url = default_url(year=year, geo=geo, unit=unit, stk_flow=stk_flow)
    cache_path = raw_cache_path(raw_dir, year=year, geo=geo, unit=unit, stk_flow=stk_flow)
    meta = fetch_raw(url, cache_path, force=force)
    payload = Path(cache_path).read_bytes()
    merged = dict(meta)
    merged["geo"] = geo
    merged["year"] = year
    return parse_siot(payload, unit=unit, stk_flow=stk_flow, provenance=merged)


# -------------------------------------------------------------------- helpers
def _slice(
    js: JsonStat, fixed: Mapping[str, str]
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Drop the pinned and singleton dimensions; return the remaining cube.

    Axes of size 1 carry no information (``freq``, ``geo``, ``time`` here), so
    they are squeezed out.  The survivors keep their ``id`` order, which is what
    makes ``dense[prd_use_position, prd_ava_position]`` meaningful downstream.
    """
    dense = js.dense()
    # Pin every fixed dimension in ONE indexing pass: sequential np.take calls
    # would use stale axis numbers once an earlier axis has been dropped.
    indexer: list[object] = [slice(None)] * js.ndim
    for dim, code in fixed.items():
        indexer[js.ids.index(dim)] = js.position(dim, code)
    dense = dense[tuple(indexer)]
    retained = [d for d in js.ids if d not in fixed]
    axis: dict[str, tuple[str, ...]] = {}
    squeeze: list[int] = []
    for position, dim in enumerate(retained):
        if len(js.codes[dim]) == 1:
            squeeze.append(position)
        else:
            axis[dim] = js.codes[dim]
    if squeeze:
        dense = np.squeeze(dense, axis=tuple(squeeze))
    order = tuple(axis)
    return dense, order, axis


def _datastructure(js: JsonStat) -> dict[str, Any]:
    ext = js.extension or {}
    ds = ext.get("datastructure", {}) if isinstance(ext, Mapping) else {}
    return {
        "id": ds.get("id"),
        "agencyId": ds.get("agencyId"),
        "version": ds.get("version"),
        "dataset_version": ext.get("version") if isinstance(ext, Mapping) else None,
    }


def _validate(siot: SIOT) -> None:
    """Cheap structural assertions; raises :class:`EurostatError` on failure."""
    n = siot.n
    for name in ("Z", "x", "y_total", "household"):
        arr = getattr(siot, name)
        if arr.shape != ((n, n) if name == "Z" else (n,)):
            raise EurostatError(f"{name} has shape {arr.shape}, expected {(n, n) if name == 'Z' else (n,)}")
    if np.isnan(siot.Z).any() or np.isnan(siot.x).any():
        raise EurostatError("parsed table contains NaN")
    if (siot.Z < -1e-9).any():
        raise EurostatError("parsed table contains negative intermediate flows")
    components = (
        siot.household + siot.government + siot.gfcf + siot.inventories + siot.exports
    )
    scale = max(float(np.max(np.abs(siot.y_total))), 1.0)
    if np.max(np.abs(components - siot.y_total)) > 1e-6 * scale:
        raise EurostatError(
            "final-demand components do not sum to total final use: "
            f"max residual {np.max(np.abs(components - siot.y_total)):.6g}"
        )
    positive = siot.x > 0
    if positive.any():
        shares = siot.Z[:, positive].sum(axis=0) / siot.x[positive]
        if (shares >= 1.0).any():
            bad = siot.codes[int(np.argmax(shares))]
            raise EurostatError(f"intermediate share >= 1 for {bad}; values are not value-additive")


def fetch_published_gva(
    *,
    year: int = 2020,
    geo: str = DEFAULT_GEO,
    timeout: int = 60,
) -> dict[str, Any]:
    """Fetch Eurostat's published EU gross value added, for an independent check.

    This is a *validation* call, deliberately separate from the SIOT build: it
    lets the manifest record how far the SIOT-implied GVA total is from the
    national-accounts total, instead of asserting the table is right.

    Args:
        year: Reference year.
        geo: Geo code.
        timeout: Socket timeout in seconds.

    Returns:
        ``{"na_item": "B1G", "value": float, "url": str, "unit": "CP_MEUR",
        "retrieved_utc": str}`` or ``{"status": "unavailable", "reason": str}``.
    """
    query = {
        "format": "JSON",
        "lang": "EN",
        "unit": "CP_MEUR",
        "na_item": "B1G",
        "geo": geo,
        "time": str(year),
    }
    url = (
        f"{API_ROOT}/nama_10_gdp?"
        f"{urllib.parse.urlencode(query)}"
    )
    record: dict[str, Any] = {"na_item": "B1G", "unit": "CP_MEUR", "url": url, "geo": geo}
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "planlab/0.1 (+data layer validation only)"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read())
        values = payload.get("value") or {}
        if not values:
            record.update(status="unavailable", reason="empty 'value' in response")
            return record
        record["value"] = float(next(iter(values.values())))
        record["published_label"] = payload.get("label")
        record["published_updated"] = payload.get("updated")
        record["retrieved_utc"] = _utc_now()
        record["status"] = "ok"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, StopIteration) as exc:
        record.update(status="unavailable", reason=str(exc))
    return record


def to_coefficients(Z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``A[i, j] = Z[i, j] / x[j]`` with zero columns for zero output."""
    Z = np.asarray(Z, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(Z, dtype=float)
    positive = x > 0
    out[:, positive] = Z[:, positive] / x[positive]
    return out


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
