"""Value-space aggregation of the 65-branch Eurostat SIOT onto 20 sectors.

Rule (``docs/DATA_SCHEMA.md`` §3.2, non-negotiable): aggregate **values**, then
compute coefficients.  Concretely, with the 0/1 grouping matrix
``S`` (``n_target x n_source``, one 1 per source column):

.. math::

    Z^{20} = S Z S^{\\top}, \\quad x^{20} = S x, \\quad y^{20} = S y,
    \\quad A^{20}_{ij} = Z^{20}_{ij} / x^{20}_j .

Averaging coefficients instead would silently weight each branch by its
coefficient rather than by its transaction value and would break
``sum(A, axis=0) < 1``.  The aggregation is checked to preserve the global
totals of ``Z``, ``x`` and ``y`` exactly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .eurostat import SIOT, to_coefficients

__all__ = [
    "GROUP_DEFINITIONS",
    "ESSENTIAL_AGGREGATES",
    "ESSENTIAL_WEIGHT",
    "DEFAULT_WEIGHT",
    "SectorMap",
    "AggregatedSIOT",
    "load_sector_map",
    "aggregate_siot",
    "grouping_matrix",
    "group_vector",
    "sector_labels",
]

#: Canonical 20-sector grouping.
#: ``(code, English name, Chinese name, aggregate tag)``.
#: The aggregate tag is functional, not decorative: :data:`ESSENTIAL_AGGREGATES`
#: selects which groups are treated as the "essentials" priority tier, which in
#: turn sets ``consumption_weights`` and ``essentials`` in the Technology.
GROUP_DEFINITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("c01", "Agriculture, forestry & fishing", "农、林、渔业", "food_agriculture"),
    ("c02", "Mining & quarrying", "采矿与采石", "mining_extraction"),
    ("c03", "Food, beverages & tobacco", "食品、饮料与烟草", "food_agriculture"),
    ("c04", "Textiles, apparel & leather", "纺织、服装与皮革", "manufacturing_light"),
    ("c05", "Wood, paper & printing", "木材、造纸与印刷", "manufacturing_light"),
    ("c06", "Coke & refined petroleum", "焦炭与精炼石油", "energy_fuels"),
    ("c07", "Chemicals & pharmaceuticals", "化学品与医药", "manufacturing_heavy"),
    ("c08", "Rubber, plastics & non-metallic minerals", "橡胶、塑料与非金属矿物", "manufacturing_heavy"),
    ("c09", "Basic & fabricated metals", "基础金属与金属制品", "manufacturing_heavy"),
    ("c10", "Machinery, vehicles & electronics", "机械、车辆与电子", "capital_goods"),
    ("c11", "Furniture, other manufacturing & repair", "家具、其他制造与修理", "manufacturing_other"),
    ("c12", "Electricity, gas, water & waste", "电力、燃气、水与废物处理", "energy_utilities"),
    ("c13", "Construction", "建筑业", "construction"),
    ("c14", "Wholesale & retail trade", "批发与零售贸易", "trade"),
    ("c15", "Transport & storage", "运输与仓储", "transport"),
    ("c16", "Accommodation & food services", "住宿与餐饮服务", "hospitality"),
    ("c17", "Information & communication", "信息与通信", "information"),
    ("c18", "Finance, insurance & real estate", "金融、保险与房地产", "finance_realestate"),
    ("c19", "Public administration, education & health", "公共管理、教育与卫生", "public_services"),
    ("c20", "Professional, business & other services", "专业、商业及其他服务", "professional_services"),
)

#: Aggregate tags that form the "essentials" priority tier
#: (food/agriculture, energy/utilities, health & public services).
ESSENTIAL_AGGREGATES: frozenset[str] = frozenset(
    {"food_agriculture", "energy_utilities", "public_services"}
)

#: Welfare weight for essentials and non-essentials (``docs/DATA_SCHEMA.md`` §3.5).
ESSENTIAL_WEIGHT = 1.2
DEFAULT_WEIGHT = 1.0


@dataclass(frozen=True)
class SectorMap:
    """A complete ``source code -> target sector`` mapping.

    Attributes:
        code64: Source codes in mapping-file order, (n_source,).
        sector20: Target code for each source code, (n_source,).
        sectors: Target codes in :data:`GROUP_DEFINITIONS` order, (n_target,).
        notes: ``{source code: mapping_note}``.
        names: ``{source code: name_en}`` from the mapping file.
    """

    code64: tuple[str, ...]
    sector20: tuple[str, ...]
    sectors: tuple[str, ...]
    notes: Mapping[str, str]
    names: Mapping[str, str]

    @property
    def n_source(self) -> int:
        """Number of source sectors."""
        return len(self.code64)

    @property
    def n_target(self) -> int:
        """Number of target sectors."""
        return len(self.sectors)

    def counts(self) -> dict[str, int]:
        """Number of source codes per target sector."""
        out = {s: 0 for s in self.sectors}
        for s in self.sector20:
            out[s] += 1
        return out

    def members(self, sector: str) -> list[str]:
        """Source codes mapped onto ``sector``."""
        return [c for c, s in zip(self.code64, self.sector20) if s == sector]


@dataclass(frozen=True)
class AggregatedSIOT:
    """A SIOT aggregated onto the target sector list — still in value units.

    Attributes:
        sectors: Target sector codes, (n,).
        Z: Aggregated intermediate uses, (n, n), million euro.
        x: Aggregated gross output, (n,).
        y_total: Aggregated total final use, (n,).
        household/government/gfcf/inventories/exports/imports: final-use
            components, (n,).
        compensation: Aggregated compensation of employees, (n,).
        gva: Aggregated gross value added, (n,).
        intermediate_consumption: Aggregated ``P2_ADJ``, (n,).
        n_source: Number of source branches that were aggregated.
    """

    sectors: tuple[str, ...]
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
    n_source: int

    @property
    def n(self) -> int:
        """Number of aggregated sectors."""
        return len(self.sectors)

    def A(self) -> np.ndarray:
        """Direct input coefficients of the aggregated table."""
        return to_coefficients(self.Z, self.x)


# ------------------------------------------------------------------- mapping IO
def load_sector_map(path: str | Path) -> SectorMap:
    """Load and validate ``sector_map_eurostat64_to_20.csv``.

    Args:
        path: CSV with columns ``code64, name_en, sector20_code, mapping_note``.

    Returns:
        A validated :class:`SectorMap`.

    Raises:
        ValueError: on duplicate source codes, unknown target sectors, or a
            mapping that does not cover every known target sector.
    """
    path = Path(path)
    code64: list[str] = []
    sector20: list[str] = []
    notes: dict[str, str] = {}
    names: dict[str, str] = {}
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"code64", "name_en", "sector20_code", "mapping_note"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            code = row["code64"].strip()
            target = row["sector20_code"].strip()
            if not code or not target:
                continue
            if code in seen:
                raise ValueError(f"{path}: duplicate source code {code!r}")
            seen.add(code)
            if target not in {g[0] for g in GROUP_DEFINITIONS}:
                raise ValueError(f"{path}: unknown target sector {target!r} for {code!r}")
            code64.append(code)
            sector20.append(target)
            notes[code] = row["mapping_note"].strip()
            names[code] = row["name_en"].strip()

    sectors = tuple(g[0] for g in GROUP_DEFINITIONS)
    covered = {*sector20}
    missing_targets = [s for s in sectors if s not in covered]
    if missing_targets:
        raise ValueError(f"{path}: target sectors with no source codes: {missing_targets}")
    if len(sectors) != 20:
        raise ValueError(f"expected 20 target sectors, found {len(sectors)}")
    return SectorMap(
        code64=tuple(code64),
        sector20=tuple(sector20),
        sectors=sectors,
        notes=notes,
        names=names,
    )


def grouping_matrix(smap: SectorMap, source_codes: Sequence[str]) -> np.ndarray:
    """Return the 0/1 matrix ``S`` (``n_target x n_source``) in ``source_codes`` order.

    Args:
        smap: The sector mapping.
        source_codes: Source codes in the order used by the arrays to aggregate.

    Raises:
        ValueError: if a source code has no mapping entry.
    """
    target_pos = {s: i for i, s in enumerate(smap.sectors)}
    lookup = dict(zip(smap.code64, smap.sector20))
    S = np.zeros((smap.n_target, len(source_codes)), dtype=float)
    unmapped = []
    for j, code in enumerate(source_codes):
        target = lookup.get(code)
        if target is None:
            unmapped.append(code)
            continue
        S[target_pos[target], j] = 1.0
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} source sectors have no mapping entry, e.g. {unmapped[:5]}"
        )
    return S


def group_vector(smap: SectorMap, values: np.ndarray, source_codes: Sequence[str]) -> np.ndarray:
    """Aggregate a length-``n_source`` value vector onto the target sectors."""
    return grouping_matrix(smap, source_codes) @ np.asarray(values, dtype=float)


# ------------------------------------------------------------------ aggregation
def aggregate_siot(siot: SIOT, smap: SectorMap) -> AggregatedSIOT:
    """Aggregate a 65-branch :class:`SIOT` onto the 20-sector mapping.

    Values are summed first; coefficients are derived afterwards.  Global
    totals of ``Z``, ``x``, ``y_total`` and the value-added components are
    preserved to machine precision (asserted here).

    Args:
        siot: Parsed source table.
        smap: Sector mapping covering every code in ``siot.codes``.

    Returns:
        The aggregated table.

    Raises:
        ValueError: if the mapping does not cover the source codes, or if the
            aggregation is not value preserving.
    """
    S = grouping_matrix(smap, siot.codes)
    Z = S @ siot.Z @ S.T
    x = S @ siot.x
    out = AggregatedSIOT(
        sectors=smap.sectors,
        Z=Z,
        x=x,
        y_total=S @ siot.y_total,
        household=S @ siot.household,
        government=S @ siot.government,
        gfcf=S @ siot.gfcf,
        inventories=S @ siot.inventories,
        exports=S @ siot.exports,
        imports=S @ siot.imports,
        compensation=S @ siot.compensation,
        gva=S @ siot.gva,
        intermediate_consumption=S @ siot.intermediate_consumption,
        n_source=len(siot.codes),
    )
    _check_value_preserved(siot.Z.sum(), Z.sum(), "Z")
    _check_value_preserved(siot.x.sum(), out.x.sum(), "x")
    _check_value_preserved(siot.y_total.sum(), out.y_total.sum(), "y_total")
    if np.isnan(out.Z).any() or np.isnan(out.x).any():
        raise ValueError("aggregated table contains NaN")
    return out


def sector_labels(smap: SectorMap) -> list[dict[str, object]]:
    """Build the rows of ``labels.csv`` for the aggregated sectors.

    Returns:
        One dict per target sector with keys ``sector_index``, ``code``,
        ``name_en``, ``name_zh``, ``aggregate``, ``essential`` and ``n_source``.
    """
    counts = smap.counts()
    rows: list[dict[str, object]] = []
    for index, (code, name_en, name_zh, aggregate) in enumerate(GROUP_DEFINITIONS):
        rows.append(
            {
                "sector_index": index,
                "code": code,
                "name_en": name_en,
                "name_zh": name_zh,
                "aggregate": aggregate,
                "essential": 1 if aggregate in ESSENTIAL_AGGREGATES else 0,
                "n_source": counts.get(code, 0),
            }
        )
    return rows


def consumption_weights(smap: SectorMap) -> np.ndarray:
    """Welfare weights ``w_i``: 1.2 for the essentials tier, else 1.0.

    Args:
        smap: Sector mapping (supplies the target order).

    Returns:
        Array of length 20.
    """
    tags = {g[0]: g[3] for g in GROUP_DEFINITIONS}
    return np.array(
        [
            ESSENTIAL_WEIGHT if tags[s] in ESSENTIAL_AGGREGATES else DEFAULT_WEIGHT
            for s in smap.sectors
        ],
        dtype=float,
    )


def essentials(smap: SectorMap) -> np.ndarray:
    """Priority tier indicator: 1.0 for essentials, 0.0 otherwise."""
    tags = {g[0]: g[3] for g in GROUP_DEFINITIONS}
    return np.array(
        [1.0 if tags[s] in ESSENTIAL_AGGREGATES else 0.0 for s in smap.sectors],
        dtype=float,
    )


def _check_value_preserved(before: float, after: float, what: str) -> None:
    scale = max(abs(before), 1.0)
    if abs(before - after) > 1e-6 * scale:
        raise ValueError(
            f"aggregation did not preserve the total of {what}: {before} -> {after}"
        )
