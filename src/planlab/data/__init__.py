"""PlanLab data layer — published input-output tables to ``Technology``.

The only module of PlanLab that knows about the outside world.  Everything it
produces is a *recorded* transformation of a published table: no number enters
the simulation that was not either taken from the source or explicitly flagged
as a documented assumption.

Public API (``docs/DATA_SCHEMA.md`` §4)::

    from planlab.data import load_technology, available_substrates, build_all

    tech = load_technology("eurostat", 2020, 20)   # offline, reads data/processed/

Substrates
----------
``eurostat``   Eurostat ``naio_10_cp1700`` symmetric input-output table at basic
               prices (product by product), EU27, 2020, million EUR, domestic
               uses, aggregated 65 -> 20 sectors.
``synthetic``  A documented generator calibrated to the Eurostat aggregate;
               offline fallback and robustness substrate.

See ``docs/DATA_NOTES.md`` for what the table contains, how the 65 -> 20 mapping
was chosen, and which quantities are approximations.
"""

from __future__ import annotations

from .aggregate import (
    DEFAULT_WEIGHT,
    ESSENTIAL_AGGREGATES,
    ESSENTIAL_WEIGHT,
    GROUP_DEFINITIONS,
    AggregatedSIOT,
    SectorMap,
    aggregate_siot,
    consumption_weights,
    essentials,
    group_vector,
    grouping_matrix,
    load_sector_map,
    sector_labels,
)
from .eurostat import (
    DATASET,
    SIOT,
    EurostatError,
    default_url,
    load_siot,
    parse_siot,
    to_coefficients,
)
from .jsonstat import JsonStat, JsonStatError
from .loader import (
    KAPPA_DEFAULT,
    TRANSFORM_VERSION,
    DataError,
    artifact_key,
    available_substrates,
    build_all,
    build_eurostat,
    build_synthetic,
    load_technology,
    make_technology,
    read_substrate_meta,
    write_manifest,
)
from .synthetic import DEFAULT_SEED, DEFAULT_SIGMA, SyntheticResult, generate_synthetic

__version__ = "0.1.0"

__all__ = [
    # public contract
    "load_technology",
    "available_substrates",
    "build_all",
    # building blocks
    "SIOT",
    "AggregatedSIOT",
    "SectorMap",
    "SyntheticResult",
    "JsonStat",
    "GROUP_DEFINITIONS",
    "ESSENTIAL_AGGREGATES",
    "ESSENTIAL_WEIGHT",
    "DEFAULT_WEIGHT",
    "KAPPA_DEFAULT",
    "TRANSFORM_VERSION",
    "DATASET",
    "DEFAULT_SEED",
    "DEFAULT_SIGMA",
    "aggregate_siot",
    "consumption_weights",
    "essentials",
    "group_vector",
    "grouping_matrix",
    "load_sector_map",
    "sector_labels",
    "generate_synthetic",
    "default_url",
    "load_siot",
    "parse_siot",
    "to_coefficients",
    "build_eurostat",
    "build_synthetic",
    "write_manifest",
    "read_substrate_meta",
    "artifact_key",
    "make_technology",
    # errors
    "DataError",
    "EurostatError",
    "JsonStatError",
]
