# PlanLab — Data Layer Specification (v1, frozen)

The data layer turns a published symmetric input–output table into a
`planlab.core.state.Technology` object. It must be **fully reproducible from a
single command** and must never require manual downloads.

## 1. Sources (in priority order)

| id | source | dataset | licence | notes |
|---|---|---|---|---|
| `eurostat` | Eurostat | `naio_10_cp1700` — symmetric input–output table, basic prices, 64×64 industries, EU27_2020, 2020 | CC-BY 4.0 / Eurostat open data | **primary**, reachable via JSON-stat API, no key |
| `synthetic` | PlanLab generator | documented calibration to eurostat aggregate totals | MIT (ours) | offline fallback + robustness substrate |
| `nbs2020` | China NBS | manual upload only | — | stub: documented loader for a user-supplied file, not in the CI path |

If the network is unavailable, `synthetic` must produce a table with the same
schema and the run must not crash. Everything that reaches the paper must record
which substrate was used.

## 2. Files written by the data layer

All under `data/processed/`, keyed by `<source>_<year>_n<n>`:

```
data/processed/eurostat_2020_n20_A.csv        # index col = sector code; values = a[i,j]
data/processed/eurostat_2020_n20_meta.json    # provenance + vectors
data/processed/eurostat_2020_n20_labels.csv   # sector_index, code, name_en, name_zh, aggregate
data/raw/eurostat_naio_10_cp1700_2020_EU27_2020.jsonstat   # exactly as downloaded
data/metadata/sector_map_eurostat64_to_20.csv # code64, name_en, sector20_code, mapping_note
data/metadata/manifest.json                   # sha256, url, timestamp, sizes, versions
```

`*_meta.json` keys (all required):

```json
{
  "source": "eurostat", "year": 2020, "n_sectors": 20, "unit": "million EUR, basic prices, current prices",
  "price_basis": "basic", "table_type": "symmetric",
  "sectors": ["c01", "..."],
  "gross_output": [/* length n */],
  "base_final": [/* length n, REQUIRED: total final demand = x - A x */],
  "final_consumption": [/* length n, household column from the table, informational */],
  "labor": [/* length n, labour compensation / gross output */],
  "kappa": [/* length n, capital-output ratio */],
  "consumption_weights": [/* length n */],
  "essentials": [/* length n, 0/1 */],
  "provenance": {
     "url": "...", "downloaded_utc": "...", "raw_sha256": "...",
     "aggregation": "64->20 per data/metadata/sector_map_eurostat64_to_20.csv",
     "transform_version": "1"
  }
}
```

`Technology.meta` = the whole `*_meta.json` dict, so a result manifest can always
name the substrate.

## 3. Construction rules (must match, or `verify_technology` fails)

1. **Direct input coefficients.** From the symmetric table's intermediate block
   `Z` (product i used by industry j) and gross output `x_j`:
   `a_ij = Z_ij / x_j`. Sectors with `x_j == 0` get a zero column and a warning;
   the table must satisfy `sum(A, axis=0) < 1` for every sector (no value added
   negative). Assert it.
2. **Aggregation 64 → 20.** Fixed, versioned mapping in
   `data/metadata/sector_map_eurostat64_to_20.csv`. The aggregated table must be
   obtained by summing *values* (not coefficients) and then dividing by the
   aggregated gross output — i.e. aggregate in value space, then compute `A`.
3. **Labour.** `labor_j = (compensation of employees in industry j) / x_j`
   (hour-free, value basis). If unavailable, fall back to `labor_j = 0.5·(1 − sum_i a_ij)`
   and flag `"labor_fallback": true` in meta.
4. **kappa.** `kappa_j = gross fixed capital formation attributable to j / x_j`,
   floored at 0.5 and capped at 8. If unavailable, `kappa = 2.5` for all sectors,
   flag `"kappa_fallback": true`.
5. **consumption_weights.** Default `1.0`; sectors whose `aggregate` group is
   "essential/food/health/energy" get `1.2`. Values are recorded in meta.
6. **essentials.** `1.0` for food/agriculture, health, energy/utilities, transport
   of people if identifiable from the mapping; otherwise `0.0`.
7. **base_output / base_final.** `x_j` is gross output. `base_final` is **total
   final demand**, defined as the residual `x - A x` and required to satisfy the
   identity `A x + y = x` exactly (the core refuses a technology that violates it).
   PlanLab is a closed-economy model, so the household / government / investment
   split is imposed *inside* total final demand by the scenario
   (`gov_share`, `depreciation`), not by separate vectors. The table's household
   column (`P3_S14`) is still recorded, under the key `final_consumption`, for
   reference and for computing household shares; it is **not** what the engine
   consumes. Earlier revisions of this document said "household final consumption"
   for `base_final`; that wording was wrong and contradicted the core contract.
8. **Sanity:** `0 ≤ a_ij ≤ 1`, no NaN, `sum_i a_ij ≤ 1`, `n == 20`,
   and the identity `A @ x + y ≈ x` reproduced within `1e-6` relative before
   aggregation.

## 4. Public API (owned by the data module)

```python
from planlab.data import load_technology, available_substrates

available_substrates() -> list[str]           # ['eurostat_2020_n20', 'synthetic_2020_n20', ...]
load_technology(source='eurostat', year=2020, n_sectors=20, *, offline_ok=True) -> Technology
build_all(force: bool = False) -> dict        # fetch + process + write manifest
```

`load_technology` must be **cheap and offline**: it reads `data/processed/` and
never hits the network. Fetching is a separate explicit step
(`python -m planlab.data.build --source eurostat`) wired into `scripts/fetch_data.py`.

## 5. Reproducibility requirements

* `scripts/fetch_data.py` is idempotent; a second run re-uses the cached raw file
  and verifies its sha256.
* `data/metadata/manifest.json` records url, sha256, byte size, retrieval time,
  and the versions of the attributes requested from the API (so a silent Eurostat
  revision is detectable).
* The processed artefacts are committed (they are small: 20×20 + 4 vectors), so the
  paper's numbers are reproducible without network access. `data/raw/` is gitignored
  when > 5 MB.
* No pandas index/column ambiguity: CSV index is the sector code, columns are
  sector codes, and `labels.csv` defines the order used inside the matrix.
