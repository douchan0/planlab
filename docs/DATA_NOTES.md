# PlanLab — Data Notes

**Scope.** This document describes the data layer: what the source table contains,
how it was read, how 65 published branches became the 20 sectors the simulation
consumes, which quantities are *measurements* and which are *documented
assumptions*, and the exact licence basis. It is written to be checkable: every
number below is either a value recorded in `data/metadata/manifest.json` or an
excerpt from the source payload.

Frozen specs this implements: `docs/DATA_SCHEMA.md` (v1) and
`docs/ARCHITECTURE.md` §2–§3. Deviations are listed in §8.

---

## 1. Source

| field | value |
|---|---|
| dataset | Eurostat `naio_10_cp1700` — *Symmetric input-output table at basic prices (product by product)* |
| geo | `EU27_2020` (EU-27 aggregate from 2020) |
| time | `2020` |
| unit | `MIO_EUR` — million euro, current prices |
| stock/flow | `DOM` — domestic uses |
| URL | `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/naio_10_cp1700?format=JSON&lang=EN&freq=A&geo=EU27_2020&time=2020&unit=MIO_EUR&stk_flow=DOM` |
| bytes / sha256 | 102 822 B / `f3c69cf70374c7c37a4fff9a500ccd31fb7f86a1a293932c0ec9a59b231c6c72` |
| dataset `updated` | `2026-09-30T23:00:00+0200` |
| datastructure | `NAIO_10_CP1700` version `61.0` |
| retrieved (UTC) | `2026-11-01T07:56:24+00:00` |

The raw payload is cached verbatim at
`data/raw/eurostat_naio_10_cp1700_2020_EU27_2020_MIO_EUR_DOM.jsonstat` (git-ignored)
together with a `.meta.json` sidecar. A rebuild re-uses the cache and re-verifies
the sha256; `--force` re-downloads. The 65-branch *derived* values are committed
as `data/processed/eurostat_2020_n65_Z.csv` + `_vectors.csv`, so every claim in
this document can be re-checked without network access.

### 1.1 What the response actually looks like (read, not assumed)

`id = [freq, unit, stk_flow, prd_use, prd_ava, geo, time]`,
`size = [1, 1, 1, 121, 123, 1, 1]`.

* Only **5 712 of 14 883** cells are defined: JSON-stat encodes the payload
  *sparsely*, and an absent cell is "not published", **not** zero. Treating it as
  zero would silently corrupt the table, so `planlab.data.jsonstat` returns `NaN`
  and the SIOT parser converts to zero only after the structure is resolved.
* No `status` flags are present in this release (no confidentiality markers).
* The response carries the **full CPA product tree** (104 product codes on each
  structural axis) but publishes values only at the **A\*64 aggregation level plus
  `CPA_T` and `CPA_U`** — 65 branches. The 39 finer codes (e.g. `CPA_B05`,
  `CPA_C10`) are listed but empty; the parser detects this from the payload,
  drops them, and asserts that doing so discards no defined value
  (`block_total_value` in the provenance).

**Note the count: 65, not 64.** The A\*64 nomenclature has 64 entries; this
release also populates `CPA_U` (*services provided by extraterritorial
organisations and bodies*), which has zero output in the EU-27 aggregate. The
mapping therefore covers 65 source codes. This is a deviation from the "64→20"
wording in `docs/DATA_SCHEMA.md` §1/§3.2; see §8.

### 1.2 Orientation — the one thing that is easy to get backwards

The two structural dimensions are **not** oriented the way their labels suggest:

| dimension | categories | reads as |
|---|---|---|
| `prd_use` | 65 production branches + final-use aggregates (`P3`, `P5`, `P6`, `TU`, …) | the **using** side |
| `prd_ava` | the same 65 products + value-added items (`B1G`, `D1`, …) + `IMP`, `P2_ADJ`, `TS_BP` | the **supplied** side |

So a cell `X[prd_use=r, prd_ava=c]` means "amount of `c` used by `r`", and the
Leontief use matrix is the **transpose** of that slice:

```
Z[i, j] = X[prd_use = j, prd_ava = i]        # product i used by industry j
x[j]    = X[prd_use = j, prd_ava = P1]       # gross output of branch j
```

Value-added items (`P1`, `D1`, `B1G`, `P2_ADJ`) are stored along the `prd_use`
axis positionally aligned with the product codes; final-use items (`P3_S14`,
`P5`, `P6`, …) are stored along the `prd_ava` axis, indexed by product. This is
the layout Eurostat actually publishes, and it is confirmed by three independent
checks — the orientation that feels natural (products as rows) fails all three:

1. **Supply–use identity.** `Σ_users X[r,c] + final_use(c) = P1(c)` holds for
   every one of the 65 products, with a maximum absolute residual of
   **0.09 MIO EUR** on a 25.14 trillion EUR base (relative 5.0e-8) — pure
   publication rounding. The alternative orientation misses by up to
   4.8e5 MIO EUR per sector.
2. **Value added.** The implied gross value added totals **12 147 083 MIO EUR**,
   which is **0.18 % below** Eurostat's independently published EU-27 2020 GVA
   (`nama_10_gdp`, `na_item=B1G`, `unit=CP_MEUR`: 12 168 846). The orientation
   error would change this total by hundreds of billions.
3. **Economic plausibility.** Implied labour shares per branch are the textbook
   ones: coke & refined petroleum 4.8 %, imputed rents 0.1 %, employment services
   70.8 %, education 66.9 %. Under the wrong orientation they are not.

The parser records the residual of check 1 in the manifest and **raises** if it
ever exceeds 1e-6 relative.

### 1.3 Final demand decomposition

`y_total = P3 + P5 + P6` (total final use, the residual `x − A x`). It is split
into components so that they sum to `y_total` exactly (asserted):

* household = `P3_S14`
* government = `P3_S13 + P3_S15` — government *and* NPISH collective consumption
* GFCF = `P51G`
* inventories = `P5 − P51G` (residual: changes in inventories + valuables; the
  `P5M` aggregate is *not* added separately, which would double count)
* exports = `P6`

The `DOM` (domestic-uses) block is used rather than `TOTAL`, because a closed
planned economy should be modelled on domestically produced supply. With `DOM`
the identity `A x + y = x` is exact to rounding; with `TOTAL` imported
intermediates enter the same columns and the identity no longer closes. `TOTAL`
remains reachable via `--stk-flow TOTAL` for robustness work.

---

## 2. The 20 sectors

Mapping: `data/metadata/sector_map_eurostat64_to_20.csv` (sha256
`661fe199…f078f`), 65 rows, exactly 20 groups, every group non-empty.

Design principles, in order of precedence:

1. **Keep the branches the research question needs distinguishable.** Energy,
   food, metals, chemicals, machinery and construction each stay separate, since
   the shock-propagation block (B3) shocks one sector at a time and the
   essentials tier needs an identifiable energy/food group.
2. **Do not split a published aggregate.** Eurostat publishes `C10-12`,
   `C13-15`, `C31_32`, `E37-39`, `J59_60`, `J62_63`, `M69_70`, `M74_75`,
   `N80-82`, `Q87_88`, `R90-92` as single branches; splitting them is impossible
   from this table, and inventing a split would be fabrication.
3. **Aggregate along supply-chain logic.** Primary → processed → capital goods;
   services grouped by function. Groups never mix a good with an unrelated
   service.
4. **Chinese names for teaching, ASCII identifiers for code.** `labels.csv`
   carries `name_zh`; sector codes are `c01…c20`.

| # | code | English | 中文 | branches merged | share of gross output |
|---|---|---|---|---|---|
| 1 | c01 | Agriculture, forestry & fishing | 农、林、渔业 | A01, A02, A03 | 1.82 % |
| 2 | c02 | Mining & quarrying | 采矿与采石 | B | 0.27 % |
| 3 | c03 | Food, beverages & tobacco | 食品、饮料与烟草 | C10-12 | 4.14 % |
| 4 | c04 | Textiles, apparel & leather | 纺织、服装与皮革 | C13-15 | 0.67 % |
| 5 | c05 | Wood, paper & printing | 木材、造纸与印刷 | C16, C17, C18 | 1.28 % |
| 6 | c06 | Coke & refined petroleum | 焦炭与精炼石油 | C19 | 0.81 % |
| 7 | c07 | Chemicals & pharmaceuticals | 化学品与医药 | C20, C21 | 2.77 % |
| 8 | c08 | Rubber, plastics & non-metallic minerals | 橡胶、塑料与非金属矿物 | C22, C23 | 1.76 % |
| 9 | c09 | Basic & fabricated metals | 基础金属与金属制品 | C24, C25 | 2.93 % |
| 10 | c10 | Machinery, vehicles & electronics | 机械、车辆与电子 | C26–C30 | 7.87 % |
| 11 | c11 | Furniture, other manufacturing & repair | 家具、其他制造与修理 | C31_32, C33 | 1.85 % |
| 12 | c12 | Electricity, gas, water & waste | 电力、燃气、水与废物处理 | D, E36, E37-39 | 3.50 % |
| 13 | c13 | Construction | 建筑业 | F | 7.16 % |
| 14 | c14 | Wholesale & retail trade | 批发与零售贸易 | G45, G46, G47 | 10.44 % |
| 15 | c15 | Transport & storage | 运输与仓储 | H49–H53 | 5.71 % |
| 16 | c16 | Accommodation & food services | 住宿与餐饮服务 | I | 1.85 % |
| 17 | c17 | Information & communication | 信息与通信 | J58, J59_60, J61, J62_63 | 5.49 % |
| 18 | c18 | Finance, insurance & real estate | 金融、保险与房地产 | K64, K65, K66, L68A, L68B | 12.38 % |
| 19 | c19 | Public administration, education & health | 公共管理、教育与卫生 | O, P, Q86, Q87_88 | 13.18 % |
| 20 | c20 | Professional, business & other services | 专业、商业及其他服务 | M69_70…M74_75, N77–N80-82, R90-92, R93, S94, S95, S96, T, U | 14.10 % |

Largest sectors by gross output: **c20** 14.10 %, **c19** 13.18 %, **c18**
12.38 %, **c14** 10.44 %, **c10** 7.87 %, **c13** 7.16 %.
Largest by household consumption: **c18** 28.63 % (dominated by imputed rents),
**c14** 17.87 %, **c03** 9.16 %.

### 2.1 Aggregation is done in value space

`Z²⁰ = S Z Sᵀ`, `x²⁰ = S x`, `y²⁰ = S y`, and only then `A²⁰ = Z²⁰ / x²⁰` (column-wise).
`S` is the 0/1 grouping matrix. Totals are preserved exactly and asserted:

| quantity | 65 branches | 20 sectors |
|---|---|---|
| intermediate uses | 11 159 053.33 | 11 159 053.33 |
| gross output | 25 143 704.52 | 25 143 704.52 |
| total final demand | 13 984 653.18 | 13 984 653.18 |
| household consumption | 5 724 438.97 | 5 724 438.97 |
| compensation of employees | 6 613 781.33 | 6 613 781.33 |
| gross value added | 12 147 083.21 | 12 147 083.21 |

A test (`test_aggregating_coefficients_would_give_a_different_answer`) fixes the
correct behaviour: averaging coefficients inside a group produces a different,
wrong matrix, and future refactors cannot silently switch to it.

---

## 3. What is measured, what is derived, what is assumed

| Technology field | status | construction |
|---|---|---|
| `A` | **measured** | `Z_ij / x_j` from the published table; 0 ≤ a ≤ 1, all column sums < 1 |
| `base_output` | **measured** | gross output `P1` per sector |
| `base_final` | **measured** | total final demand `x − A x` (= P3+P5+P6 split into components); see §8.1 |
| `labor` | **measured** | `D1_j / x_j` = compensation of employees over gross output (value basis, hours-free) |
| `kappa` | **assumed** | flat 2.5 — see §3.2 |
| `consumption_weights` | **normative** | 1.2 for the essentials tier, 1.0 otherwise (`docs/DATA_SCHEMA.md` §3.5) |
| `essentials` | **normative** | 1.0 for c01, c03, c12, c19; 0.0 otherwise |

### 3.1 Labour

`labor_j = D1_j / x_j` is real data. Range: **0.048** (coke & refined petroleum)
to **0.542** (public administration, education & health). The fallback
`0.5·(1 − Σ_i a_ij)` in `docs/DATA_SCHEMA.md` §3.3 is **not** used;
`labor_fallback: false` in meta.

Caveat worth stating in the paper: `D1` is compensation of *employees only*, so
for sectors with substantial self-employment (agriculture, trade, professional
services, arts) it understates the true labour input. Mixed income (`B3G`) is
empty in this release and cannot be added.

### 3.2 Capital intensity κ — a documented assumption, not data

`naio_10_cp1700` contains **no by-industry capital stock, no by-industry GFCF,
and no consumption of fixed capital**. `P51C` is entirely empty in this release
(checked: column total = 0.0), and `P51G` exists only on the *product* axis (which
capital goods were produced), not per using industry.

A capital-output ratio therefore **cannot be computed** from this table. Rather
than invent sectoral heterogeneity, the data layer uses the flat value
`kappa = 2.5` permitted by `docs/DATA_SCHEMA.md` §3.4 and `docs/PROTOCOL.md` §4,
and flags it: `kappa_fallback: true`, with an explanatory note in meta.

*Consequence:* every result inherits a common capacity-to-output ratio. Because
all six mechanisms and all sectors share it, the *comparison* between mechanisms
is unaffected; only the absolute level of capital utilisation is. This belongs in
the limitations section of the paper. It also means the B4/B3 capacity shocks
operate on a normalised capacity base.

The alternative — deriving κ from investment goods produced (`P51G_i / x_i`) —
was rejected: that is a *supply* share (how much capital equipment a branch
makes), not a capital stock, and it is ≈ 0 for most service and consumption
branches, which would be economically meaningless.

### 3.3 Welfare weights and essentials

`docs/PROTOCOL.md` §4 fixes `w_i = 1` for non-essentials and `1.2` for the
essentials tier. The essentials tier is defined from the mapping's `aggregate`
tags (`food_agriculture`, `energy_utilities`, `public_services`), giving four
sectors: **c01** (agriculture), **c03** (food), **c12** (energy/water/waste),
**c19** (public administration, education & health).

Two judgement calls, both deliberate and both declinable by an experiment file:

* Refined petroleum (c06) is **not** in the tier. It is an energy *input*, but
  household direct consumption is 1.15 % of the total versus 3.0 % for
  electricity/gas and 9.2 % for food; and it is consumed mostly indirectly.
* Transport (c15) is **not** in the tier. `docs/DATA_SCHEMA.md` §3.6 allows
  "transport of people if identifiable" — it is not identifiable here, because
  `H49` mixes passenger and freight and `H51` mixes them too.

Healthcare is inside c19 together with education and public administration, so
"essentials" for c19 should be read as "public/merit services", not "health
only". If a scenario needs a narrower tier it should override `essentials` rather
than assume the default matches a normative definition.

---

## 4. Reproducing this

```bash
cd /Users/ceontou/Documents/code/planlab
source scripts/env.sh

# full rebuild (download → parse → aggregate → synthetic → manifest); ~3 s warm
python -m planlab.data.build --source both

# Eurostat substrate only, from the cache
python -m planlab.data.build --source eurostat

# force a fresh download and rebuild everything
python -m planlab.data.build --source both --force

# thin wrapper, identical behaviour
python scripts/fetch_data.py --source eurostat

# offline consumer path (never touches the network)
python -c "from planlab.data import load_technology; t=load_technology('eurostat',2020,20); print(t.A.shape, t.base_output.sum())"

# tests
.venv/bin/python -m pytest tests/test_data_*.py -q
```

Artifacts written:

```
data/processed/eurostat_2020_n20_A.csv         20x20 coefficients, header row = sector codes
data/processed/eurostat_2020_n20_meta.json     provenance + all vectors (schema §2 keys)
data/processed/eurostat_2020_n20_labels.csv    index, code, name_en, name_zh, aggregate, ...
data/processed/eurostat_2020_n20_vectors.csv   the value vectors, comma high precision
data/processed/eurostat_2020_n65_Z.csv         audit: the unsaggregated 65x65 value block
data/processed/eurostat_2020_n65_vectors.csv   audit: the 65-branch vectors
data/processed/eurostat_2020_n65_meta.json     audit: per-branch identity + A checks
data/processed/synthetic_2020_n20_*            same schema, different structure
data/metadata/sector_map_eurostat64_to_20.csv  the mapping (versioned, hand-authored)
data/metadata/manifest.json                    sha256/url/timestamps/sizes/versions/checks
```

`data/processed/eurostat_2020_n65_*` is **additional** to `docs/DATA_SCHEMA.md`
§2. It exists so the pre-aggregation identity can be re-verified offline from
committed files rather than from the (git-ignored) 100 KB raw payload.

The whole pipeline is idempotent: a second run reads the cached payload,
re-verifies its sha256, and rewrites byte-identical artifacts (asserted in
`test_second_build_reuses_the_cached_raw_payload`).

---

## 5. Verification

| check | result |
|---|---|
`A x + y = x` at 65 branches | max abs residual **0.090 MIO EUR**, relative **5.0e-8** (tolerance 1e-6)
`A x + y = x` at 20 sectors | max abs **0.41 MIO EUR**, relative 1.2e-7 (rounding accumulates across 4 225 cells)
`Σ_i a_ij < 1`, 65 branches | max **0.6783**, min 0.0
`Σ_i a_ij < 1`, 20 sectors | max **0.6749** (food), min **0.2522** (public admin/education/health)
spectral radius ρ(A), 20 sectors | **0.4676** (synthetic 0.4648) → productive, `(I−A)⁻¹` finite and non-negative
aggregate intermediate share | **0.444** of gross output (both substrates, by construction)
`Σ_j labor_j · x_j` | 6 613 781.3 MIO EUR = total compensation of employees exactly
no NaN / no negative flows | pass (payload has no negative intermediate flows)
implied GVA vs published EU-27 GVA | 12 147 083 vs 12 168 846 MIO EUR → **−0.179 %**
zero-output branches | 1: `CPA_U` (extraterritorial organisations), correctly given a zero column

ρ(A) < 1 is the precondition for the engine's short-side fixed point to have a
finite solution. It is asserted in the test suite and recorded in
`meta["checks"]["productive"]`, so a future mapping change cannot quietly ship an
explosive economy.

The GVA cross-check is the important one: it is an *independent* Eurostat dataset
(`nama_10_gdp`), so it catches a systematic mis-read that internal identity
checks cannot. It is fetched during the build and recorded verbatim in
`manifest.json → validation`; if the build is offline the field records
`"status": "unavailable"` rather than inventing a number.

---

## 6. Synthetic substrate

`data/processed/synthetic_2020_n20_*` is a **documented generator**, not a model
of any real economy, and must never be presented as measured data.

**Calibrated to the Eurostat aggregate** (copied verbatim): gross output per
sector, each sector's intermediate budget `Σ_i Z_ij` (hence its GVA share), total
final demand, total household consumption, the compensation and GVA vectors, and
the per-sector final-demand composition.

**Drawn** (seed `20200101`, `numpy.random.default_rng`): the intermediate flow
matrix, from a log-normal topology
`K[i,j] = exp(1.5·[i=j] + 0.6·[A^eurostat_ij > 0] + N(0, 0.9))` rescaled with
iterative proportional fitting (RAS) onto the calibrated row and column margins.
RAS preserves both margins exactly, so `Σ_i a_ij` is identical to the real table
and the accounting identity closes by construction (residual 1e-5 MIO EUR,
3e-12 relative).

Result: same schema, same margins, genuinely different structure — relative
Frobenius distance `‖A_real − A_syn‖/‖A_real‖ = 0.80`, entry correlation 0.48,
only 2 of the 10 largest input linkages in common.

Known calibration residual, recorded in meta: because the published table is
rounded to 0.01 MIO EUR, `x − Z·1` differs from the reported final-demand total
by ≈ 2 MIO EUR (1.4e-7 relative). The generator preserves the column budgets
(which fix the sectoral GVA shares, the property that matters for the
coefficient matrix) and lets that sub-ppm difference pass into the final-demand
total. Household consumption matches exactly.

---

## 7. Licence

Eurostat data are re-usable for commercial and non-commercial purposes under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** licence, per the
Eurostat legal notice implementing Commission Decision 2011/833/EU on the reuse
of Commission documents. This is the basis given in `docs/DATA_SCHEMA.md` §1.

Attribution used in this repository and required in any publication derived from
it:

> Source: Eurostat, *Symmetric input-output table at basic prices (product by
> product)* [`naio_10_cp1700`], EU-27, 2020, extracted 2026-11-01.

The derived artifacts in `data/processed/` and `data/metadata/` are distributed
under the same CC BY 4.0 terms; the code that produces them is MIT (repository
licence). Attribution must not suggest that Eurostat endorses PlanLab or its
conclusions.

---

## 8. Deviations from `docs/DATA_SCHEMA.md`

Four deviations, all reported rather than hidden. None change the frozen schema's
*meaning*; two add fields and two are clarifications of wording that conflicts
with the core contract.

### 8.1 `base_final` is total final demand, not household consumption

`docs/DATA_SCHEMA.md` §2/§3.7 says `base_final` is "the column of household final
consumption". `planlab.core.io_model.build_technology` requires
`base_final == base_output − A @ base_output` (relative error < 1e-6) and
`core/engine.py` documents the field as *total final demand*. These cannot both
hold unless the economy has no government, investment or exports.

Resolved in favour of the **core contract**, because it is the interface the
engine actually consumes and because `A x + y = x` must close for the base year:

* `meta["final_consumption"]` — household final consumption `P3_S14` (the schema key, unchanged)
* `meta["base_final"]` — total final demand `x − A x` (what `Technology.base_final` receives)
* `meta["base_final_basis"] = "total final demand (x - A x)"` and `meta["final_demand_total"]` documents the same vector

The loader prefers `base_final`, then `final_demand_total`, then
`final_consumption` (legacy). `planlab.core.io_model.build_technology` accepts
the resulting object unchanged — there is a test for it.

### 8.2 65 source branches, not 64

As described in §1.1: the A\*64 nomenclature has 64 entries, but this release
also publishes `CPA_U`. All 65 are mapped; the file is still named
`sector_map_eurostat64_to_20.csv` to match the frozen spec, and the mapping's
`mapping_note` column explains each row.

### 8.3 Additional artifacts

`data/processed/<source>_<year>_n65_{Z,vectors,meta}.json|csv` and
`data/processed/*_n20_vectors.csv` are additions (audit trail and comma-
precision vectors). They do not replace any specified file.

### 8.4 Extra `meta.json` keys

Beyond the required keys the meta file records, for auditability:
`base_final`, `base_final_basis`, `final_demand_total`, `compensation_of_employees`,
`gross_value_added`, `intermediate_consumption`, `imports`, `gfcf`, `exports`,
`labor_fallback`, `kappa_fallback`(+note), `checks{…}`. Nothing required was
removed or renamed.

### 8.5 Not a deviation, but worth flagging: taxes on products

The table's `D21X31` column total is 379 788 MIO EUR, whereas Eurostat's published
EU-27 2020 taxes-less-subsidies on products is 1 431 739 MIO EUR — the column is
clearly not the full national-accounts aggregate. This does **not** affect the
data layer: `A` and `x` come from the flow block and `P1`, and the implied GVA
still matches the published total to 0.18 %. Recorded here so no one later
mistakes `D21X31` for a tax total.
