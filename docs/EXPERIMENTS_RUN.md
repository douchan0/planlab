# What was run, and how to reproduce it

This is the experiment log of the PlanLab simulation wave: what the grid is, how
many runs each block entails, the measured wall time, which institutions were
present, the headline numbers, and the exact commands that regenerate everything.
Every number below comes from `results/`; the tables and figures are generated, never
typed.

Companion documents: `docs/PROTOCOL.md` (pre-registration),
`docs/AMENDMENTS.md` (amendments made before the run),
`docs/DEVIATIONS.md` (everything the experiment layer deviates from or adds).

---

## 1. The grid at a glance

**Grid fingerprint:** `d2b4b85a7ba96e98` (recorded in every manifest).
**Code fingerprint:** at run time, per cell (see `code_fingerprints_seen`).

| block | what it is | cells | seeds | runs | measured compute |
|---|---|---|---|---|---|
| B1 | calibration: 6 mechanisms × 2 substrates, deterministic | 12 | 1 | 12 | 0.4 s |
| B1b | information rent: noise only, no shock, no lag | 6 | 20 | 120 | 4.9 s |
| B2 | gain × lag × mechanism (Amendment 3) | 204 | 20 | 4 080 | 192 s |
| B3 | propagation: 20 shocked sectors × 3 mechanisms | 60 | 20 | 1 200 | 24 s |
| B4 | rationing vs shadow price at 10/30/50 % capacity loss | 6 | 20 | 120 | 2 s |
| B5 | dual track: 11 plan shares θ × 2 lags | 22 | 20 | 440 | 14 s |
| B6 | robustness (substrate, gains, noise, A-perturbation, weights, **labour slack**, balance rounds) | 26 | 20 | 520 | 10 s |
| B7 | price-response probe: b_p per sector (4 elasticities) | 160 | 3 | 480 | 5 s |
| B8 | material-balance decomposition: gain × balancing rounds | 12 | 16 | 192 | 7 s |
| B2L | priority ladder (exploratory) | 8 | 20 | 160 | 10 s |
| P2 | target-response probe (sign of the plan-revision loop) | 40 | 1 | 40 | 1 s |
| **total** | | **556** | | **7 364** | **≈ 190 s** |

Measured on the development machine (Apple silicon, 8 cores, 7 worker processes):
**7 364 runs, ≈ 2 620 CPU-seconds of simulation, 471.8 s wall for the whole
end-to-end pipeline** (`CLEAN=1 bash scripts/reproduce.sh`: blocks + tables +
figures + tests), i.e. ~50 ms of wall time per run and ~350 ms of CPU per run at 20
sectors and 80 quarters. The per-block times in the table above are from a
less-loaded run and sum to 190 s; the clean reproduction was measured under load
from the other workstreams, which is why it is the number quoted for acceptance.

**Mechanisms present in the final run:** all six —
`hybrid_dual_track, lp_central, market, material_balance, rationing_priority,
tatonnement`. No cell was skipped: `cells_failed = 0` and
`cells_skipped_missing_mechanism = 0` in every block manifest.

---

## 2. The standard environment

Fixed by `docs/PROTOCOL.md` §4 as amended, and recorded in `manifest.environment`:

* 20-sector Eurostat 2020 symmetric input–output table; 80 quarters, burn-in 30.
* Capacity shock: size 0.50 (a 50 % cut), duration 8 quarters, at t = 40, in the
  sector the block varies (standard sector `c12`).
* `stock_ratio = 0.10`.
* Standard information regime: measurement noise σ = 0.02, AR(1) demand noise
  (ar1 0.4, σ 0.01), price elasticity 0.4.
* One documented **reference operating point** for every institution
  (`manifest.mechanism_base_params` + `identification_audit`): the common block
  (`backlog_carry 0`, `stock_target_ratio 0.10`, target ratios 2.5/0.25) plus
  `revision_gain 0.30`, `balance_rounds 3`, `price_gain 0.30`, `theta 0.50`,
  `market_gain 0.25`, `quota_speed 0.50`, `revision_interval 1`, and
  `lp_central.use_shadow_prices = 0`. Blocks that sweep a knob override that knob
  and nothing else; `assemble_block` **raises** if a cell is missing the block.

---

## 3. Headline numbers

### 3.1 B1 — the implementation test

All 12 deterministic runs return `x/base = 1.000000`, mean shortage `0.000000`,
plan fulfilment `1.000000`, identical cumulative welfare within each substrate
(12 978.440262 on the real table, 1 121.595320 on the toy fixture), and a maximum
absolute accounting residual of **0** (the engine would raise above 1e−8). Both
substrates agree, so the implementation is not substrate-specific.

### 3.2 B1b — the information rent, with no shock at all

20 seeds, noise only; `x/ȳ` is output relative to the base year:

| institution | x/ȳ | mean shortage | mean welfare | spoilage |
|---|---|---|---|---|
| hybrid dual track | 0.9951 | 1.63 % | 259.29 | 0.00 |
| tatonnement | 0.9938 | 1.56 % | 257.22 | 1.86 % |
| market | 0.9908 | 1.62 % | 259.20 | 0.00 |
| rationing priority | 0.9822 | 1.65 % | 259.02 | 0.00 |
| central LP | 0.9710 | 1.64 % | 258.78 | 0.00 |
| material balance | **0.9371** | **4.69 %** | **256.82** | 1.23 % |

The shortage column is nearly invariant across institutions while output and
welfare are not: with income-elastic demand, a mechanism that destroys output also
destroys the demand that would have been recorded as unmet. Read the ratios, never
the shortage level alone (table note in `t02_noise_only_information_rent`).

### 3.3 B2 — delay × mechanism at the reference gain

Mean shortage rises monotonically with the reporting lag for every institution; the
slope of the welfare loss against the lag is steepest for `lp_central`
(0.00452 per quarter, cluster-robust SE 6.8e−5), then `rationing_priority`
(0.00415), and flattest for `tatonnement` (0.00039) and `material_balance`
(0.00061). **H2 is not falsified**: the institution that needs the freshest
information degrades fastest. The ranking on the shortage *level* is different from
the ranking on the welfare *loss*, which is why both columns are reported.

### 3.4 B3 — propagation and incidence

Primary statistic `shock_output_loss` (peak aggregate output loss) and its ratio to
the output-weighted capacity loss. Output loss spans 1.3–14.3 % across the twenty
shocked sectors for `material_balance`; the mean amplification is 7.9 (H4's
"amplification > 1" is supported, p = 9.5e−7). The *forward* linkage (intermediate
share of a sector's own output) is the predictor of propagation, not the Leontief
column-sum multiplier — the latter has a small-denominator problem (an 87×
"amplification" for a sector that is 0.27 % of output) and is reported only as a
secondary, caveated statistic.

Incidence of the shortfall for the shocked good (`material_balance`, mean over the
20 sectors): the household share of the rationed shortfall is **0.735**,
government/investment 0.265, intermediate users **0** by construction (they are
served first), and the peak household unmet rate on the shocked good is **0.864**
— i.e. up to 86 % of household demand for the shocked good goes unmet at the trough
while production loses it through the plan gap (intermediate service rate 0.78).

### 3.5 B7 / H1 — the stability boundary is barely reachable

Measured local gain `b_p = −ds/d log p`, per sector, from exogenous ±5 % price
offsets with the feedback off:

| price elasticity | b_p mean | b_p min | b_p max |
|---|---|---|---|
| 0.2 | 0.125 | −0.003 | 0.251 |
| 0.4 | 0.197 | −0.003 | 0.478 |
| 0.8 | 0.342 | −0.003 | 0.787 |
| 2.0 | 0.616 | −0.004 | 1.526 |

`b_p` scales roughly linearly with the elasticity, as it must if the price channel
runs through household demand. The loop gain is `g = η·b_p`, so the largest tested
gain (1.6) achieves `g ≤ 0.32` at ε = 0.4 and `g ≤ 0.55` at ε = 0.8, against the
bound `g*(m) = 2 sin(π/(2(2m+1)))` = 1.00, 0.618, 0.445, … for m = 1, 2, 3, …
The boundary is therefore **approachable only at long delays** (m ≥ 5 at ε = 0.4,
m ≥ 3 at ε = 0.8); the pre-registered instability rule fires only in those corners,
and one of the two firing points fires *below* the bound (`rule_fired_below_bound`
in `t05_stability_boundary`), which says the rule is not specific. The honest
statement: in this economy the delay bound is at the edge of reachability; price
adjustment is stable and therefore inert at every plausible speed, and the binding
constraint on administrative coordination is the sign of the quantity loop, not the
speed of the price loop.

### 3.6 B8 — where the material-balance loss comes from

`x/ȳ` by revision gain × balancing rounds (no shock, noise only):

| gain \ rounds | 0 | 1 | 3 |
|---|---|---|---|
| 0.0 | 0.9999 | 0.9999 | 0.9999 |
| 0.1 | 0.9999 | 0.9998 | 0.9995 |
| 0.3 | 0.9999 | 0.9467 | 0.9333 |
| 1.0 | 0.9999 | 0.8869 | **0.8738** |

Neither the revision alone nor the balancing alone costs anything: the loss is the
*interaction*. The cause carried in the same table: the balancing step is a
downward-only filter (`plan_lowering_periods` rises from ~11 to ~66 quarters out of
80 as the gain rises), investment — a low-priority claimant — falls from 0.0500 to
0.0434 of base output, the capital stock ends at 0.903 of base, labour utilisation
falls to 0.71, and the transitory information error is capitalised. The labour-slack
axis in B6 turns this into a conditional result: at 2.0× labour supply the same taut
plan costs nothing, because it can be executed.

### 3.7 B5 — dual track: H5 is falsified

Welfare rises weakly and monotonicity cannot be rejected (lag 0: 252.21 at θ = 0 to
252.33 at θ = 0.5 and 252.33 at θ = 1; lag 3: 251.65 → 251.67 → 251.74), with a
spread of ~0.3 against a seed standard deviation of the same order. There is no
detectable interior optimum; price volatility falls to exactly zero at θ = 1.
Reported as a clean negative.

---

## 4. Artefacts

```
results/
  manifest.json                      # global: wall time, fingerprints, block table
  <BLOCK>/manifest.json              # versions, hashes, seeds, identification audit
  <BLOCK>/summary.csv                # one row per (cell, seed): run-level metrics
  <BLOCK>/aggregate.csv              # one row per cell: mean/sd/95% CI across seeds
  <BLOCK>/runs.parquet               # one row per (cell, seed, period) — gitignored
  <BLOCK>/sector_panel.parquet       # per (cell, seed, sector), incl. incidence
  <BLOCK>/cells.json                 # the full design (every scenario, every seed)
  <BLOCK>/audit_records.npz          # ≥200 stratified (run, period) raw arrays
  <BLOCK>/audit_index.csv            # which records are in the audit file
  tables/t00..t13_*.csv|.tex         # every table of the paper, plus its booktabs
  tables/tests.csv                   # every test statistic (59 rows)
  tables/manifest.json               # what was written, from which results
  figures_index.csv                  # every figure, its block and its generator
figures/fig_*.png|.pdf               # 11 figures, 300 dpi + vector
scenarios/*.yaml                     # the 16 standard, standalone-runnable scenarios
```

The audit sample is the verifier's entry point: each record carries `x, Z, C, G,
invest, inventory_end, demand, unmet, capacity, plan_targets, supply, price,
spoilage` **and its own substrate** (`A`, `base_output`, `base_final`,
`essentials`, `consumption_weights`, `n_sectors`, `sectors`), so
`Z = A·diag(x)`, the balance identity (including spoilage), `C ≤ demand`,
`x ≤ capacity` and the metric formulas can all be re-derived from the file alone.

---

## 5. How to reproduce

```bash
cd planlab && source scripts/env.sh

# everything, from scratch (simulation → tables → figures → tests)
bash scripts/reproduce.sh

# or step by step
python scripts/run_all.py                 # blocks in order, resumable
python -m planlab.experiments.aggregate   # tables + tests.csv
python -m planlab.experiments.figures     # figures + figures_index.csv
python -m planlab.experiments.aggregate --help
```

`make reproduce`, `make experiments`, `make tables`, `make figures`, `make paper`,
`make test` are the same commands. A single cell can be re-run with the CLI:

```bash
python -m planlab.cli run scenarios/lag_material_balance_k3.yaml
```

Resumability is per cell and keyed on a content hash of the cell definition, so
re-running `run_all.py` after changing nothing is a no-op, and after adding a
mechanism it runs exactly the missing cells. Determinism: every RNG is seeded from
`(scenario.id, seed)`, every block shares one `scenario.id` across its mechanisms
(common shock realisations and common noise draws — *not* identical demand paths,
see `docs/DEVIATIONS.md` §10), and `pytest tests/test_exp_*.py` checks the grid,
the runner, the aggregation and the figures.

---

## 6. Test status and measured wall time

* `pytest tests/ -q` → **292 passed** (the experiment layer contributes
  `test_exp_grid.py`, `test_exp_runner.py`, `test_exp_aggregate.py`,
  `test_exp_figures.py`).
* Clean end-to-end: `CLEAN=1 bash scripts/reproduce.sh` produced this directory in
  **471.8 s** (`results/manifest.json`: `total_wall_time_s`), of which 7 364 runs and
  2 619 s of worker CPU; the test suite is 7 s. The acceptance budget is 60 minutes.
