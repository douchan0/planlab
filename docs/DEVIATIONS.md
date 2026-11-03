# Deviations from the pre-registration, and additions to it

`docs/PROTOCOL.md` is the pre-registration; `docs/AMENDMENTS.md` records the
amendments made *before* the main grid ran. This file records everything the
**experiment layer** deviates from, or adds to, those two documents, with the date
and the reason. The rule is the same: silent changes are not allowed.

Every item below is visible in `results/<block>/manifest.json` (environment,
mechanism parameters, seed counts, grid fingerprint), so a reader can check it.

---

## 1. Horizon and timing: T = 80 quarters, burn-in 30, shock at t = 40 (2026-11-01)

**Protocol:** `T = 60` quarters, 30 burn-in quarters discarded.

**Deviation:** all blocks run `T = 80` quarters with `burn_in = 30`, and the
standard capacity shock starts at `t = 40` and lasts 8 quarters (Amendment 2).

**Reason.** The amended shock is 8 quarters long; the pre-registered oscillation
diagnostics use a 20-period detrending window and the predicted oscillation period
is `2(2m+1)` quarters, i.e. up to 26 quarters at `m = 6`. With `T = 60` there would
be at most 20 post-shock quarters — less than one cycle at the longest delay, and the
H1 comparison would be measuring a transient. `T = 80` gives 40 post-shock quarters
and costs one third more run time. The first 60 quarters are untouched, so any
six-quarter statistic computed on the protocol window is a subset of these runs.

---

## 2. Recovery half-life is recomputed in the experiment layer (2026-11-01)

**Protocol:** run-level `recovery_half_life` from `planlab.core.metrics`.

**Deviation:** none in the reported statistic — the table carries both
`recovery_half_life` (core) and `recovery_half_life_own` (recomputed independently
from the per-period series in `results/<block>/runs.parquet`), and they are asserted
to agree.

**Reason.** The original core implementation mixed an absolute period number with a
window-relative index and returned `NaN` for every shock starting after the burn-in.
That was reported to the Lead and fixed in core; the independent recomputation is
kept as a permanent cross-check, because a recovery statistic that silently returns
`NaN` would quietly drop rows from a table.

---

## 3. The mechanism comparison is run at one documented operating point (2026-11-01)

**Protocol:** each mechanism at its own defaults.

**Deviation:** every cell of every block carries, in `scenario.mechanism_params`:

| parameter | value | why |
|---|---|---|
| `backlog_carry` | 0.0 | household-side; re-injects last period's unmet demand into this period's claim, a positive feedback channel unrelated to the coordination rule |
| `stock_target_ratio` | 0.10 | planner-side; a differing buffer would change what the planner is doing |
| `max_target_ratio` / `min_target_ratio` | 2.5 / 0.25 | planning bounds |
| `material_balance.revision_gain` | 0.30 | reference operating point (swept in B2/B2L/B8) |
| `material_balance.balance_rounds` | 3 | reference operating point (swept in B6/B8) |
| `tatonnement.price_gain` | 0.30 | reference operating point (swept in B2/B7) |
| `hybrid_dual_track.theta` | 0.50 | reference operating point (swept in B5) |
| `hybrid_dual_track.market_gain` | 0.25 | reference operating point |
| `rationing_priority.quota_speed` | 0.50 | reference operating point |
| `rationing_priority.revision_interval` | 1 | reference operating point |
| `lp_central.use_shadow_prices` | 0.0 | see §4 |

`assemble_block` **raises** if any cell is missing the block, so an unidentified
ranking cannot be written to `results/`. The manifest records the exact dict
(`mechanism_base_params`) and an `identification_audit` with the cell count checked.

**Reason.** The mechanisms shipped different household-side defaults
(`backlog_carry` 0.0 vs 0.30, `stock_target_ratio` 0.10 vs 0.15). Since
`backlog_carry` feeds last period's unmet demand back into this period's claim, that
channel inflates the shortage of whichever mechanism happens to have it and has
nothing to do with coordination. Measured on the noise-only block, switching to the
common block moved `material_balance` from 3.36% to 4.69% mean shortage and
`hybrid_dual_track` from 1.41% to 1.63%, i.e. it changes the ranking's magnitudes.
Knobs that *are* the institution (`capacity_ceiling`, `claim_elasticity`,
`price_step`, `market_gain`, `supply_gain`) are left at their class values and are
listed as part of the institution, not as free parameters.

---

## 4. `lp_central.use_shadow_prices = 0.0` (2026-11-01)

**Reason.** The LP's duals are Hilbert-space shadow prices; feeding them into a
demand system whose prices are consumer prices is inconsistent, and with the switch
on the information-rich benchmark collapses to `x/base = 0.76` and 29% shortage in
*calm* conditions — a modelling artefact, not a result. With it off the benchmark is
exactly `1.0000 / 0.0000` (block B1). Pinned in every cell, not left to the class
default, so results cannot move if the default changes.

---

## 5. The standard information regime (2026-11-01)

**Protocol:** `σ` and the demand process are scenario parameters, with no value fixed.

**Choice:** `info.noise = 0.02`, `demand.sigma = 0.01`, `demand.ar1 = 0.4`,
`demand.price_elasticity = 0.4` for the standard environment (B2, B3, B4, B5, B6
reference, B2L, P2), and `stock_ratio = 0.10` (Amendment 2).

**Reason.** Calibrated before the main runs so that the calm economy is quiet but
the Monte Carlo has dispersion: a no-shock run then has ~0.9% mean shortage, the
standard c12 shock peaks at ~0.19 with a seed-to-seed sd of ~0.002, and the shocked
economy's mean shortage rises monotonically with the reporting lag. The block B6
cells `sigma_low`/`sigma_high`/`obs_noise_zero`/`obs_noise_high` vary it, and B1b
uses the Lead's prescribed `σ_obs = 0.03`, `ar1 = 0.3`, `σ_dem = 0.02`.

---

## 6. New blocks and the added axes (2026-11-01)

These are additions to the pre-registered grid, requested by the Lead after the
pilot and recorded as such. They are additive: no pre-registered cell was changed.

* **B1b — information rent.** No shock, no lag, noise only (Lead's request: "the
  thesis in its purest form"). 6 mechanisms × 20 seeds.
* **B7 — price-response probe.** Exogenous ±5% price offsets with the feedback
  switched off, one good at a time, to measure the local gain `b_p` sector by sector
  and hence the analytic boundary `η*(m) = g*(m)/b_p`. 3 elasticities × 20 sectors ×
  2 offsets × 3 seeds. The Lead's identification review dropped the regression route
  (simultaneity) and kept only this displacement route.
* **B8 — material-balance decomposition.** `revision_gain ∈ {0, 0.1, 0.3, 1.0}` ×
  `balance_rounds ∈ {0, 1, 3}` × 16 seeds, noise-only. The finding is the
  interaction: neither the revision alone nor the balancing alone costs output.
* **B2L — priority ladder** (exploratory): production-first vs consumption-first,
  4 lags × 20 seeds.
* **B6 labour-slack axis**: `labor_supply_factor ∈ {1.05, 1.15, 1.40, 2.0}` against
  three B8 conditions, 10 seeds each, because the B8 cause turned out to be labour
  slack rather than inventory.
* **B6 planning-effort axis**: `balance_rounds ∈ {0, 6}` instead of claiming a
  planning-effort result from `mean_iterations` (see §9).

---

## 7. Amplification: two statistics, one primary (2026-11-01)

**Protocol §5:** `amplification = sum_i Δshortfall_i / Δcapacity_shock`, evaluated
at the shock peak.

**Deviation.** The pre-registered ratio is reported
(`shock_amplification`, named "distributional ratio" in the tables) **and** the
like-for-like statistic `shock_output_loss` = peak aggregate output loss, with
`shock_output_amplification = shock_output_loss / capacity-loss share` is the
**primary** H4 statistic.

**Reason.** When the denominator was the *unweighted* worst capacity cut, a 50% cut
to a sector that is 1.7% of output was treated as a 50% shock to the economy, and
the ratio exploded. The denominator is now output-weighted (core fix), and the
multiplier still divides by a sector's output share, so it explodes for tiny sectors
(87× for a 0.27%-of-output sector) — a small-denominator artefact. The output-loss
statistic has no such denominator and is what the tables and the H4 test use; the
multiplier is reported as secondary with the caveat in the table note.

---

## 8. H1 is tested on prices, H1′ on targets (Amendment 1, implemented here)

H1's stability-boundary test uses `tatonnement.price_gain` with
`demand.price_elasticity > 0`; the plan-revision loop of `material_balance` is
reported as **H1′** ("taut plans are self-defeating") because raising output targets
raises intermediate demand and therefore *raises* shortage — a positive loop, which
by Corollary 4 has no admissible gain. The measured `b_p` is small enough that the
price loop never reaches its bound in this economy; the tables say "boundary not
approachable" rather than claiming a crossing.

---

## 9. `mean_iterations` is a diagnostic, not a planning-effort result (2026-11-01)

`mean_iterations` is 1.0 in every calm cell (and ~1.5 only in the consumption-first
collapse): the feasibility programme is solved only when the target vector is
infeasible, so the count is not a measure of planning effort. It is reported in the
tables as a diagnostic and is **not** used for any hypothesis. Where the paper wants
a planning-effort axis, it uses `balance_rounds` (B6/B8), which is a genuine
institution-level knob.

---

## 10. Common random numbers: common *draws*, not common observations (2026-11-01)

All cells of a block share one `scenario.id`, and the engine seeds its RNG from
`(scenario.id, seed)`. What is therefore common across mechanisms is the
**exogenous stream**: capacity-shock factors and the AR(1) / measurement-noise
draws. Realised demand is *not* common, because it carries three endogenous
channels — income elasticity, price elasticity, and the backlog carry-over of last
period's unmet demand. Every table and manifest phrase says "common shock
realisations and common noise draws"; no artefact claims identical demand paths
(see `tests/test_exp_runner.py` for both directions: identical when all three
channels are switched off, different when they are on).

---

## 11. Provenance: no git history for the outputs themselves

The checkout is a git repository from 2026-11-01 (initial commit `bb54a16`).
Manifests record `git_commit`, a `git_dirty` flag and a content `code_fingerprint`
computed **at run time** per cell (`code_fingerprints_seen` lists every one, and
`mixed_code_versions` is true if the directory mixes them). Bulk per-period output
(`runs.parquet`) is gitignored; manifests, `summary.csv`, `aggregate.csv`, tables and
figures are what get committed.

---

## 7. The H1 stability criterion was reinterpreted after seeing that its first arm cannot discriminate (2026-11-01, recorded by the Lead)

**Protocol §6** defines "unstable" as *both* (i) the lag-1 autocorrelation of the
detrended price deviation exceeding 0.5 and (ii) the amplitude exceeding the
`k = 0` amplitude by at least three standard deviations.

**What happened.** Arm (i) is *degenerate in this design*: every cell in the B2 grid
has `acf₁` between 0.90 and 0.95, including the cells with no feedback at all,
because a quarterly series driven by AR(1) demand and a decaying capacity shock is
strongly autocorrelated whether or not any control loop is near a stability
boundary. Reading the criterion literally therefore declares every cell unstable,
which is not a test of anything. Arm (ii) retains discriminating power and does not
rise with lag.

**Deviation.** The paper rests the H1 null on arm (ii), reports arm (i) as a
diagnostic, and says in the text that the criterion was reinterpreted after the
degeneracy became visible.

**Disclosure.** This is a change to a pre-registered *analysis rule* adopted after
seeing the data, and it is recorded as such rather than presented as a
straightforward pass. It is the one place in the paper where a deviation was
discovered during analysis rather than before it. A reader who insists on the literal
criterion should conclude that the criterion failed to discriminate — not that the
boundary was crossed. That reading is also supported by the measured loop gain, which
lies one to two orders of magnitude below `g*(k)` in every sector at every lag
(§6.4), and by the fact that the amplitude does not increase with lag.

**Consequence for H1's status.** H1 is reported as *not supported in this economy*,
not as falsified and not as confirmed. The boundary is verified mathematics
(`verification/`, agreement `2.9e-11` over `m = 0..30`) that this calibration cannot
reach.

---

## 8. Provenance repairs made at the final build (2026-11-01, Lead)

Two citations in the manuscript pointed at artefacts that did not exist at the time
of writing, and both were repaired rather than removed:

* `results/tables/t06_propagation.csv` — the rank correlations behind §6.5. The
  numbers were correct (`+0.5128/+0.5669/+0.4135` for forward linkage,
  `+0.0195/+0.0165/−0.0406` for the Leontief multiplier) but the file had not been
  written. It is now generated reproducibly by `t_h4_correlations` in
  `scripts/paper_tables.py`, and the regenerated values match the paper to four
  decimals.
* `paper/tables/dual_track.csv` — the `theta` column was corrupt (every row read
  `0.3`) because the table generator was reading the mechanism's `market_gain`
  instead of the dual-track share. Fixed in `scripts/paper_tables.py`; §6.7 was
  rewritten from the corrected file and no longer reads the table positionally.

Both were defects in the Lead's table-assembly code, not in the model or the
experiment layer.

---

## 9. H3 added to the adjudication table by the Lead (2026-11-01)

**Gap.** `results/tables/t00_hypotheses.csv` adjudicated H1, H1', H2, H4 and H5 —
five of the six pre-registered hypotheses. H3 was stated in `docs/PROTOCOL.md` and
in the paper's §5 as a falsifiable prediction, and the paper's §6.5 adjudicated it,
but the machine-readable table did not carry it. The paper and the table therefore
disagreed about which hypotheses the design covers, which is exactly the kind of
disagreement the protocol's "every number traces to a file" rule is meant to prevent.

**Why the Lead rather than the experiment layer.** The experiment layer had gone
inactive by the time the gap was found, and the row is a pure derivation from
`results/B3/summary.csv`, so an inactive owner is not a reason to leave a hole.

**Row content**, derived not typed: 20 sectoral cells (one per shocked sector under
the amended 50 per cent cut, 8 quarters, 20 seeds, lag 2, gain 0.30); every cell
clears the falsifier, so 20/20 "significant"; `key_statistic` is the **weakest** cell
mean, 0.011624, which is 2.32x the 0.5 per cent threshold; the strongest cell is
0.06038 (12.08x); the mean over cells is 0.02935. No p-value is recorded because H3's
falsification condition is a threshold, not a hypothesis test — recording a p-value
there would have implied an inferential procedure the protocol never specified.

**Verification.** `verification/check6` fails if the adjudication table does not carry
six hypothesis rows; that check now passes, and the verifier independently re-derived
the same statistics from `results/B3/summary.csv` before the row existed.

## Audit revision (2026-11-03)

This revision follows the findings in [the project audit](项目审计报告.md).
Closing inventory is capped at the total storage limit. Opening inventory is
already included in remaining supply. A zero labour endowment now binds.
Ordinary scenarios reject price shocks explicitly. Controlled price probes
remain available through the experiment layer.

The analysis is a revision of the historical protocol. B2 compares
price amplitude separately by elasticity and seed at each tested reporting lag
against lag zero. The one-sided price update cannot establish the signed scalar
threshold. H2 compares the per-seed slopes of mean utility points at reference
gain 0.30 and price elasticity 0.4. H3 is a descriptive threshold check with no
p-value. B3 matches each shock to a same-seed no-shock trajectory. H4 uses the
largest incremental output loss from periods 40 through 60 and averages the
resulting amplification ratios over fixed sectors within each seed. The older
base-output comparison remains in the run data. H5 compares each candidate to
both endpoints, with Holm correction across the candidate tests. Earlier
reference results and the process report document the previous analysis.
