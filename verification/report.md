# PlanLab independent verification report

Generated: 2026-11-01 17:19:28 +0800  
Repository: `/Users/ceontou/Documents/code/planlab`  
git commit: `647fdf12fd5dcd688c01ec8193fb0c2b0b395364` (dirty: True)  
Python 3.10.13 on macOS-15.7.7-arm64-arm-64bit  
Artefacts audited (mtime): paper/main.md 2026-11-01 17:14:23; paper/tables 2026-11-01 17:13:16; results/tables/tests.csv 2026-11-01 16:55:18; results/tables/t00_hypotheses.csv 2026-11-01 16:55:18  
Wall time: 64.3 s  
Checks: 5 PASS, 1 FAIL, 0 SKIP, 0 ERROR

This suite is written by an independent verifier. It re-derives the theorem, re-implements a period from the written specification, audits the raw records behind the paper's numbers, re-runs sampled cells, and re-computes the statistics with its own code. It deliberately does not import the project's engine, metrics or aggregate modules to judge their output; where it does use project code it is only for *inputs* (the technology matrix) or for *records* (the stored arrays).

## Findings reported to the Lead (the verifier's own record)

An independent verifier is only worth having if its record shows what it caught. These are the defects found in the Lead's work, in the order they were reported.

### F1 — error (documentation) (2026-11-01)

- Where: docs/notes_delay_theorem.md (lines 50-51 and the table at 163-170)
- Finding: The numeric list and the table shifted the stability boundary by one lag index against the note's own formula: it gave g*(1)=2, g*(2)=1, g*(3)=0.618, g*(4)=0.445, whereas g*(m)=2 sin(pi/(2(2m+1))) gives g*(1)=1, g*(2)=0.618034, g*(3)=0.445042 and 2 is g*(0).
- Evidence: check 1 (three independent routes: Schur-Cohn bisection, companion-matrix spectral radius, time-domain growth exponent), m=0..30.
- Resolution: FIXED by the Lead the same day; the note now agrees with the paper and records that the correction came from this check.

### F2 — error (documentation) (2026-11-01)

- Where: docs/notes_delay_theorem.md line 105
- Finding: "Every period of delay more than halves the admissible gain" is false: 2 -> 1 halves only at the first step; 1 -> 0.618 is a 38% cut and the ratios approach 1 (g*(m) ~ pi/(2m+1)). The bounded object is m*g*(m) <= pi/2.
- Evidence: check 1 numbers for m=0..30; the paper already stated the correct version.
- Resolution: FIXED by the Lead the same day.

### F3 — gap in proof (not an error) (2026-11-01)

- Where: docs/notes_delay_theorem.md, proof of Proposition 1 (lines 84-90 of the original)
- Finding: The proof asserted that j=0 gives the smallest gain without ruling out j>=1, for which theta_j > pi and sin(theta/2) decreases. The statement is true (g_j = 2 sin(pi(1+4j)/(2(2m+1))) >= g_0 for 0<=j<=m, with equality also at j=m, the conjugate pair), but the note did not show it.
- Evidence: check 1: the unit-circle identity and the root-free boundary agree with the closed form for m=0..30; the paper's Section 4 already contains the correct argument.
- Resolution: CLOSED by the Lead with the paper's x_m in [x_0, pi-x_0] argument.

### F4 — error (scientific explanation) (2026-11-01)

- Where: Lead's proposed explanation of the material_balance gain x rounds interaction
- Finding: The interaction is real (gain 0 / rounds 3 -> 1.0000; gain 1 / rounds 0 -> 1.0000; gain 1 / rounds 3 -> 0.9089 +- 0.0073), but the stated cause — feasible_output being asymmetric in the *inventory* estimate — is refuted in the regime that produces the loss: R(0.9 r) = R(r) = R(1.1 r) to 0.0e+00 in every one of the mechanism's balancing calls (max goods-balance tightness 0.25; labour used/L = 0.99-1.00), removing the inventory estimate entirely (r_est = 0) does not change the loss (0.9083 vs 0.9087), stock_ratio 0 and 0.5 do not change it (0.9061 / 0.9110), while labour supply x3 removes it completely (1.0000). The asymmetry is real only at the knife-edge goods-balance frontier (0 gain / -8.7% loss), where the mechanism never operates. The correct mechanism is the one-sided projection onto the LABOUR-constrained feasible set, and the result is conditional on labor_supply_factor = 1.15.
- Evidence: check 6 (6a-6d); replicated independently by the Lead (factor 1.15 -> 0.8698, 1.40 -> 0.9922, 2.0 -> 1.0000).
- Resolution: OPEN at the time of writing: the paper's mechanism section was being rewritten to the labour-projection version and labor_supply_factor was added to B6 as a treatment. Re-run check 6 after the rewrite.

### F5 — bug (results artefacts, not the model) (2026-11-01)

- Where: results/<block>/audit_records.npz
- Finding: For the 6-sector cells the runner writes `A` FLATTENED into a fixed 20x20 buffer (the 36 finite entries are the row-major 6x6 matrix, the other 364 are NaN), while x/Z/... are NaN-padded but correctly placed and `sectors` is empty. Any naive reader gets a NaN-filled A for toy cells.
- Evidence: check 3 loader: A.shape == (20,20) with 36 finite entries for a record whose n_sectors == 6; reshaping the finite entries reproduces toy_technology(6).A exactly (max |diff| = 0.0).
- Resolution: FIXED before the final grid: the final audit files store A in the [:n, :n] block (the remaining padding is NaN but the valid prefix is correct); check 3 now reads it directly. Recorded here because the earlier dumps were unusable for toy cells.

### F6 — bug (provenance) (2026-11-01)

- Where: results/<block>/cells.json
- Finding: The 6-sector B1 cells record scenario.technology = {source: eurostat, year: 2020, n_sectors: 20} while the authoritative cell technology is {toy: 6}. The recorded scenario therefore does not reproduce the cell (a re-run from the scenario dict alone produces 20-sector arrays), and the manifest's scenario_hashes hash the cell dict rather than the scenario.
- Evidence: check 4: re-running the recorded scenario with cell.tech gives bit-identical results, with scenario.technology it does not (0.436 relative difference in x).
- Resolution: FIXED before the final grid: cells.json now records the cell technology ({toy: 6}) instead of the eurostat placeholder; re-running from the recorded scenario reproduces the cell.

### F7 — error (paper, results interpretation) (2026-11-01)

- Where: paper/main.md §6.6 against results/B4/cells.json and paper/tables/rationing_shadow.csv
- Finding: The B4 cells are named loss50/loss70/loss90 and pass scenario shock `size` = 0.5/0.3/0.1; because Shock.factor treats `size` as the capacity MULTIPLIER, the realised capacity losses are 50%/70%/90%, not the protocol's 10%/30%/50% losses. The paper labels the 70% cell 'thirty per cent' and the 90% cell 'ten per cent', and the narrative built on those labels does not survive: the output gap reverses sign between the 50% and 70% cells (-0.9pp, rationing ahead, then +3.35pp, tatonnement ahead) and widens to +4.97pp at 90%. The protocol (docs/PROTOCOL.md §4 B4) and Amendment 2 ('keeps its pre-registered size sweep {0.10, 0.30, 0.50}') are silent on the multiplier-vs-loss ambiguity.
- Evidence: check 5's data audit and the B4 cell definitions; numbers reproduced from paper/tables/rationing_shadow.csv.
- Resolution: FIXED in the current manuscript (paper/main.md mtime in the header): §6.6 now states that the pre-registered sizes are capacity multipliers 0.1/0.3/0.5, that the realised losses are 90/70/50 per cent, and the narrative follows the rows (ranking reverses at 70 per cent, widens at 90 per cent). Verified by grep on the current file.

### F8 — overclaim (paper, H4) (2026-11-01)

- Where: paper/main.md §6.5 'Both halves are confirmed' against results/tables/tests.csv
- Finding: The H4 centrality half rests on raw Spearman p-values. The project's own test log applies the pre-registered Holm correction and marks EVERY centrality test non-significant: material_balance rho=0.5128 p=0.0208 p_holm=0.166; tatonnement rho=0.5669 p=0.0091 p_holm=0.082; priority rationing rho=0.4135 p=0.0699 p_holm=0.489. My 10,000-draw permutation test agrees with the raw p (0.024 / 0.012 / 0.077) and with the rho values to 1e-6, so the numbers are right and the INTERPRETATION is not: after the paper's own multiplicity rule, the centrality result is not significant for any of the three mechanisms. The only significant centrality rows in tests.csv use shock_output_amplification (p_holm 0.0011/0.0073/0.00024), which the paper itself demotes as a small-denominator artefact. A second, independent sensitivity: with the textbook forward-linkage measure (row sums of A) the correlation is only 0.31-0.33 (p about 0.15-0.18).
- Evidence: check 5 (5c) recomputation from summary.csv + permutation null; results/tables/tests.csv H4-centrality rows.
- Resolution: FIXED in the current manuscript (§6.5 now reports both linkage definitions, the row-sum null and the Holm caveat). One residual: the paper says two correlations survive Holm across the three institutions, while results/tables/tests.csv applies Holm within the whole B3 block (p_holm 0.166 / 0.082 / 0.489, none survive). See F16.

### F9 — factual error (paper) (2026-11-01)

- Where: paper/main.md §6.5
- Finding: "The most damaging sectors are c05, c20, c02 and c12 under all three institutions" is false for priority rationing: its top four are c20, c19, c05, c02 (c12 is fifth; c19 is second). The sentence holds for material balance and tatonnement.
- Evidence: paper/tables/propagation.csv ranks by output_loss.
- Resolution: FIXED in the current manuscript (§6.5 now gives the per-institution ordering with c19 second and c12 fifth under priority rationing).

### F10 — stale text (paper) (2026-11-01)

- Where: paper/main.md §6.7
- Finding: The paragraph warns that the theta column of the dual-track file 'repeats a single value' and says the table is read by row order. The regenerated results/tables/dual_track.csv has all 11 theta values (0.0-1.0) and 22 rows, so the caveat now describes a defect that no longer exists. The quoted welfare ranges (252.21-252.34 at lag 0; 251.64-251.74 at lag 3) do match the regenerated table.
- Evidence: results/tables/dual_track.csv, check 5's data audit.
- Resolution: REPORTED; remove the caveat or regenerate the table from the final results.

### F11 — claim not matching the table (paper) (2026-11-01)

- Where: paper/main.md §6.4, abstract; §6.1
- Finding: (a) §6.4: 'the detrended amplitude of the mean log price sits between 0.017 and 0.041 across the whole grid, the lag-one autocorrelation between 0.93 and 0.95'. At the reference gain 0.3 the amplitude spans 0.000-0.170 (market 0.124-0.170, tatonnement 0.128-0.146) and the acf spans 0.65-0.95; the 0.017-0.041 range corresponds to tatonnement at gains 0.05-0.1, not to the lag grid at the reference gain. (b) §6.1 calls 259.568805 'cumulative welfare'; it is the mean per-period welfare (cum_welfare = 12978.44). (c) The abstract says the loss 'vanishes at eighty' per cent labour slack, while §6.3 and §6.8 report it vanishing at labour_supply_factor 2.00 (100% slack), with 0.4% left at 1.40. (d) The abstract's 'a factor of two to fifty below it' mixes the most-responsive sector's 1.9 with the median sector's 12-50.
- Evidence: results/tables/main_lag_grid.csv, results/B1/summary.csv, paper/tables/labour_slack.csv.
- Resolution: MOSTLY FIXED in the current manuscript: the phrase cumulative welfare is now welfare, the abstract says the loss vanishes at a hundred per cent of labour slack, and the factor-of-two-to-fifty claim is replaced by the reachability statement. Two numbers remain off in the amplitude sentence of section 6.4: tatonnement spans 0.118-0.152 in the reference-gain table, not 0.128-0.156, and 0.249 is the pooled-grid maximum at gain 0.8, not a reference-gain cell (market 0.124-0.170 is correct).

### F12 — governance / documentation (2026-11-01)

- Where: docs/ARCHITECTURE.md §2, docs/AMENDMENTS.md, the engine
- Finding: The engine gained a bounded-storage SPOILAGE channel (Scenario.max_stock_ratio, default 2.0; S_end = min(residual, max(stock_cap - inv, 0)); spoilage = residual - S_end) which changes the balance identity to S0 + x = Z 1 + C + G + I + S_end + spoilage. It is in the results and the paper's tables but not in docs/ARCHITECTURE.md (whose 'frozen' identity omits it) nor in AMENDMENTS/DEVIATIONS, contrary to the project's own 'no silent amendments' rule. My audit finds up to 16% of supply destroyed in a single record (1.2-3.3% in the reference cells), so an external replicator implementing the documented identity would see a 1-16% discrepancy and call it a leak.
- Evidence: check 3 (identity holds to 1.6e-16 WITH spoilage, and fails by the spoilage amount without it); docs greps.
- Resolution: REPORTED; add an amendment entry and update ARCHITECTURE §2.

### F13 — table defect (paper) (2026-11-01)

- Where: paper/tables/main_lag_grid.csv
- Finding: The table has 204 rows but only 144 unique (mechanism, gain, reporting-lag) keys: the tatonnement cells at price elasticities 0.4 and 0.8 are indistinguishable because the elasticity column is absent. The values do come from results/B2 (my sorted-value traceability comparison passes), but a reader cannot tell which row is which, and §6.4's 'at the reference gain' sentences could refer to either.
- Evidence: check 5d (all headline tables re-derived from results/, worst 5e-7).
- Resolution: FIXED in the current table and verified in check 5d: the table now carries a price_elasticity column, so every row is identified.

### F14 — error (paper, H1 narrative) (2026-11-01)

- Where: paper/main.md §6.4 and the abstract, against results/tables/t12_bp_by_sector.csv and results/tables/t05_stability_boundary.csv
- Finding: §6.4 states b_p 'is of order 0.01 to 0.08 by sector' and that the loop gain 'is of order 0.02 at the median sector and reaches 0.13 only in the most responsive one'. The final measurement table says the opposite: at the reference elasticity 0.4 the median b_p is 0.1707, the mean 0.1973 and the maximum 0.4782 (at 0.8: median 0.3319, max 0.7867). At eta=1.6 the median sector's loop gain is therefore 0.273, ABOVE g*(7)=0.209 and g*(6)=0.241, and the most responsive sector reaches 0.765, above g*(m) for every m>=3. t05 records g_max_over_bound = 1.11 at m=5 and 1.51 at m=7 (elasticity 0.4), 1.23-2.62 at 0.8, `boundary_approachable` True in 6 cells and `rule_fired_below_bound` True in 2. Hence 'nowhere near the boundary' is false for the final grid, 'the median gain of 0.02 would turn at about seventy-eight quarters' should be about 31 quarters, and the abstract's 'a factor of two to fifty below it' is not supported (3.7x below g*(1) but 1.3x ABOVE g*(7) at the largest gain). The stale numbers look like b_p measured before the price-elasticity change in docs/DEVIATIONS.md.
- Evidence: check 5d (t12 and t05 re-derived from results/), and the H1 test rows in results/tables/tests.csv.
- Resolution: FIXED in the current manuscript: §6.4 now reports b_p = 0.197/0.342, maximum loop gains 0.316/0.548, the m>=5 / m>=3 crossing lags, and the abstract says the bound is 'rarely reached ... crossed only at the far end of the lag axis'. These match t12/t05 exactly.

### F15 — understated result (paper) (2026-11-01)

- Where: paper/main.md §6.4 against results/tables/tests.csv row H1
- Finding: The paper reports H1 as a null. The project's own test log records the pre-registered amplitude test at gain=1.6 (the largest), lag 6 vs lag 0: paired one-sided Wilcoxon statistic 817, p = 4.5e-12, significant_05 = yes. The null holds at the reference gain 0.3 only. Reporting the rejection that appears exactly when the loop gain is pushed toward the boundary would support rather than weaken the paper's reading of delay.
- Evidence: results/tables/tests.csv (H1 rows).
- Resolution: FIXED in the current manuscript (§6.4 now reports the Wilcoxon 817 / p = 4.5e-12 row beside the reference-gain null, framed as a reachable boundary seen from below).

### F16 — inconsistency (paper vs test log) (2026-11-01)

- Where: paper/main.md §6.5 vs results/tables/tests.csv
- Finding: §6.5 says 'After Holm correction across the three institutions, two survive at alpha=0.05 (material balance and tatonnement)'. That is true for a three-test family (p = 0.0091, 0.0208, 0.0699 -> adjusted 0.027, 0.042, 0.070), but results/tables/tests.csv applies Holm within the whole B3 block (nine centrality rows plus the other B3 tests) and records p_holm = 0.166, 0.082 and 0.489 - none survives. The protocol says 'Holm correction within each block', so the paper's narrower family needs to be stated explicitly (and the test log regenerated for that family, or vice versa).
- Evidence: check 5b (Holm values recomputed from tests.csv, 37 values, agreement 1e-12) and the permutation p-values in check 5c.
- Resolution: REPORTED; state the family and reconcile the two files.

### F17 — internal contradiction (abstract vs body) (2026-11-01)

- Where: paper/main.md abstract (line 10) vs §6.4 and §6.7
- Finding: The abstract still ends 'Two pre-registered hypotheses are falsified: no crossing of the delay boundary is observed, and the dual-track share shows no detectable interior optimum', while the rebuilt body says the opposite in both places: §6.4 states 'The boundary is not unreachable; it is reached by driving the loop hard', and §6.7 states 'the pre-registered prediction is formally satisfied and one of the two H5 tests survives Holm correction at p = 0.0045'. The abstract also contradicts itself two sentences earlier ('crossed only at the far end of the lag axis by a hard-driven loop').
- Evidence: grep of paper/main.md at the audited commit/mtime; the corresponding test rows are in results/tables/tests.csv (H1 amplitude row significant; H5 row p_holm 0.0089).
- Resolution: REPORTED; the abstract must be rebuilt from the final §6.4/§6.7 on whether H1 and H5 are falsified, or the body reverted.

### F18 — gap (adjudication table) — the Lead's own finding, confirmed (2026-11-01)

- Where: results/tables/t00_hypotheses.csv vs paper/main.md §5/§6.5
- Finding: The machine-readable hypothesis table has rows for H1, H1', H2, H4, H5, incidence and ladder but NO row for H3, although §5 states H3 as a falsifiable prediction and §6.5 adjudicates it. The adjudication itself is correct: I re-derived H3's statistic from results/B3/summary.csv (material_balance, 20 shocked sectors): mean shortage 1.162%-6.038%, mean 2.935%, ratios 2.32-12.08 against the 0.5% falsifier, 0 of 20 cells below the threshold - exactly the paper's 1.16-6.04/2.93/2.3-12.1. The amended shock (size 0.50 = 50% cut, duration 8) is documented in Amendment 2 and disclosed in §5.
- Evidence: check 6e (new): hypothesis-table completeness + H3 statistic re-derived from B3 + amendment disclosure. The check FAILS until the H3 row exists.
- Resolution: OPEN at the time of writing: add the H3 row to the adjudication table (and regenerate results/tables/t00_hypotheses.csv).

## What remains unverified

An honest verification report has to say where it stops. These are the gaps that a
reader should treat as open, not as verified by silence.

* **The paper's prose is checked against `paper/tables/*.csv`, and the tables against `results/`, but not every table was re-derived from the raw runs.** Check 4 re-executes 80 of 7 364 runs and reproduces every stored run-level metric bit-for-bit; check 3 audits 2 344 raw period records. A table built from a run outside those samples would be caught only if its inputs were among the 80.
* **The H4 primitive is audited at the summary level, not from the sector panel.** I recompute the amplification statistic from `summary.csv` and replicate the paper's rho to 1e-6, but I did not rebuild `shock_output_loss` from `sector_panel.parquet`; a bug in the per-period aggregation would be invisible to me.
* **The measured response coefficients b (price and target) are taken from `results/tables/t12_bp_by_sector.csv`, not independently re-estimated by me.** The theory-to-simulation mapping g = eta*b therefore inherits whatever the experiments layer's probe procedure does; I verified the probe contract reproduces the stored cells (check 4), not that the probe measures what it claims to measure.
* **The dynamic stability boundary of the full simulation is not independently located.** Check 1 verifies Proposition 1 for the linear recursion; whether the simulation follows it is the paper's H1 question, and I audit the stored test statistics rather than re-running the boundary search.
* **Reproducibility is tested on this machine and this code state.** A different BLAS, a different HiGHS build, or a change to `A` would move the last digits; nothing here certifies cross-machine reproducibility or the long-term availability of the grid.
* **The paper is a moving target.** F7-F11 were found against the manuscript and tables of 2026-11-01 16:5x; if the tables or the prose are regenerated, the traceability claims must be re-checked (the suite is re-runnable in about three minutes).

## Check theorem — Independent re-derivation of Proposition 1

**Status: PASS**

Prop. 1 confirmed independently for m=0..30: g*(m)=2 sin(pi/(2(2m+1))), boundary from a root-free Schur test agrees to 2.9e-11, the spectral radius crosses 1 exactly there, the time-domain growth exponent changes sign, and the boundary oscillation period is 2(2m+1).

Measured:

- `A_max_unit_circle_residual`: 2.0717e-16
- `B_max_boundary_diff`: 2.91037e-11
- `B_boundaries`: 0=2, 1=1, 2=0.618034, 3=0.445042, 4=0.347296, 5=0.28463, 6=0.241073, 7=0.209057, 8=0.184537, 9=0.165159, 10=0.14946, 11=0.136485, 12=0.125581, 13=0.11629
- `C_max_rho_at_boundary_error`: 2.66454e-15
- `D_growth_table`:
    - m=1, slope_below=-0.025647, slope_above=0.024395, factor_below_over_rho=1, factor_above_over_rho=1
    - m=2, slope_below=-0.014868, slope_above=0.014164, factor_below_over_rho=1, factor_above_over_rho=1
    - m=3, slope_below=-0.01052, slope_above=0.010025, factor_below_over_rho=1, factor_above_over_rho=1
    - m=4, slope_below=-0.00815, slope_above=0.007768, factor_below_over_rho=1, factor_above_over_rho=1
    - m=5, slope_below=-0.006655, slope_above=0.006344, factor_below_over_rho=1, factor_above_over_rho=1
    - m=6, slope_below=-0.005625, slope_above=0.005362, factor_below_over_rho=1, factor_above_over_rho=1
    - m=7, slope_below=-0.004872, slope_above=0.004644, factor_below_over_rho=1, factor_above_over_rho=1
    - m=8, slope_below=-0.004296, slope_above=0.004096, factor_below_over_rho=1, factor_above_over_rho=1
- `E_periods`:
    - m=1, predicted=6, zero_cross=6, fft=5.971, root_arg=6
    - m=2, predicted=10, zero_cross=10, fft=9.942, root_arg=10
    - m=3, predicted=14, zero_cross=14, fft=13.932, root_arg=14
    - m=4, predicted=18, zero_cross=18, fft=17.886, root_arg=18
    - m=5, predicted=22, zero_cross=22, fft=21.904, root_arg=22
    - m=6, predicted=26, zero_cross=26, fft=25.924, root_arg=26
    - m=7, predicted=30, zero_cross=30, fft=29.898, root_arg=30
    - m=8, predicted=34, zero_cross=34, fft=33.851, root_arg=34
    - m=9, predicted=38, zero_cross=38, fft=37.926, root_arg=38
    - m=10, predicted=42, zero_cross=42, fft=41.796, root_arg=42
    - m=11, predicted=46, zero_cross=46, fft=45.511, root_arg=46
    - m=12, predicted=50, zero_cross=50, fft=49.951, root_arg=50
- `wall_s`: 0.56

Notes:

- A: substituting z=e^{i theta} into z^{m+1}-z^m+g=0 gives g = z^m(1-z); on |z|=1, real positive g requires (m+1/2)theta = pi/2 mod 2pi, whose smallest-|g| solution is theta_0 = pi/(2m+1) with g = 2 sin(theta_0/2). Worst numerical residual over m=0..30: 2.07e-16.

What this does not prove. This verifies a linear difference equation. It does not prove that the simulated economy follows that linear loop: the reduction from the model to u_{t+1} = u_t - g u_{t-m} is a local linearisation whose gain g is an equilibrium object that must be measured numerically, and the model's nonlinearities (rationing, short-side programme, inventory) can change both the boundary and the period. Agreement between the simulation and the formula is a hypothesis test, not an identity.

## Check engine — Independent one-period implementation vs the engine

**Status: PASS**

My from-scratch period reproduces the engine on 192 captured periods (regimes, non-exclusive: 101 calm / 18 capacity-binding / 91 tier-binding): worst relative difference 0.00e+00 over x, Z, C, G, I, S_end, unmet, supply. The balance identity, Z = A diag(x), capacity/target/labour bounds and the tier ladder hold; single-pass output is certified optimal by two independent HiGHS algorithms; the 10 collapse periods are structurally forced by outranking claims.

Measured:

- `periods_captured`: 192
- `collapse_blocked_goods`: [4, 19, 19, 20, 20, 20, 20, 20, 3, 17]
- `regime_counts`: calm=101, capacity=18, tier=91
- `max_rel_diff_by_regime`: calm=0, capacity=0, tier=0
- `max_relative_difference`: 0
- `max_relative_difference_at`: 
- `ladder_violations`: 0
- `engine_status_counts`: infeasible-ub=180, optimal=12
- `n_collapse_periods`: 10
- `n_collapse_forced`: 10
- `n_first_pass_optimality_probes`: 182
- `max_cascade_exit_deficit_rel`: 2.34915e-16
- `wall_s`: 1.46

What this does not prove. I implemented the *published* algorithm, so agreement shows that the code does what the specification says; it does not show that the specification is the right model. The per-period comparison also cannot detect a bias that is present identically in both my implementation and the engine's (e.g. a shared but wrong definition of a claim row).

## Check accounting — Accounting audit over the experiment records

**Status: PASS**

2344 audit records across 11 blocks: balance identity worst 1.63e-16 (tol 1e-08), Z=A diag(x) worst 0.00e+00, no negative arrays, C<=demand / x<=capacity / x<=targets all hold, unmet=demand-C holds; recomputed shortage_rate, welfare, plan_fulfilment and four aggregates match runs.parquet to 0.00e+00; the priority ladder was verified exactly on 381 re-run periods (0 violations) and the stored arrays reproduce from the current code to 0.00e+00.

Measured:

- `n_audit_records`: 2344
- `n_blocks`: 11
- `max_balance_identity_residual`: 1.62532e-16
- `max_Z_vs_Ax_residual`: 0
- `max_metric_error`: shortage_rate=0, welfare=0, plan_fulfilment=0, consumption_total=0, demand_total=0, output_total=0, unmet_total=0
- `n_metric_comparisons`: shortage_rate=2344, welfare=2344, plan_fulfilment=2344, consumption_total=2344, demand_total=2344, output_total=2344, unmet_total=2344
- `n_records_with_negative`: 0
- `n_records_C_gt_demand`: 0
- `n_records_x_gt_capacity`: 0
- `n_records_bad_unmet`: 0
- `n_substrate_mismatches`: 0
- `runner_reported_residual_max`: 1.62532e-16
- `runner_residual_vs_mine`: 0
- `n_duplicate_metric_rows`: 0
- `max_spoilage_share_of_supply`: 0.159973
- `n_sectors_with_implied_cap_ratios`: 0
- `ladder_records_checked`: 381
- `ladder_violations`: 0
- `cells_rerun`: 83
- `max_period_reproduction_diff`: 0
- `wall_s`: 26.52

Notes:

- audit dump stores 20-wide arrays for the 6-sector cells and NaN-pads them (24 records in ['B1']); I sliced the valid prefix.
- audit dump writes A for the 6-sector cells FLATTENED into a 20x20 buffer (24 records); I recovered it by reshaping the finite entries, but the dump should be fixed (see report).

What this does not prove. The balance identity is imposed by construction inside the engine (inventory is the residual), so a leak that moves value *between* categories while keeping the total will pass the identity. The audit therefore also recomputes the published metrics from the raw arrays and checks feasibility and the priority ladder, but it cannot prove that the categories themselves are economically the right ones. It also only covers the sampled records: a bug confined to un-sampled periods or un-sampled cells would survive.

## Check reproducibility — Reproducibility audit of the stored results

**Status: PASS**

80 of 7364 (cell, seed) runs re-executed across 11 block(s): 2960 stored run-level metrics match to 0.0e+00; my independent recomputation of mean_shortage_rate, cum_welfare and mean_plan_fulfilment agrees to 2.0e-16; 147120 per-period metric values re-checked (worst 0.0e+00); manifests carry versions, git commit and scenario hashes that match; runs.parquet and summary.csv agree; 16 scenario YAML(s) re-run.

Measured:

- `n_runs_available`: 7364
- `n_runs_sampled`: 80
- `n_samples`: 80
- `n_metric_comparisons_4a`: 2960
- `worst_run_metric_diff_4a`: 0
- `worst_run_metric_diff_at`: 
- `independent_metric_worst`: mean_shortage_rate=9.71445e-17, cum_welfare=2.02723e-16, mean_plan_fulfilment=1.11022e-16, n_periods_used=0
- `per_period_comparisons`: 147120
- `per_period_worst_diff`: 0
- `n_cells_without_definition`: 0
- `n_tech_spec_disagreements`: 0
- `n_manifest_blocks_ok`: 11
- `n_consistency_checks_4e`: 14728
- `n_scenario_yamls_rerun`: 16
- `wall_s`: 14.63

Notes:

- 4d: B1: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B1b: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B2: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B2L: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B3: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B4: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B5: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B6: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B7: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: B8: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4d: P2: seeds recorded only as counts (seeds_per_cell); the actual values live in cells.json/summary.csv
- 4f: baseline_lp_k0.yaml: no stored summary row for scenario id 'baseline_lp_k0'
- 4f: baseline_mb_k0.yaml: no stored summary row for scenario id 'baseline_mb_k0'
- 4f: dual_track_theta0p5_k0.yaml: no stored summary row for scenario id 'dual_track_theta0p5_k0'
- 4f: dual_track_theta0p5_k3.yaml: no stored summary row for scenario id 'dual_track_theta0p5_k3'
- 4f: lag_hybrid_dual_track_k3.yaml: no stored summary row for scenario id 'lag_hybrid_dual_track_k3'
- 4f: lag_lp_central_k3.yaml: no stored summary row for scenario id 'lag_lp_central_k3'
- 4f: lag_market_k3.yaml: no stored summary row for scenario id 'lag_market_k3'
- 4f: lag_material_balance_k3.yaml: no stored summary row for scenario id 'lag_material_balance_k3'
- 4f: lag_rationing_priority_k3.yaml: no stored summary row for scenario id 'lag_rationing_priority_k3'
- 4f: lag_tatonnement_k3.yaml: no stored summary row for scenario id 'lag_tatonnement_k3'
- 4f: robust_synthetic_substrate.yaml: no stored summary row for scenario id 'robust_synthetic_substrate'
- 4f: shock_loss30_rationing_priority.yaml: no stored summary row for scenario id 'shock_loss30_rationing_priority'
- 4f: shock_loss50_tatonnement.yaml: no stored summary row for scenario id 'shock_loss50_tatonnement'
- 4f: shock_propagation_template.yaml: no stored summary row for scenario id 'shock_propagation_template'
- 4f: shock_sector_c01_material_balance.yaml: no stored summary row for scenario id 'shock_sector_c01_material_balance'
- 4f: shock_sector_c19_material_balance.yaml: no stored summary row for scenario id 'shock_sector_c19_material_balance'

What this does not prove. Reproducibility is not validity: it shows that the stored numbers follow from the recorded inputs with the current code, not that the model is a good description of any economy. It also cannot detect a wrong result that is deterministically wrong, and the re-run uses the same underlying libraries, so a numerical bug inside numpy/scipy would reproduce identically.

## Check statistics — Statistical audit (Wilcoxon/bootstrap/Holm/H4)

**Status: PASS**

5a machinery validated independently (Wilcoxon exact vs scipy, Holm vs statsmodels, Spearman vs scipy, permutation vs full enumeration, bootstrap reproducible); 5b 1 paired tests and 37 Holm values recomputed; 5c H4 correlations recomputed with a permutation null; 5d 1018 published table values re-derived from results/ (7 tables, worst 5.0e-07)

Measured:

- `5a_wilcoxon_exact_max_diff`: 0
- `5a_wilcoxon_normal_max_diff`: 1.66533e-16
- `5a_holm_example`: [0.011, 0.08, 0.351, 0.351, 0.351, 0.36, 0.37, 0.82, 0.82, 0.82, 0.82]
- `5a_spearman_max_diff`: 1.11022e-16
- `5a_permutation_vs_exact`: perm=0.0157992, exact=0.0166667
- `5a_bootstrap_ci`: ci=(-0.6462720519005788, 0.10470015457123541), ci_other_seed=(-0.6491652296509687, 0.11330669802630018)
- `5b_n_rows`: 59
- `5b_paired_rows_checked`: 1
- `5b_p_values_checked`: 1
- `5b_worst_p_diff`: 1.36719e-12
- `5b_worst_effect_rel_diff`: 0
- `5b_holm_values_checked`: 37
- `5b_status`: AUDITED
- `5c_n_cells`: 60
- `5c_h4`:
    - mechanism=material_balance, outcome=output_loss, predictor=forward_rowsum_A, rho=0.329323, p_perm=0.154085, n=20
    - mechanism=material_balance, outcome=output_loss, predictor=forward_intermediate_share, rho=0.512782, p_perm=0.024298, n=20
    - mechanism=material_balance, outcome=output_loss, predictor=leontief_colsum, rho=0.019549, p_perm=0.937306, n=20
    - mechanism=material_balance, outcome=output_loss, predictor=leontief_rowsum, rho=0.368421, p_perm=0.108089, n=20
    - mechanism=material_balance, outcome=amplification, predictor=forward_rowsum_A, rho=-0.294737, p_perm=0.20438, n=20
    - mechanism=material_balance, outcome=amplification, predictor=forward_intermediate_share, rho=0.759398, p_perm=0.0001, n=20
    - mechanism=material_balance, outcome=amplification, predictor=leontief_colsum, rho=0.312782, p_perm=0.180382, n=20
    - mechanism=material_balance, outcome=amplification, predictor=leontief_rowsum, rho=-0.377444, p_perm=0.10369, n=20
    - mechanism=rationing_priority, outcome=output_loss, predictor=forward_rowsum_A, rho=0.312782, p_perm=0.176482, n=20
    - mechanism=rationing_priority, outcome=output_loss, predictor=forward_intermediate_share, rho=0.413534, p_perm=0.076792, n=20
    - mechanism=rationing_priority, outcome=output_loss, predictor=leontief_colsum, rho=-0.040602, p_perm=0.866913, n=20
    - mechanism=rationing_priority, outcome=output_loss, predictor=leontief_rowsum, rho=0.365414, p_perm=0.113189, n=20
    - mechanism=rationing_priority, outcome=amplification, predictor=forward_rowsum_A, rho=-0.392481, p_perm=0.087591, n=20
    - mechanism=rationing_priority, outcome=amplification, predictor=forward_intermediate_share, rho=0.691729, p_perm=0.0012, n=20
    - mechanism=rationing_priority, outcome=amplification, predictor=leontief_colsum, rho=0.245113, p_perm=0.29687, n=20
    - mechanism=rationing_priority, outcome=amplification, predictor=leontief_rowsum, rho=-0.46015, p_perm=0.042396, n=20
    - mechanism=tatonnement, outcome=output_loss, predictor=forward_rowsum_A, rho=0.311278, p_perm=0.179782, n=20
    - mechanism=tatonnement, outcome=output_loss, predictor=forward_intermediate_share, rho=0.566917, p_perm=0.011599, n=20
    - mechanism=tatonnement, outcome=output_loss, predictor=leontief_colsum, rho=0.016541, p_perm=0.947705, n=20
    - mechanism=tatonnement, outcome=output_loss, predictor=leontief_rowsum, rho=0.35188, p_perm=0.127387, n=20
- `5c_permutation_vs_paper`:
    - mechanism=material_balance, predictor=forward_intermediate_share, rho=0.512782, p_perm=0.024298, paper_p=0.0207797, paper_p_holm=0.166238, paper_significant=no
    - mechanism=material_balance, predictor=leontief_colsum, rho=0.019549, p_perm=0.937306, paper_p=0.934803, paper_p_holm=1, paper_significant=no
    - mechanism=rationing_priority, predictor=forward_intermediate_share, rho=0.413534, p_perm=0.076792, paper_p=0.0699207, paper_p_holm=0.489445, paper_significant=no
    - mechanism=rationing_priority, predictor=leontief_colsum, rho=-0.040602, p_perm=0.866913, paper_p=0.865047, paper_p_holm=1, paper_significant=no
    - mechanism=tatonnement, predictor=forward_intermediate_share, rho=0.566917, p_perm=0.011599, paper_p=0.00914627, paper_p_holm=0.0823164, paper_significant=no
    - mechanism=tatonnement, predictor=leontief_colsum, rho=0.016541, p_perm=0.947705, paper_p=0.944817, paper_p_holm=1, paper_significant=no
- `5c_status`: AUDITED
- `5d_tables_checked`: 7
- `5d_values_checked`: 1018
- `5d_worst_rel_diff`: 4.98353e-07
- `5d_worst_at`: robustness.csv:mean_x_over_base
- `5d_status`: AUDITED
- `wall_s`: 11.33

Notes:

- 5c: H4 correlations recomputed from summary.csv with a 10,000-draw permutation null; the paper's reported rho should match the forward_rowsum_A or forward_intermediate_share row for the same mechanism.

What this does not prove. I can recompute the test statistics and their multiplicity correction; I cannot verify that the pre-registered test was the right one for the question, that the run-level unit of analysis was respected in every table, or that the pre-registration was not silently narrowed after the results were seen. The permutation test assumes exchangeability of the shocked sectors, which is exact under the null only if the shock assignment is exchangeable — with a fixed 20-sector design that is the natural null but it does not model dependence through common shocks.

## Check claims — Mechanism claims for the headline interaction (Lead's target)

**Status: FAIL**

Measured:

- `6a_grid`: gain0.0_rounds0={'mean': 1.0, 'sd': 0.0}, gain0.0_rounds3={'mean': 1.0, 'sd': 0.0}, gain1.0_rounds0={'mean': 1.0, 'sd': 0.0}, gain1.0_rounds3={'mean': 0.9088942499133882, 'sd': 0.007325500460691469}
- `6a_interaction_size`: 0.0911058
- `6b_regimes`: capacity(=base_output)={'R': 25143704.520000003, 'gain_+10': 0.0, 'loss_-10': 0.0}, goods_balance_frontier={'R': 69604096.66419159, 'gain_+10': 0.0, 'loss_-10': -0.08709849708381157}, mechanism_ub(1.4*base)={'R': 32151896.43165186, 'gain_+10': 0.0, 'loss_-10': 0.0}
- `6b_operating_regime_is_symmetric`: True
- `6b_capacity_regime_is_symmetric`: True
- `6c_isolation`: labor_x3={'mean': 1.0, 'sd': 0.0}, no_inventory_estimate={'mean': 0.9069411767373234, 'sd': 0.004692443583169586}, stock_ratio_0={'mean': 0.9064146660204067, 'sd': 0.007471058628637053}, stock_ratio_0.5={'mean': 0.9106420076198702, 'sd': 0.007114084658695494}, no_capacity_estimate={'mean': 0.9040740178005819, 'sd': 0.008503385704784542}, no_backlog_estimate={'mean': 0.9062722807402683, 'sd': 0.004843373297985909}
- `6c_inventory_estimate_irrelevant`: True
- `6c_labor_drives_loss`: True
- `6d_labour_used_over_supply_mean`: 0.99179
- `6d_labour_binding_fraction`: 0.862132
- `6d_goods_tightness_max`: -0.161008
- `6d_min_sector_fill_mean`: 0.845039
- `6_current_claim_status`: SUPPORTED
- `6_old_inventory_claim_status`: REFUTED
- `6e_hypothesis_rows`: ['H1', "H1'", 'H2', 'H4', 'H5', 'incidence', 'ladder']
- `6e_H3_statistic`: n_cells=20, min=0.0116237, max=0.0603803, mean=0.0293487, threshold=0.005, min_ratio=2.32475, max_ratio=12.0761, cells_below_threshold=0
- `6e_H3_shock_size`: [0.5]
- `6e_status`: AUDITED
- `wall_s`: 9.79

Failures:

- 6e: the hypothesis adjudication table has no row for ['H3'] (present: ['H1', "H1'", 'H2', 'H4', 'H5', 'incidence', 'ladder'])

Notes:

- The interaction is reproduced (1.0000 either alone vs 0.9089 together). The corrected mechanism is supported: inventory estimates are irrelevant in the operating regime (r_est=0 gives 0.9069 vs 0.9089), labour x3 removes the loss (1.0000), the mechanism's own balancing calls are labour-binding (used/L = 0.992 mean, 86% of calls) and the inventory asymmetry exists only at the knife-edge goods-balance frontier (gain +0.00%, loss -8.71%), where the mechanism never operates. The earlier 'inventory-estimate asymmetry' explanation is therefore refuted (finding F4).

What this does not prove. This checks the claims I was asked to check, on one substrate and one calibration. It does not establish that the labour-projection mechanism is the only cause of the interaction (only that it is the binding margin in the tested configuration), nor that the interaction is economically meaningful outside the tested grid of gains, rounds and labour supply factors.
