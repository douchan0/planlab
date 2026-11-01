# PlanLab independent verification report

Generated: 2026-11-01 16:55:55 +0800  
Repository: `/Users/ceontou/Documents/code/planlab`  
git commit: `f3b9b3166452be2242bd4ea324afacf6b5ef1528` (dirty: True)  
Python 3.10.13 on macOS-15.7.7-arm64-arm-64bit  
Wall time: 76.7 s  
Checks: 4 PASS, 2 FAIL, 0 SKIP, 0 ERROR

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
- Resolution: WORKED AROUND in check 3 (recover by reshaping); reported to the Lead and to the experiments layer. Fix recommended before the final grid.

### F6 — bug (provenance) (2026-11-01)

- Where: results/<block>/cells.json
- Finding: The 6-sector B1 cells record scenario.technology = {source: eurostat, year: 2020, n_sectors: 20} while the authoritative cell technology is {toy: 6}. The recorded scenario therefore does not reproduce the cell (a re-run from the scenario dict alone produces 20-sector arrays), and the manifest's scenario_hashes hash the cell dict rather than the scenario.
- Evidence: check 4: re-running the recorded scenario with cell.tech gives bit-identical results, with scenario.technology it does not (0.436 relative difference in x).
- Resolution: REPORTED to the Lead and the experiments layer; check 4 resolves the technology from cell.tech and notes the disagreement.

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
- `wall_s`: 0.76

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
- `wall_s`: 2.21

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
- `wall_s`: 41.12

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
- `wall_s`: 15.82

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

**Status: FAIL**

Measured:

- `5a_wilcoxon_exact_max_diff`: 0
- `5a_wilcoxon_normal_max_diff`: 1.66533e-16
- `5a_holm_example`: [0.011, 0.08, 0.351, 0.351, 0.351, 0.36, 0.37, 0.82, 0.82, 0.82, 0.82]
- `5a_spearman_max_diff`: 1.11022e-16
- `5a_permutation_vs_exact`: perm=0.0157992, exact=0.0166667
- `5a_bootstrap_ci`: ci=(-0.6462720519005788, 0.10470015457123541), ci_other_seed=(-0.6491652296509687, 0.11330669802630018)
- `5b_n_rows`: 59
- `5b_paired_rows_checked`: 0
- `5b_p_values_checked`: 0
- `5b_worst_p_diff`: 0
- `5b_worst_effect_rel_diff`: 0
- `5b_holm_values_checked`: 37
- `5b_status`: AUDITED
- `5c_n_cells`: 63
- `5c_h4`:
    - mechanism=hybrid_dual_track, predictor=forward_rowsum_A, rho=nan, p_perm=nan, n=1
    - mechanism=hybrid_dual_track, predictor=forward_intermediate_share, rho=nan, p_perm=nan, n=1
    - mechanism=hybrid_dual_track, predictor=leontief_colsum, rho=nan, p_perm=nan, n=1
    - mechanism=hybrid_dual_track, predictor=leontief_rowsum, rho=nan, p_perm=nan, n=1
    - mechanism=lp_central, predictor=forward_rowsum_A, rho=nan, p_perm=nan, n=1
    - mechanism=lp_central, predictor=forward_intermediate_share, rho=nan, p_perm=nan, n=1
    - mechanism=lp_central, predictor=leontief_colsum, rho=nan, p_perm=nan, n=1
    - mechanism=lp_central, predictor=leontief_rowsum, rho=nan, p_perm=nan, n=1
    - mechanism=market, predictor=forward_rowsum_A, rho=nan, p_perm=nan, n=1
    - mechanism=market, predictor=forward_intermediate_share, rho=nan, p_perm=nan, n=1
    - mechanism=market, predictor=leontief_colsum, rho=nan, p_perm=nan, n=1
    - mechanism=market, predictor=leontief_rowsum, rho=nan, p_perm=nan, n=1
    - mechanism=material_balance, predictor=forward_rowsum_A, rho=-0.282707, p_perm=0.224078, n=20
    - mechanism=material_balance, predictor=forward_intermediate_share, rho=0.754887, p_perm=0.0001, n=20
    - mechanism=material_balance, predictor=leontief_colsum, rho=0.311278, p_perm=0.184682, n=20
    - mechanism=material_balance, predictor=leontief_rowsum, rho=-0.365414, p_perm=0.114689, n=20
    - mechanism=rationing_priority, predictor=forward_rowsum_A, rho=-0.390977, p_perm=0.088591, n=20
    - mechanism=rationing_priority, predictor=forward_intermediate_share, rho=0.688722, p_perm=0.0011, n=20
    - mechanism=rationing_priority, predictor=leontief_colsum, rho=0.239098, p_perm=0.309269, n=20
    - mechanism=rationing_priority, predictor=leontief_rowsum, rho=-0.458647, p_perm=0.043096, n=20
- `5c_status`: AUDITED
- `wall_s`: 6.89

Failures:

- 5b: block B3: Holm-adjusted p disagrees by 2.00e-06
- 5c: hybrid_dual_track/forward_rowsum_A: undefined Spearman rho
- 5c: hybrid_dual_track/forward_intermediate_share: undefined Spearman rho
- 5c: hybrid_dual_track/leontief_colsum: undefined Spearman rho
- 5c: hybrid_dual_track/leontief_rowsum: undefined Spearman rho
- 5c: lp_central/forward_rowsum_A: undefined Spearman rho
- 5c: lp_central/forward_intermediate_share: undefined Spearman rho
- 5c: lp_central/leontief_colsum: undefined Spearman rho
- 5c: lp_central/leontief_rowsum: undefined Spearman rho
- 5c: market/forward_rowsum_A: undefined Spearman rho
- 5c: market/forward_intermediate_share: undefined Spearman rho
- 5c: market/leontief_colsum: undefined Spearman rho
- 5c: market/leontief_rowsum: undefined Spearman rho

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
- `6_explanation_status`: REFUTED
- `wall_s`: 9.86

Failures:

- 6: the stated explanation (inventory-estimate asymmetry) does not hold in the regime that produces the loss; the interaction is labour-driven (see 6b/6c/6d numbers)

Notes:

- The interaction is reproduced, and the inventory asymmetry the Lead described is real at the goods-balance frontier (0 gain / -8.7% loss), but in the mechanism's operating regime (+-10% inventory changes R by 0.0e+00, R(r_est=0) changes nothing, labour x3 makes the loss vanish, labour used/L = 0.992, goods tightness max -0.161). The loss is driven by the one-sided projection onto the LABOUR-constrained feasible set, not by the inventory estimate. This is a claim-level failure, not a code bug.

What this does not prove. This checks the claims I was asked to check, on one substrate and one calibration. It does not establish that the labour-projection mechanism is the only cause of the interaction (only that it is the binding margin in the tested configuration), nor that the interaction is economically meaningful outside the tested grid of gains, rounds and labour supply factors.
