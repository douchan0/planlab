"""Run every independent verification check and write verification/report.md.

Exit codes
----------
0   every check PASSED
1   at least one check FAILED (a finding that must not be silently shipped)
2   no failures, but at least one check was SKIPPED because its input
    (results/, results/tables/tests.csv, ...) did not exist yet: the audit is
    INCOMPLETE and this must not be mistaken for a pass.

Usage
-----
    cd planlab && source scripts/env.sh && .venv/bin/python verification/run_all.py
    ... verification/run_all.py --only theorem,engine      # subset
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
import traceback

from .common import RESULTS, ROOT, CheckResult, rel

CHECK_MODULES = (
    ("theorem", "check1_theorem", "Independent re-derivation of Proposition 1"),
    ("engine", "check2_engine", "Independent one-period implementation vs the engine"),
    ("accounting", "check3_accounting", "Accounting audit over the experiment records"),
    ("reproducibility", "check4_repro", "Reproducibility audit of the stored results"),
    ("statistics", "check5_stats", "Statistical audit (Wilcoxon/bootstrap/Holm/H4)"),
    ("claims", "check6_claims", "Mechanism claims for the headline interaction"),
)

#: Findings this verifier reported to the Lead, with the date they were reported.
#: Kept here (not in a mutable log) so that the record survives re-runs.
FINDINGS = [
    {
        "id": "F1",
        "date": "2026-11-01",
        "severity": "error (documentation)",
        "where": "docs/notes_delay_theorem.md (lines 50-51 and the table at 163-170)",
        "finding": "The numeric list and the table shifted the stability boundary by one lag "
                   "index against the note's own formula: it gave g*(1)=2, g*(2)=1, g*(3)=0.618, "
                   "g*(4)=0.445, whereas g*(m)=2 sin(pi/(2(2m+1))) gives g*(1)=1, g*(2)=0.618034, "
                   "g*(3)=0.445042 and 2 is g*(0).",
        "evidence": "check 1 (three independent routes: Schur-Cohn bisection, companion-matrix "
                    "spectral radius, time-domain growth exponent), m=0..30.",
        "resolution": "FIXED by the Lead the same day; the note now agrees with the paper and "
                      "records that the correction came from this check.",
    },
    {
        "id": "F2",
        "date": "2026-11-01",
        "severity": "error (documentation)",
        "where": "docs/notes_delay_theorem.md line 105",
        "finding": "\"Every period of delay more than halves the admissible gain\" is false: "
                   "2 -> 1 halves only at the first step; 1 -> 0.618 is a 38% cut and the ratios "
                   "approach 1 (g*(m) ~ pi/(2m+1)). The bounded object is m*g*(m) <= pi/2.",
        "evidence": "check 1 numbers for m=0..30; the paper already stated the correct version.",
        "resolution": "FIXED by the Lead the same day.",
    },
    {
        "id": "F3",
        "date": "2026-11-01",
        "severity": "gap in proof (not an error)",
        "where": "docs/notes_delay_theorem.md, proof of Proposition 1 (lines 84-90 of the original)",
        "finding": "The proof asserted that j=0 gives the smallest gain without ruling out j>=1, "
                   "for which theta_j > pi and sin(theta/2) decreases. The statement is true "
                   "(g_j = 2 sin(pi(1+4j)/(2(2m+1))) >= g_0 for 0<=j<=m, with equality also at "
                   "j=m, the conjugate pair), but the note did not show it.",
        "evidence": "check 1: the unit-circle identity and the root-free boundary agree with the "
                    "closed form for m=0..30; the paper's Section 4 already contains the correct "
                    "argument.",
        "resolution": "CLOSED by the Lead with the paper's x_m in [x_0, pi-x_0] argument.",
    },
    {
        "id": "F4",
        "date": "2026-11-01",
        "severity": "error (scientific explanation)",
        "where": "Lead's proposed explanation of the material_balance gain x rounds interaction",
        "finding": "The interaction is real (gain 0 / rounds 3 -> 1.0000; gain 1 / rounds 0 -> "
                   "1.0000; gain 1 / rounds 3 -> 0.9089 +- 0.0073), but the stated cause — "
                   "feasible_output being asymmetric in the *inventory* estimate — is refuted in "
                   "the regime that produces the loss: R(0.9 r) = R(r) = R(1.1 r) to 0.0e+00 in "
                   "every one of the mechanism's balancing calls (max goods-balance tightness "
                   "0.25; labour used/L = 0.99-1.00), removing the inventory estimate entirely "
                   "(r_est = 0) does not change the loss (0.9083 vs 0.9087), stock_ratio 0 and "
                   "0.5 do not change it (0.9061 / 0.9110), while labour supply x3 removes it "
                   "completely (1.0000). The asymmetry is real only at the knife-edge "
                   "goods-balance frontier (0 gain / -8.7% loss), where the mechanism never "
                   "operates. The correct mechanism is the one-sided projection onto the "
                   "LABOUR-constrained feasible set, and the result is conditional on "
                   "labor_supply_factor = 1.15.",
        "evidence": "check 6 (6a-6d); replicated independently by the Lead (factor 1.15 -> 0.8698, "
                    "1.40 -> 0.9922, 2.0 -> 1.0000).",
        "resolution": "OPEN at the time of writing: the paper's mechanism section was being "
                      "rewritten to the labour-projection version and labor_supply_factor was "
                      "added to B6 as a treatment. Re-run check 6 after the rewrite.",
    },
    {
        "id": "F5",
        "date": "2026-11-01",
        "severity": "bug (results artefacts, not the model)",
        "where": "results/<block>/audit_records.npz",
        "finding": "For the 6-sector cells the runner writes `A` FLATTENED into a fixed 20x20 "
                   "buffer (the 36 finite entries are the row-major 6x6 matrix, the other 364 "
                   "are NaN), while x/Z/... are NaN-padded but correctly placed and `sectors` is "
                   "empty. Any naive reader gets a NaN-filled A for toy cells.",
        "evidence": "check 3 loader: A.shape == (20,20) with 36 finite entries for a record whose "
                    "n_sectors == 6; reshaping the finite entries reproduces toy_technology(6).A "
                    "exactly (max |diff| = 0.0).",
        "resolution": "FIXED before the final grid: the final audit files store A in the "
                      "[:n, :n] block (the remaining padding is NaN but the valid prefix is "
                      "correct); check 3 now reads it directly. Recorded here because the "
                      "earlier dumps were unusable for toy cells.",
    },
    {
        "id": "F6",
        "date": "2026-11-01",
        "severity": "bug (provenance)",
        "where": "results/<block>/cells.json",
        "finding": "The 6-sector B1 cells record scenario.technology = {source: eurostat, year: "
                   "2020, n_sectors: 20} while the authoritative cell technology is {toy: 6}. The "
                   "recorded scenario therefore does not reproduce the cell (a re-run from the "
                   "scenario dict alone produces 20-sector arrays), and the manifest's "
                   "scenario_hashes hash the cell dict rather than the scenario.",
        "evidence": "check 4: re-running the recorded scenario with cell.tech gives bit-identical "
                    "results, with scenario.technology it does not (0.436 relative difference in "
                    "x).",
        "resolution": "FIXED before the final grid: cells.json now records the cell technology "
                      "({toy: 6}) instead of the eurostat placeholder; re-running from the "
                      "recorded scenario reproduces the cell.",
    },
    {
        "id": "F7",
        "date": "2026-11-01",
        "severity": "error (paper, results interpretation)",
        "where": "paper/main.md §6.6 against results/B4/cells.json and "
                 "paper/tables/rationing_shadow.csv",
        "finding": "The B4 cells are named loss50/loss70/loss90 and pass scenario shock "
                   "`size` = 0.5/0.3/0.1; because Shock.factor treats `size` as the capacity "
                   "MULTIPLIER, the realised capacity losses are 50%/70%/90%, not the protocol's "
                   "10%/30%/50% losses. The paper labels the 70% cell 'thirty per cent' and the "
                   "90% cell 'ten per cent', and the narrative built on those labels does not "
                   "survive: the output gap reverses sign between the 50% and 70% cells "
                   "(-0.9pp, rationing ahead, then +3.35pp, tatonnement ahead) and widens to "
                   "+4.97pp at 90%. The protocol (docs/PROTOCOL.md §4 B4) and Amendment 2 "
                   "('keeps its pre-registered size sweep {0.10, 0.30, 0.50}') are silent on "
                   "the multiplier-vs-loss ambiguity.",
        "evidence": "check 5's data audit and the B4 cell definitions; numbers reproduced from "
                    "paper/tables/rationing_shadow.csv.",
        "resolution": "FIXED in the current manuscript (paper/main.md mtime in the header): §6.6 now states that the pre-registered sizes are capacity "
                      "multipliers 0.1/0.3/0.5, that the realised losses are 90/70/50 per "
                      "cent, and the narrative follows the rows (ranking reverses at 70 per "
                      "cent, widens at 90 per cent). Verified by grep on the current file.",
    },
    {
        "id": "F8",
        "date": "2026-11-01",
        "severity": "overclaim (paper, H4)",
        "where": "paper/main.md §6.5 'Both halves are confirmed' against results/tables/tests.csv",
        "finding": "The H4 centrality half rests on raw Spearman p-values. The project's own "
                   "test log applies the pre-registered Holm correction and marks EVERY "
                   "centrality test non-significant: material_balance rho=0.5128 p=0.0208 "
                   "p_holm=0.166; tatonnement rho=0.5669 p=0.0091 p_holm=0.082; priority "
                   "rationing rho=0.4135 p=0.0699 p_holm=0.489. My 10,000-draw permutation "
                   "test agrees with the raw p (0.024 / 0.012 / 0.077) and with the rho values "
                   "to 1e-6, so the numbers are right and the INTERPRETATION is not: after the "
                   "paper's own multiplicity rule, the centrality result is not significant for "
                   "any of the three mechanisms. The only significant centrality rows in "
                   "tests.csv use shock_output_amplification (p_holm 0.0011/0.0073/0.00024), "
                   "which the paper itself demotes as a small-denominator artefact. A second, "
                   "independent sensitivity: with the textbook forward-linkage measure (row "
                   "sums of A) the correlation is only 0.31-0.33 (p about 0.15-0.18).",
        "evidence": "check 5 (5c) recomputation from summary.csv + permutation null; "
                    "results/tables/tests.csv H4-centrality rows.",
        "resolution": "FIXED in the current manuscript (§6.5 now reports both linkage definitions, "
                      "the row-sum null and the Holm caveat). One residual: the paper says "
                      "two correlations survive Holm across the three institutions, while "
                      "results/tables/tests.csv applies Holm within the whole B3 block "
                      "(p_holm 0.166 / 0.082 / 0.489, none survive). See F16.",
    },
    {
        "id": "F9",
        "date": "2026-11-01",
        "severity": "factual error (paper)",
        "where": "paper/main.md §6.5",
        "finding": "\"The most damaging sectors are c05, c20, c02 and c12 under all three "
                   "institutions\" is false for priority rationing: its top four are c20, c19, "
                   "c05, c02 (c12 is fifth; c19 is second). The sentence holds for material "
                   "balance and tatonnement.",
        "evidence": "paper/tables/propagation.csv ranks by output_loss.",
        "resolution": "FIXED in the current manuscript (§6.5 now gives the per-institution ordering "
                      "with c19 second and c12 fifth under priority rationing).",
    },
    {
        "id": "F10",
        "date": "2026-11-01",
        "severity": "stale text (paper)",
        "where": "paper/main.md §6.7",
        "finding": "The paragraph warns that the theta column of the dual-track file 'repeats a "
                   "single value' and says the table is read by row order. The regenerated "
                   "results/tables/dual_track.csv has all 11 theta values (0.0-1.0) and 22 "
                   "rows, so the caveat now describes a defect that no longer exists. The "
                   "quoted welfare ranges (252.21-252.34 at lag 0; 251.64-251.74 at lag 3) do "
                   "match the regenerated table.",
        "evidence": "results/tables/dual_track.csv, check 5's data audit.",
        "resolution": "REPORTED; remove the caveat or regenerate the table from the final "
                      "results.",
    },
    {
        "id": "F11",
        "date": "2026-11-01",
        "severity": "claim not matching the table (paper)",
        "where": "paper/main.md §6.4, abstract; §6.1",
        "finding": "(a) §6.4: 'the detrended amplitude of the mean log price sits between 0.017 "
                   "and 0.041 across the whole grid, the lag-one autocorrelation between 0.93 "
                   "and 0.95'. At the reference gain 0.3 the amplitude spans 0.000-0.170 "
                   "(market 0.124-0.170, tatonnement 0.128-0.146) and the acf spans "
                   "0.65-0.95; the 0.017-0.041 range corresponds to tatonnement at gains "
                   "0.05-0.1, not to the lag grid at the reference gain. (b) §6.1 calls "
                   "259.568805 'cumulative welfare'; it is the mean per-period welfare "
                   "(cum_welfare = 12978.44). (c) The abstract says the loss 'vanishes at "
                   "eighty' per cent labour slack, while §6.3 and §6.8 report it vanishing at "
                   "labour_supply_factor 2.00 (100% slack), with 0.4% left at 1.40. (d) The "
                   "abstract's 'a factor of two to fifty below it' mixes the most-responsive "
                   "sector's 1.9 with the median sector's 12-50.",
        "evidence": "results/tables/main_lag_grid.csv, results/B1/summary.csv, "
                    "paper/tables/labour_slack.csv.",
        "resolution": "MOSTLY FIXED in the current manuscript: the phrase cumulative welfare is now welfare, the abstract says the loss vanishes at a hundred per cent of labour slack, and the factor-of-two-to-fifty claim is replaced by the reachability statement. Two numbers remain off in the amplitude sentence of section 6.4: tatonnement spans 0.118-0.152 in the reference-gain table, not 0.128-0.156, and 0.249 is the pooled-grid maximum at gain 0.8, not a reference-gain cell (market 0.124-0.170 is correct).",
    },
    {
        "id": "F12",
        "date": "2026-11-01",
        "severity": "governance / documentation",
        "where": "docs/ARCHITECTURE.md §2, docs/AMENDMENTS.md, the engine",
        "finding": "The engine gained a bounded-storage SPOILAGE channel (Scenario."
                   "max_stock_ratio, default 2.0; S_end = min(residual, max(stock_cap - inv, "
                   "0)); spoilage = residual - S_end) which changes the balance identity to "
                   "S0 + x = Z 1 + C + G + I + S_end + spoilage. It is in the results and the "
                   "paper's tables but not in docs/ARCHITECTURE.md (whose 'frozen' identity "
                   "omits it) nor in AMENDMENTS/DEVIATIONS, contrary to the project's own "
                   "'no silent amendments' rule. My audit finds up to 16% of supply destroyed "
                   "in a single record (1.2-3.3% in the reference cells), so an external "
                   "replicator implementing the documented identity would see a 1-16% "
                   "discrepancy and call it a leak.",
        "evidence": "check 3 (identity holds to 1.6e-16 WITH spoilage, and fails by the "
                    "spoilage amount without it); docs greps.",
        "resolution": "REPORTED; add an amendment entry and update ARCHITECTURE §2.",
    },
    {
        "id": "F13",
        "date": "2026-11-01",
        "severity": "table defect (paper)",
        "where": "paper/tables/main_lag_grid.csv",
        "finding": "The table has 204 rows but only 144 unique (mechanism, gain, reporting-lag) "
                   "keys: the tatonnement cells at price elasticities 0.4 and 0.8 are "
                   "indistinguishable because the elasticity column is absent. The values do "
                   "come from results/B2 (my sorted-value traceability comparison passes), but "
                   "a reader cannot tell which row is which, and §6.4's 'at the reference "
                   "gain' sentences could refer to either.",
        "evidence": "check 5d (all headline tables re-derived from results/, worst 5e-7).",
        "resolution": "FIXED in the current table and verified in check 5d: the table now "
                      "carries a price_elasticity column, so every row is identified.",
    },
    {
        "id": "F14",
        "date": "2026-11-01",
        "severity": "error (paper, H1 narrative)",
        "where": "paper/main.md §6.4 and the abstract, against results/tables/t12_bp_by_sector.csv "
                 "and results/tables/t05_stability_boundary.csv",
        "finding": "§6.4 states b_p 'is of order 0.01 to 0.08 by sector' and that the loop gain "
                   "'is of order 0.02 at the median sector and reaches 0.13 only in the most "
                   "responsive one'. The final measurement table says the opposite: at the "
                   "reference elasticity 0.4 the median b_p is 0.1707, the mean 0.1973 and the "
                   "maximum 0.4782 (at 0.8: median 0.3319, max 0.7867). At eta=1.6 the median "
                   "sector's loop gain is therefore 0.273, ABOVE g*(7)=0.209 and g*(6)=0.241, "
                   "and the most responsive sector reaches 0.765, above g*(m) for every m>=3. "
                   "t05 records g_max_over_bound = 1.11 at m=5 and 1.51 at m=7 (elasticity 0.4), "
                   "1.23-2.62 at 0.8, `boundary_approachable` True in 6 cells and "
                   "`rule_fired_below_bound` True in 2. Hence 'nowhere near the boundary' is "
                   "false for the final grid, 'the median gain of 0.02 would turn at about "
                   "seventy-eight quarters' should be about 31 quarters, and the abstract's "
                   "'a factor of two to fifty below it' is not supported (3.7x below g*(1) but "
                   "1.3x ABOVE g*(7) at the largest gain). The stale numbers look like b_p "
                   "measured before the price-elasticity change in docs/DEVIATIONS.md.",
        "evidence": "check 5d (t12 and t05 re-derived from results/), and the H1 test rows in "
                    "results/tables/tests.csv.",
        "resolution": "FIXED in the current manuscript: §6.4 now reports b_p = 0.197/0.342, maximum "
                      "loop gains 0.316/0.548, the m>=5 / m>=3 crossing lags, and the "
                      "abstract says the bound is 'rarely reached ... crossed only at the far "
                      "end of the lag axis'. These match t12/t05 exactly.",
    },
    {
        "id": "F15",
        "date": "2026-11-01",
        "severity": "understated result (paper)",
        "where": "paper/main.md §6.4 against results/tables/tests.csv row H1",
        "finding": "The paper reports H1 as a null. The project's own test log records the "
                   "pre-registered amplitude test at gain=1.6 (the largest), lag 6 vs lag 0: "
                   "paired one-sided Wilcoxon statistic 817, p = 4.5e-12, significant_05 = yes. "
                   "The null holds at the reference gain 0.3 only. Reporting the rejection that "
                   "appears exactly when the loop gain is pushed toward the boundary would "
                   "support rather than weaken the paper's reading of delay.",
        "evidence": "results/tables/tests.csv (H1 rows).",
        "resolution": "FIXED in the current manuscript (§6.4 now reports the Wilcoxon 817 / "
                      "p = 4.5e-12 row beside the reference-gain null, framed as a "
                      "reachable boundary seen from below).",
    },
    {
        "id": "F16",
        "date": "2026-11-01",
        "severity": "inconsistency (paper vs test log)",
        "where": "paper/main.md §6.5 vs results/tables/tests.csv",
        "finding": "§6.5 says 'After Holm correction across the three institutions, two survive "
                   "at alpha=0.05 (material balance and tatonnement)'. That is true for a "
                   "three-test family (p = 0.0091, 0.0208, 0.0699 -> adjusted 0.027, 0.042, "
                   "0.070), but results/tables/tests.csv applies Holm within the whole B3 block "
                   "(nine centrality rows plus the other B3 tests) and records p_holm = 0.166, "
                   "0.082 and 0.489 - none survives. The protocol says 'Holm correction within "
                   "each block', so the paper's narrower family needs to be stated explicitly "
                   "(and the test log regenerated for that family, or vice versa).",
        "evidence": "check 5b (Holm values recomputed from tests.csv, 37 values, agreement "
                    "1e-12) and the permutation p-values in check 5c.",
        "resolution": "REPORTED; state the family and reconcile the two files.",
    },
    {
        "id": "F17",
        "date": "2026-11-01",
        "severity": "internal contradiction (abstract vs body)",
        "where": "paper/main.md abstract (line 10) vs §6.4 and §6.7",
        "finding": "The abstract still ends 'Two pre-registered hypotheses are falsified: no "
                   "crossing of the delay boundary is observed, and the dual-track share shows "
                   "no detectable interior optimum', while the rebuilt body says the opposite in "
                   "both places: §6.4 states 'The boundary is not unreachable; it is reached by "
                   "driving the loop hard', and §6.7 states 'the pre-registered prediction is "
                   "formally satisfied and one of the two H5 tests survives Holm correction at "
                   "p = 0.0045'. The abstract also contradicts itself two sentences earlier "
                   "('crossed only at the far end of the lag axis by a hard-driven loop').",
        "evidence": "grep of paper/main.md at the audited commit/mtime; the corresponding test "
                    "rows are in results/tables/tests.csv (H1 amplitude row significant; H5 row "
                    "p_holm 0.0089).",
        "resolution": "REPORTED; the abstract must be rebuilt from the final §6.4/§6.7 on "
                      "whether H1 and H5 are falsified, or the body reverted.",
    },
    {
        "id": "F18",
        "date": "2026-11-01",
        "severity": "gap (adjudication table) — the Lead's own finding, confirmed",
        "where": "results/tables/t00_hypotheses.csv vs paper/main.md §5/§6.5",
        "finding": "The machine-readable hypothesis table has rows for H1, H1', H2, H4, H5, "
                   "incidence and ladder but NO row for H3, although §5 states H3 as a "
                   "falsifiable prediction and §6.5 adjudicates it. The adjudication itself is "
                   "correct: I re-derived H3's statistic from results/B3/summary.csv "
                   "(material_balance, 20 shocked sectors): mean shortage 1.162%-6.038%, mean "
                   "2.935%, ratios 2.32-12.08 against the 0.5% falsifier, 0 of 20 cells below "
                   "the threshold - exactly the paper's 1.16-6.04/2.93/2.3-12.1. The amended "
                   "shock (size 0.50 = 50% cut, duration 8) is documented in Amendment 2 and "
                   "disclosed in §5.",
        "evidence": "check 6e (new): hypothesis-table completeness + H3 statistic re-derived "
                    "from B3 + amendment disclosure. The check FAILS until the H3 row exists.",
        "resolution": "OPEN at the time of writing: add the H3 row to the adjudication table "
                      "(and regenerate results/tables/t00_hypotheses.csv).",
    },
]

#: per-check "what this does not prove" text
LIMITS = {
    "theorem": (
        "This verifies a linear difference equation. It does not prove that the simulated "
        "economy follows that linear loop: the reduction from the model to u_{t+1} = u_t - g "
        "u_{t-m} is a local linearisation whose gain g is an equilibrium object that must be "
        "measured numerically, and the model's nonlinearities (rationing, short-side programme, "
        "inventory) can change both the boundary and the period. Agreement between the "
        "simulation and the formula is a hypothesis test, not an identity."
    ),
    "engine": (
        "I implemented the *published* algorithm, so agreement shows that the code does what the "
        "specification says; it does not show that the specification is the right model. The "
        "per-period comparison also cannot detect a bias that is present identically in both "
        "my implementation and the engine's (e.g. a shared but wrong definition of a claim row)."
    ),
    "accounting": (
        "The balance identity is imposed by construction inside the engine (inventory is the "
        "residual), so a leak that moves value *between* categories while keeping the total will "
        "pass the identity. The audit therefore also recomputes the published metrics from the "
        "raw arrays and checks feasibility and the priority ladder, but it cannot prove that the "
        "categories themselves are economically the right ones. It also only covers the sampled "
        "records: a bug confined to un-sampled periods or un-sampled cells would survive."
    ),
    "reproducibility": (
        "Reproducibility is not validity: it shows that the stored numbers follow from the "
        "recorded inputs with the current code, not that the model is a good description of any "
        "economy. It also cannot detect a wrong result that is deterministically wrong, and the "
        "re-run uses the same underlying libraries, so a numerical bug inside numpy/scipy would "
        "reproduce identically."
    ),
    "statistics": (
        "I can recompute the test statistics and their multiplicity correction; I cannot verify "
        "that the pre-registered test was the right one for the question, that the run-level "
        "unit of analysis was respected in every table, or that the pre-registration was not "
        "silently narrowed after the results were seen. The permutation test assumes "
        "exchangeability of the shocked sectors, which is exact under the null only if the "
        "shock assignment is exchangeable — with a fixed 20-sector design that is the natural "
        "null but it does not model dependence through common shocks."
    ),
    "claims": (
        "This checks the claims I was asked to check, on one substrate and one calibration. It "
        "does not establish that the labour-projection mechanism is the only cause of the "
        "interaction (only that it is the binding margin in the tested configuration), nor that "
        "the interaction is economically meaningful outside the tested grid of gains, rounds and "
        "labour supply factors."
    ),
}


def _stamp(path) -> str:
    """mtime of an audited artefact, so a stale read can be detected next time."""
    try:
        import datetime as _d
        ts = path.stat().st_mtime
        return _d.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:                                      # pragma: no cover
        return "missing"


def _git() -> dict:
    out = {"commit": "", "dirty": None, "diff_stat": ""}
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        out["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip())
        out["diff_stat"] = subprocess.run(
            ["git", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()[-500:]
    except Exception:                                      # pragma: no cover
        pass
    return out


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _section(res: CheckResult) -> str:
    lines = [f"## Check {res.key} — {res.title}", ""]
    lines.append(f"**Status: {res.status}**")
    lines.append("")
    if res.summary:
        lines.append(res.summary)
        lines.append("")
    if res.numbers:
        lines.append("Measured:")
        lines.append("")
        for key, value in res.numbers.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                flat = ", ".join(f"{k}={_fmt(v)}" for k, v in list(value.items())[:14])
                lines.append(f"- `{key}`: {flat}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"- `{key}`:")
                for entry in value[:20]:
                    cells = ", ".join(f"{k}={_fmt(v)}" for k, v in entry.items())
                    lines.append(f"    - {cells}")
            else:
                lines.append(f"- `{key}`: {_fmt(value)}")
        lines.append("")
    if res.failures:
        lines.append("Failures:")
        lines.append("")
        for item in res.failures:
            lines.append(f"- {item}")
        lines.append("")
    if res.notes:
        lines.append("Notes:")
        lines.append("")
        for item in res.notes:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("What this does not prove. " + LIMITS.get(res.key, ""))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated check keys")
    parser.add_argument("--report", default=str(ROOT / "verification" / "report.md"))
    parser.add_argument("--json", default=str(ROOT / "verification" / "report.json"))
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    started = time.time()
    results: list[CheckResult] = []
    for key, module_name, title in CHECK_MODULES:
        if only and key not in only:
            continue
        print(f"== running {key} ...", flush=True)
        t0 = time.time()
        try:
            module = __import__(f"verification.{module_name}", fromlist=["run"])
            result = module.run()
        except Exception:                                  # pragma: no cover
            result = CheckResult(key=key, title=title, status="ERROR")
            result.fail(traceback.format_exc()[-4000:])
        result.numbers.setdefault("wall_s", round(time.time() - t0, 2))
        print(f"   {result.status:5s} in {time.time() - t0:.1f}s — {result.summary[:120]}",
              flush=True)
        results.append(result)

    elapsed = time.time() - started
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)
    n_err = sum(r.status == "ERROR" for r in results)
    n_pass = sum(r.status == "PASS" for r in results)

    git = _git()
    stamp = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    header = [
        "# PlanLab independent verification report",
        "",
        f"Generated: {stamp}  ",
        f"Repository: `{ROOT}`  ",
        f"git commit: `{git['commit'] or 'n/a'}` (dirty: {git['dirty']})  ",
        f"Python {platform.python_version()} on {platform.platform()}  ",
        f"Artefacts audited (mtime): paper/main.md {_stamp(ROOT / 'paper' / 'main.md')}; "
        f"paper/tables {_stamp(ROOT / 'paper' / 'tables')}; "
        f"results/tables/tests.csv {_stamp(ROOT / 'results' / 'tables' / 'tests.csv')}; "
        f"results/tables/t00_hypotheses.csv "
        f"{_stamp(ROOT / 'results' / 'tables' / 't00_hypotheses.csv')}  ",
        f"Wall time: {elapsed:.1f} s  ",
        f"Checks: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_err} ERROR",
        "",
        "This suite is written by an independent verifier. It re-derives the theorem, "
        "re-implements a period from the written specification, audits the raw records behind "
        "the paper's numbers, re-runs sampled cells, and re-computes the statistics with its "
        "own code. It deliberately does not import the project's engine, metrics or aggregate "
        "modules to judge their output; where it does use project code it is only for *inputs* "
        "(the technology matrix) or for *records* (the stored arrays).",
        "",
    ]
    if n_skip or n_err:
        header.append(
            "> **The audit is INCOMPLETE.** Skipped or errored checks mean the "
            "corresponding evidence has NOT been verified, not that it passed."
        )
        header.append("")

    findings = [
        "## Findings reported to the Lead (the verifier's own record)",
        "",
        "An independent verifier is only worth having if its record shows what it caught. "
        "These are the defects found in the Lead's work, in the order they were reported.",
        "",
    ]
    for f in FINDINGS:
        findings.append(f"### {f['id']} — {f['severity']} ({f['date']})")
        findings.append("")
        findings.append(f"- Where: {f['where']}")
        findings.append(f"- Finding: {f['finding']}")
        findings.append(f"- Evidence: {f['evidence']}")
        findings.append(f"- Resolution: {f['resolution']}")
        findings.append("")

    remaining = [
        "## What remains unverified",
        "",
        "An honest verification report has to say where it stops. These are the gaps that a",
        "reader should treat as open, not as verified by silence.",
        "",
        "* **The paper's prose is checked against `paper/tables/*.csv`, and the tables against "
        "`results/`, but not every table was re-derived from the raw runs.** Check 4 re-executes "
        "80 of 7\u202f364 runs and reproduces every stored run-level metric bit-for-bit; check 3 "
        "audits 2\u202f344 raw period records. A table built from a run outside those samples "
        "would be caught only if its inputs were among the 80.",
        "* **The H4 primitive is audited at the summary level, not from the sector panel.** I "
        "recompute the amplification statistic from `summary.csv` and replicate the paper's "
        "rho to 1e-6, but I did not rebuild `shock_output_loss` from `sector_panel.parquet`; a "
        "bug in the per-period aggregation would be invisible to me.",
        "* **The measured response coefficients b (price and target) are taken from "
        "`results/tables/t12_bp_by_sector.csv`, not independently re-estimated by me.** The "
        "theory-to-simulation mapping g = eta*b therefore inherits whatever the experiments "
        "layer's probe procedure does; I verified the probe contract reproduces the stored "
        "cells (check 4), not that the probe measures what it claims to measure.",
        "* **The dynamic stability boundary of the full simulation is not independently "
        "located.** Check 1 verifies Proposition 1 for the linear recursion; whether the "
        "simulation follows it is the paper's H1 question, and I audit the stored test "
        "statistics rather than re-running the boundary search.",
        "* **Reproducibility is tested on this machine and this code state.** A different "
        "BLAS, a different HiGHS build, or a change to `A` would move the last digits; nothing "
        "here certifies cross-machine reproducibility or the long-term availability of the "
        "grid.",
        "* **The paper is a moving target.** F7-F11 were found against the manuscript and "
        "tables of 2026-11-01 16:5x; if the tables or the prose are regenerated, the "
        "traceability claims must be re-checked (the suite is re-runnable in about three "
        "minutes).",
        "",
    ]
    body = "\n".join(remaining) + "\n" + "\n".join(_section(r) for r in results)
    report = "\n".join(header) + "\n" + "\n".join(findings) + "\n" + body
    Path = __import__("pathlib").Path
    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.json).write_text(
        json.dumps({"generated": stamp, "git": git, "elapsed_s": round(elapsed, 2),
                    "checks": [r.as_dict() for r in results]}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nwrote {rel(Path(args.report))} and {rel(Path(args.json))}")
    print(f"summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_err} ERROR")
    if n_fail or n_err:
        return 1
    if n_skip:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
