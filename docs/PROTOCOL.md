# PlanLab — Research Protocol (pre-registration)

**Version:** 1.0 (frozen before any result was generated).
**Project:** PlanLab — a computational laboratory for planned-economy coordination mechanisms.
**Deliverables:** MIT-licensed simulation core · reproducible data pipeline · 16 standard
scenarios · methodology paper · teaching notebook suite.

This protocol fixes *what will be measured and how* before running the experiments, so
that the reported results are confirmatory rather than exploratory. Deviations are
recorded in `docs/DEVIATIONS.md` with date and reason.

---

## 1. Research question (primary)

**How does delay in the information available to a central coordinator — that is, how
stale and noisy the planner's picture of demand, capacity and inventories is — affect
the stability and welfare performance of alternative coordination mechanisms in an
input–output economy; and which mechanism damps, and which amplifies, the resulting
shortage dynamics?**

Sub-questions, each tied to one experiment block:

* **Q1 (delay–stability).** Is there a critical information lag beyond which
  administrative price adjustment (Lange-type tâtonnement, cost-plus marking-up) becomes
  oscillatory, and does the observed critical lag match the analytic stability boundary
  of the linear delay-difference approximation?
* **Q2 (mechanism ranking).** Under a fixed information regime, how do material-balance
  planning, priority rationing, shadow-price coordination, competitive markets and
  dual-track hybrids rank on shortage, plan fulfilment, welfare and volatility? Does
  the ranking reverse as lag increases?
* **Q3 (propagation).** Does the sectoral network (the I-O matrix) amplify a single-sector
  capacity shock, and how is the amplification moderated by priority rules and by which
  sector is hit?
* **Q4 (hybrid).** In a plan–market dual track with plan share θ, is there an interior θ
  that dominates both pure planning and pure markets on a welfare–volatility trade-off?

**Framing.** The contribution is *not* "a planned-economy simulator". It is a controlled
comparison of coordination institutions on an identical technological substrate, plus a
small theory of how delay destabilises administratively set signals.

---

## 2. Hypotheses (stated as falsifiable predictions)

* **H1.** For the tâtonnement mechanism, there exists a critical lag `k*(η)` such that for
  `k < k*` the price path converges to the neighbourhood of equilibrium after a shock,
  and for `k >= k*` it exhibits persistent oscillation (spectral radius of the linearised
  delay system crossing 1). The shaded-injury prediction: `k*` decreases as the step size
  `η` grows.
  *Primary test:* spectral radius of the characteristic polynomial of the linearised
  system vs. observed oscillation amplitude and the sign pattern of autocorrelation in
  simulated price deviations up to lag 4.
  **Falsified if** simulated stability boundary differs from the analytic boundary by
  more than one period of lag or the oscillation amplitude at `k >= k*` is not
  significantly larger (one-sided Mann–Whitney, α = 0.05) than at `k = 0`.
* **H2.** Effective coordination under delay is *ordered by the amount of up-to-date
  information the mechanism needs*: the omniscient LP (which needs full current data)
  degrades fastest in realised welfare as `k` rises, while material balance (which uses
  only plans, not current demand) degrades slowest.
  **Falsified if** the welfare loss slope against `k` is not largest for `lp_central`.
* **H3.** Material-balance planning does *not* eliminate shortages: shortage rate is
  positive and increasing in the shock size, despite plan targets being "balanced".
  **Falsified if** mean shortage rate under a 30% sectoral capacity shock is < 0.5% .
* **H4.** A single-sector capacity shock produces downstream shortfalls whose aggregate
  share exceeds the direct impulse (amplification > 1), and amplification is larger when
  the shocked sector has a high forward-linkage centrality.
  **Falsified if** mean amplification over the 20 shocked sectors is ≤ 1.
* **H5.** The dual-track mechanism has an interior optimum in θ on the
  welfare–price-volatility frontier: `argmax_θ W(θ)` lies strictly inside (0, 1).
  **Falsified if** welfare is monotone in θ over the tested grid.

Every hypothesis maps to pre-specified metrics (§5) and a pre-specified test (§6).
Exploratory findings are labelled as such in the paper.

---

## 3. Model (summary; exact spec in `docs/ARCHITECTURE.md`)

* 20-sector Leontief economy, quarterly periods, gross output, inventories, capital with
  capacity `= K/κ`, labour.
* Direct input coefficients `A` from a published symmetric input–output table (Eurostat
  `naio_10_cp1700`, EU27, 2020, basic prices), aggregated 64 → 20 sectors with the
  mapping in `data/metadata/sector_map.csv`; a synthetic, fully documented generator is
  provided as the offline fallback and as a robustness substrate.
* Production is **short-side**: output is the largest feasible vector given targets,
  capacity, labour and actually available intermediates (monotone fixed-point, §5 of the
  architecture document). This is the mechanism through which a shortage in one sector
  spreads to its users.
* Information: the coordinator observes `(demand, capacity, inventory, backlog, prices)`
  from `k` periods ago plus log-normal measurement noise `σ`. Mechanisms never see
  current truth.

### 3.1 Mechanisms (institutions) — all on the same technology

| id | institution | information it uses | allocation rule |
|---|---|---|---|
| `lp_central` | omniscient central LP: `max w·y s.t. (I−A)x ≥ y, x ≤ cap, y ≥ 0` | latest observation of demand & capacity | optimal plan, pro-rata fulfilment |
| `material_balance` | resource–need balancing, `R` iterative rounds, priority tier first | plan targets + observed stocks | proportional within priority tier |
| `rationing_priority` | administrative quotas, tier 1 then tier 2 | observation | quota per claimant, priority order |
| `tatonnement` | Lange-style price board: `p ← p + η·excess demand`, price-mediated claims | observation + own prices | willingness-to-pay order |
| `market` | within-period Walrasian tâtonnement, flexible prices | current prices within period | market clearing |
| `hybrid_dual_track` | plan track (share θ at admin price) + market track (1−θ) | mixed | two-tier |

All mechanisms receive the *same* observation object and the same technology; only the
institution differs. This is the core identifying assumption of the design.

---

## 4. Experimental design (16 standard scenarios → grid)

**Common settings:** `T = 60` quarters, 30 burn-in quarters discarded from statistics
(so the system has settled), base year 2020 table, welfare weights `w_i = 1` for
consumption goods and `1.2` for essentials tier, `κ` from the IO capital coefficients
(or, if unavailable, a flat capital-output ratio of 2.5, documented).

| block | what varies | levels | runs |
|---|---|---|---|
| **B1 calibration/validation** | nothing (deterministic) | 2 scenarios × 1 seed | 2 |
| **B2 lag × mechanism** | `k ∈ {0,1,2,3,4,6}`, mechanism ∈ 6 ids | 6 × 6 | 36 cells × 20 seeds = 720 |
| **B3 shock propagation** | shocked sector ∈ 20, mechanism ∈ 3 (material_balance, rationing_priority, tatonnement) | 60 cells | 60 × 10 seeds = 600 |
| **B4 rationing vs shadow price** | mechanism ∈ {rationing_priority, tatonnement}, shock size ∈ {10%, 30%, 50%} | 6 cells | 6 × 20 seeds = 120 |
| **B5 dual track** | θ ∈ {0, .1, ..., 1} (11 levels), lag ∈ {0, 3} | 22 cells | 22 × 20 seeds = 440 |
| **B6 robustness** | substrate (eurostat vs synthetic), `η`, `σ`, `A`-perturbation | 12 cells | 12 × 10 seeds = 120 |

Total ≈ 2 000 runs × 60 periods ≈ 120k period records. Runtime target: < 20 min on one
laptop core per block, parallel across cells.

**Seeds.** Seed `s ∈ [0, S)` drives (i) demand noise, (ii) capacity shocks, (iii)
measurement noise. Identical seed → identical exogenous path across mechanisms, so
mechanism comparisons are *paired* (common random numbers). This must be preserved:
`planlab.experiments.grid` derives all exogenous streams from `(scenario_id, seed)`.

**Baseline reference.** `baseline_lp_k0` with no shocks, no noise, no lag defines the
welfare normalisation; all reported welfare is relative to it unless stated.

---

## 5. Pre-specified outcomes (primary / secondary)

**Primary (used for hypothesis tests):**
1. `mean_shortage_rate` (post burn-in)
2. `cum_welfare` (discounted sum of `Σ w_i log(C_i + c0)`, β = 0.98)
3. `welfare_loss_vs_baseline` = `1 − cum_welfare / baseline_cum_welfare`
4. `oscillation_amplitude` = std of the HP-filtered residual of `log p(t)`? No — simpler
   and pre-registered: std of the deviation of `mean log price` from its 20-period moving
   average, plus its lag-1..4 autocorrelation.
5. `amplification` (B3 only): `sum_i Δshortfall_i / Δcapacity_shock`, evaluated at the
   shock peak.
6. `plan_fulfilment_mean`.

**Secondary:** inventory ratio, capital utilisation, plan deviation, price dispersion,
cascade iteration count (planning effort), recovery half-life, shortage Gini across
sectors.

**Never reported alone:** aggregate output. (Same output can hide opposite distributional
outcomes — a point the paper makes explicitly.)

---

## 6. Statistical procedures (fixed in advance)

* Unit of analysis: the run (one mechanism × seed × cell). Periods are not independent,
  so inference uses run-level means, never period-level pooling.
* Two-mechanism comparisons: paired differences (common random numbers) → Wilcoxon
  signed-rank test + bootstrap 95% CI (10 000 resamples) of the paired difference.
* Multi-mechanism comparisons: Kruskal–Wallis + Holm-corrected pairwise Mann–Whitney.
* Lag effect: linear and quadratic regression of run-level outcome on `k` with
  mechanism fixed effects; report slope with cluster-robust SE by seed.
* Stability boundary: for each η, the smallest k whose lag-1 autocorrelation of the
  price deviation exceeds 0.5 and whose amplitude exceeds the k=0 amplitude by ≥ 3 SD.
* Multiplicity: Holm correction within each block; α = 0.05.
* Effect sizes always reported (Cliff's δ or paired-mean/σ).

---

## 7. Validity threats and how they are addressed

| threat | mitigation |
|---|---|
| "computational feasibility mistaken for institutional feasibility" | the planner never sees current data; delay and noise are first-class treatments; we report planning-effort metrics, not only outcomes |
| welfare weights are normative | weights are scenario parameters; every headline result is re-run at ±50% on the shortage and inventory penalty (B6) |
| unit heterogeneity (value vs volume) | all quantities constant-price value; stated as a limitation; aggregation across sectors is value-weighted |
| one I-O table, one year | two substrates (real + synthetic) and a 5% random perturbation of `A` in B6 |
| simulation overfitting to chosen parameters | pre-registered grid, 20 seeds per cell, common random numbers, Holm correction |
| mechanism confounds (different agents → different everything) | identical technology, identical observation, identical shock streams; only the coordination rule differs |

---

## 8. Analysis plan → artefacts

1. `results/` populated by `scripts/run_all.py` (blocks B1–B6), each block with
   `manifest.json`.
2. `results/tables/*.tex|csv` produced by `planlab.experiments.aggregate` — the paper's
   tables are generated, never hand-typed.
3. `figures/*.pdf` produced by `planlab.experiments.figures`.
4. `paper/main.md` → `paper/main.html` → `paper/main.pdf` via `scripts/build_paper.py`
   (Markdown + KaTeX/MathJax, Edge headless print-to-PDF; no LaTeX needed).
5. `verification/report.md` by an independent verifier that re-runs a random sample of
   cells from scratch and re-derives the analytic stability boundary.

## 9. Reporting rules for the paper

* Every number in a table or in the text that summarises data must trace to a file in
  `results/` (or `verification/`) — no hand-computed statistics.
* Report both the mean and the dispersion across seeds; never a single trajectory as
  evidence.
* Negative and null results are reported with the same prominence (e.g. if H2 fails).
* Limitations section must include: constant-price value units, single-year calibration,
  Leontief rigid technology, no money/finance, no intertemporal optimisation by
  households, no incentive/gaming behaviour (left as future work).
