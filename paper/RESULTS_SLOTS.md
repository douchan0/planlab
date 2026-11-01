# Results slots — what the paper needs from the experiment wave

Every placeholder below is a `[[...]]` token in `paper/main.md` that the builder
cannot resolve, or a registered float whose data file has not arrived.
`python scripts/build_paper.py` prints the open list and `paper/build_report.json`
records it machine-readably.

## State of play (wave 3)

Sections 6.1, 6.2, 6.3, 6.4 (the H1 null), 6.5 (propagation), 6.7 (the dual track)
and 8 are **written**, from the Lead's reference run (`scripts/lead_reference.py`,
real 20-sector Eurostat substrate, amended parameters) and the verifier's causal
isolation. They quote numbers that `experiments` will re-derive into `results/`.
When those files land, every number must be checked line by line; where they
disagree, **the files win and the prose changes**. That is the protocol's reporting
rule and it is not negotiable.

One mechanism was revised three times before it could be defended, and the version in
§6.3 is the one the verifier's causal isolation supports: the binding margin is
**labour**, the cost appears only when the revised plan needs more labour than exists,
and the inventory channel is exactly zero in the operating regime. Do not reinstate
any inventory-based explanation, and do not describe the loss as arising from
"asymmetry in the inventory response": that hypothesis was tested and is false.

Twelve slots remain. They are held open deliberately: a missing result must look like
a missing result.

Two pre-registered hypotheses are reported as **falsified** (H1's crossing, H5's
interior optimum) and must stay falsified in any later edit.

## How to fill a slot

| kind | drop-in file | then |
|:--|:--|:--|
| figure | `paper/figures/<name>.png` (300 dpi, ≤ 6.5 in wide) | write `paper/figures/<name>.cap`; the `[[FIG:name]]` token is already placed |
| table | `paper/tables/<name>.csv` (header row, 6 dp) or `<name>.md` | write `paper/tables/<name>.cap`; the `[[TABLE:name]]` token is already placed |
| prose | nothing to drop in | replace the `[[RESULTS:...]]` token |

Captions may contain `[[REF:...]]`, `[[@key]]`, `$math$` and `` `code` ``. Never type
"Table 7" or "Figure 3" by hand; the builder numbers floats and resolves references.

## Floats still to land

### `paper/tables/noise_only.csv` — Table, §6.2
One row per mechanism, **20 seeds**, noise-only configuration (lag 0, no shock,
$\sigma=0.03$, demand AR(1) 0.3/0.02, stock ratio 0.10, $T=80$, burn-in 30).
Columns: mechanism id; output/base mean; output/base SD; mean shortage rate;
shortage SD; mean welfare; welfare SD; mean cascade iterations.
Reference values already quoted in §6.2 — the file must reproduce them:
`hybrid_dual_track` 0.9948 [0.0040] / 1.64% [0.11]; `tatonnement` 0.9937 [0.0008] /
1.56% [0.12]; `market` 0.9889 [0.0078] / 1.64% [0.09]; `rationing_priority` 0.9813
[0.0041] / 1.64% [0.13]; `lp_central` 0.9712 [0.0069] / 1.61% [0.09];
`material_balance` 0.8761 [0.0115] / 4.35% [0.82].

### `paper/tables/labour_slack.csv` — Table, §6.3 (new)
The labour-slack sweep that makes the cost of taut planning conditional. Fix the
revision gain at 1.0 and the balancing rounds at 3, vary the labour supply factor,
and report output/base and mean shortage. Reference points: factor 1.15 → 0.8698,
1.40 → 0.9922, 2.00 → 1.0000, 3.00 → 1.0000. `experiments` is being asked to vary
labour_supply_factor in B6, so this table may extend the sweep.

### `paper/tables/balance_decomposition.csv` — Table, §6.3
The 7-row grid crossing `revision_gain` with `balance_rounds` in the same noise-only
environment, 16 seeds. Columns: revision gain; balancing rounds; output/base; mean
shortage. Reference values, in the order the prose uses them, are
(0.0, 0, 0.9999, 1.64%), (0.0, 3, 0.9999, 1.60%), (1.0, 0, 1.0000, 1.60%),
(0.3, 1, 0.9472, 3.37%), (0.3, 3, 0.9367, 4.25%), (1.0, 3, 0.8711, 4.78%).
The Lead has since quoted the last cell as a range across seeds, 0.8698–0.9087; if the
file disagrees with the single figure in the prose, quote the range and say so. The caption should say plainly that the interaction is the
finding: neither factor alone moves output, and both together do.

### `paper/tables/calibration.csv` — Table, §6.1
One row per mechanism, zero lag / zero noise / zero shock, on both substrates:
output/base, mean shortage, mean welfare, max absolute accounting residual,
base-year reproduction error. Reference values: output/base $1.000000$, shortage
$0.000000$, welfare $259.568805$ (dual track $259.568804$).

### `paper/tables/main_lag_grid.csv` — Table, §6.4
Full B2 grid: institution; `price_gain` or `revision_gain`; reporting lag $\ell$;
effective delay $k=\ell+1$; mean and SD of shortage, welfare loss, detrended
amplitude, lag-1 autocorrelation, plan fulfilment, cascade iterations. Twenty seeds.

### `paper/tables/incidence.csv` — Table, §6.5
B3 incidence: shocked sector; forward-linkage centrality; `shock_output_amplification`
(primary H4 statistic, pilot 1.25–5.7); `shock_amplification` (distributional ratio,
pilot 2–11); household / intermediate / investment-government shares of the unmet
demand for the shocked good (pilot: up to 85 per cent on households).

### `paper/figures/welfare_vs_lag.png` — Figure, §6.4
Welfare loss against effective delay $k$, one line per institution, error bars from
the bootstrap CI of the paired difference against `baseline_lp_k0`, with the fitted
slope per institution drawn — H2 is a statement about slopes, not levels.

### `paper/figures/stability_boundary_empirical.png` — Figure, §6.4
Two panels. (a) amplitude and lag-1 autocorrelation against gain for each $m$, with
the measured breakdown marked (expected: none, within sampling variation). (b)
measured loop gain $|g|=\eta\,b_p$ against the analytic $g^\star(m)$ on a log axis,
so that the one-to-two-orders-of-magnitude gap is visible; **the 45° line must not be
drawn**, because the point of the panel is that the measured points are nowhere near
it. Label the axis in $m$, not $\ell$.

## Prose slots

| slot | what is needed |
|:--|:--|
| `[[RESULTS:B2]]` | H1' test on the target loop (shortage monotone in `revision_gain` **only with balancing on**, which §6.3 has already argued); the ordering test of H2 across the full lag grid; the $|g|k$ test of Corollary 2. Report the H1 null explicitly as a null with its measured distance to the boundary. |
| `[[RESULTS:B3]]` | `shock_output_amplification` by sector and institution, its correlation with forward-linkage centrality (H4), and the incidence decomposition. |
| `[[RESULTS:B4]]` | Rationing against shadow prices at three shock sizes, split into frontier and incidence components per Corollary 3. |
| `[[RESULTS:B5]]` | Welfare and volatility against $\theta$ at $\ell\in\{0,3\}$; the interior-optimum test for H5. |
| `[[RESULTS:B6]]` | One paragraph per perturbation, stating explicitly whether each headline ordering survives. |
| `[[RESULTS:INTRO_FINDINGS]]` | Four to six sentences at the end of §1, in test order: H1 (null, with the reachability argument), H1' (the interaction), H2 (ordering by information requirement), H3, H4, H5. Negative results as plainly as positive ones. |
| `[[RESULTS:ABSTRACT_FINDINGS]]` | Two or three sentences: the price loop is stable because it is too weak to be otherwise; the quantity loop is expensive because balancing is one-sided; the rent is the interaction of revision with balancing. |
| `[[RESULTS:DISCUSSION]]` | The whole of §7, organised around: (i) what the unreachable boundary implies for the informational reading of the calculation debate; (ii) the one-sidedness of the feasibility map as the general mechanism, not just a PlanLab artefact; (iii) what the dual-track result implies; (iv) the limits of a single substrate. |

## Numbers the prose must not invent

1. `b_p` per sector, its SD across seeds, and the derived $\eta^\star(m)=g^\star(m)/b_p$.
2. The measured loop gain $\eta\,b_p$ at each gain in the grid, and the resulting distance to $g^\star(m)$.
3. The first `revision_gain` at which shortage exceeds the gain-zero level by three SD, **with balancing on and with balancing off** — the contrast is the result.
4. The H2 slope per institution with cluster-robust SEs, ordered.
5. Mean shortage under `material_balance` at a 50 per cent capacity shock (H3 threshold: 0.5 per cent).
6. Both amplification ratios and the household share of the unmet demand.
7. $\theta^\star$ at $\ell\in\{0,3\}$ with bootstrap CI and whether it is interior (H5).
8. Cascade iterations per period by mechanism and by `balance_rounds`.

## Rules that survive the deadline

The pre-registration and `docs/AMENDMENTS.md` fix the grid, seeds, outcomes and tests.
If a number is needed that no block produces, report it as missing in §8 — do not run
a new cell and present it as pre-specified. A failed hypothesis is reported as failed;
it does not get re-framed as exploratory. §7 is where interpretation belongs.
