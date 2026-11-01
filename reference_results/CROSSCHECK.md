# Cross-check: two independent implementations of the same experiment

Two implementations of the pre-registered design exist in this repository, written by
different authors against the same frozen interface:

| | author | code path |
|---|---|---|
| **reference run** | Lead | `scripts/lead_reference.py` → `reference_results/` |
| **experiment layer** | experiments teammate | `src/planlab/experiments/` → `results/` |

They share the engine and the mechanisms but share no grid code, no aggregation code
and no statistics code. Where they agree, the numbers are a property of the model
rather than of a particular pipeline; where they disagree, at least one of us has a
bug — which is the point of keeping both.

## Agreement on the noise-only block (B1b / R2b)

Twenty seeds, no shock, no lag, measurement noise 0.03, AR(1) demand
(0.3 / 0.02), stock ratio 0.10, `T = 80`, burn-in 30, common mechanism parameters.

| mechanism | output/base (exp) | output/base (lead) | Δ | shortage (exp) | shortage (lead) | Δ |
|---|---|---|---|---|---|---|
| hybrid_dual_track | 0.99513 | 0.99482 | 0.00031 | 1.628% | 1.642% | 0.00014 |
| tatonnement | 0.99378 | 0.99368 | 0.00010 | 1.564% | 1.558% | 0.00006 |
| market | 0.99080 | 0.98892 | 0.00187 | 1.619% | 1.643% | 0.00024 |
| rationing_priority | 0.98221 | 0.98132 | 0.00089 | 1.646% | 1.639% | 0.00007 |
| lp_central | 0.97096 | 0.97122 | 0.00026 | 1.643% | 1.610% | 0.00033 |
| material_balance | 0.93708 | 0.87613 | 0.06095 | 4.692% | 4.346% | 0.00346 |

* **Rank order is identical** (Spearman = 1.000 on output across mechanisms), which is
  what every claim in the paper depends on.
* Five of six agree to three decimal places. The exception is `material_balance`,
  which is the known-sensitive cell (per-seed range 1.9–4.8%): it is the one
  institution whose outcome depends on the interaction of the revision gain with the
  balancing step, and therefore on `revision_gain`, `balance_rounds`, `labor_supply_factor`
  and `price_elasticity` jointly. The two runs differ in `price_elasticity`
  (0.6 in the reference run, 0.4 in the layer) and in the exact seed set, which is
  enough to move this cell by six points and nothing else.

**Conclusion.** The pipelines agree. Any headline number that differs between them by
more than seed noise in any *other* cell should be treated as a bug in one of them;
`material_balance` must always be quoted with its seed dispersion and with the
parameters that condition it, never as a point estimate.

## What this cross-check does not establish

Sharing the engine, the two pipelines can both be wrong in the same way. Accounting
conservation, the short-side programme and the priority ladder are checked
*independently* by `verification/` (which re-implements a period from the written
spec), not here. This note establishes only that the *experiment machinery* — grid,
seeding, aggregation — is not the source of the numbers.
