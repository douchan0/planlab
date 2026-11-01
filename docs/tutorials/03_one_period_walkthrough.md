# Tutorial 03 — one period, opened up

**Notebook:** [`notebooks/03_one_period_walkthrough.ipynb`](../../notebooks/03_one_period_walkthrough.ipynb)
· **Runtime:** ~3 s · **Prerequisites:** tutorial 01 (the technology and the identity)

This is the flagship notebook of the suite. Its promise: **every number is computed
twice** — once by the engine, once by explicit code you can read — and the two must
agree, or the notebook fails. If they disagree and none of the five failure modes
below explains it, that is a bug in the model worth reporting, not a number to adjust
in the prose.

## Learning objectives

1. list the ten steps of an engine period and say which three the mechanism owns;
2. build an `Observation` and explain the timing convention (`t-1-k`, effective lag
   `k+1`) and why `lag = 0` is "last quarter's returns", not clairvoyance;
3. read an `Allocation`: rows are claimants, columns are goods, `tiers` is the
   priority ladder, and the ladder is an *institutional* choice;
4. solve the short-side programme by hand on a three-sector economy and prove
   optimality with an inequality chain;
5. implement priority rationing yourself and show it equals the engine's;
6. audit the balance identity `S0 + x = Z + C + G + I + S_end` and, separately, the
   *rationing ledger* that says where every granted unit went;
7. instrument the engine without modifying it (delegation-only proxy + `_reconcile`
   subclass) and prove the instrumentation is behaviour-neutral.

## What it establishes (with the numbers)

* **Three-sector hand example** (all integers): `A`, `x0 = (100,200,300)`,
  `y0 = (50,110,230)`; inventory `(10,10,30)`; capacity `(100,60,300)`; target
  `(100,140,300)`. The programme's optimum is `(100, 60, 270)` with total output 430,
  proved by hand: `0.2 x_c <= 70 - 10 - 6`. Rationing gives the household nothing of
  good *b* (intermediates absorb all 70 units), and the identity closes with residual
  0.0 exactly.
* **Real period** (t = 47, the peak): the instrumented run is bit-identical to the
  plain run in all 60 periods; the engine needs **1 pass**; re-deriving the period
  from the written spec reproduces `x, Z, C, G, I, S_end, unmet` with **0.0**
  difference; the balance identity closes to **2.4e-17**.
* **Tier serving** at the peak: intermediates **100.00%**, essential final 82.2%,
  other final 74.5%, government 88.4%, investment 83.0%, stockbuilding 0%.
* **Plan tautness**: all 20 sectors are asked for more than their capacity; 9 sit at
  `1.5 × observed capacity`, and 0 at the `2.5 × base` cap — so the stale capacity
  estimate enters the plan directly through its ceiling.
* **Metric cross-checks**: `shortage_rate` and `welfare` recomputed from the raw
  arrays equal the stored metrics to machine precision; the stored `PeriodRecord`
  arrays are bit-identical to the hand-recomputed ones.
* **Lag**: at `t = 49` (the shock is over) the `lag = 0` planner sees period 48 and
  the `lag = 3` planner sees period 45; `c12`'s perceived capacity differs by 49.5%.

## Path through the notebook

0: the ten steps · 1: the three-sector economy · 2: observation and planning ·
3: the claim matrix · 4: the feasibility programme by hand · 5: rationing and
accounting (+ 5.1 the ledger) · 6: the real period (6.1 tautness, 6.2 recomputation,
6.3 the account, 6.4 who is sacrificed, 6.5 the lag) · 7: the receipt · 8: debugging
guide · 9: exercises.

## Exercises

1. **Break it.** Set `ladder="consumption_first"` and rerun §6.2. The engine now needs
   more than one pass; find the period that needs the most, and explain why the
   reserve logic exists.
2. Raise `stock_ratio` from 0.10 to 0.30 and rerun. What happens to `S_end` and
   `unmet`? Does the peak quarter move? (Inventory only changes `S0`.)
3. Replace `S_end = maximum(supply - used, 0)` with `supply - used` and rerun §6.2.
   Does it still pass? Is the `maximum` insurance or necessity?
4. Take the three sectors with the largest `S_end` in §6.3 and name, for each, which
   term of the rationing ledger produced it.
5. **Is the identity a tautology?** `Z` comes from the technology, `C` from
   rationing, `G`/`I` from demand caps. Show that this makes the identity a genuine
   constraint rather than an accounting identity that closes by construction, and
   identify which step makes it non-trivial.

## How to know you got it right

The receipt prints **26 checks** with the measured error in each. The two that matter
most: `hand == engine for every array` (0.0) and `balance identity (20 sectors)`
(2.4e-17). Anything above 1e-12 in the first is a real disagreement.

## Common failure modes

1. Computing intermediate requirements from `targets` instead of the realised `x`.
2. Forgetting `C = min(grants_final, demand)` — granted can exceed demanded, and the
   excess returns to stock.
3. Treating `S_end` as a residual rather than a line item.
4. Float-comparing tiers (`<` with a tolerance matters).
5. Dropping the labour constraint (slack here, not always).
