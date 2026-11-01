# Tutorial 04 — the mechanism zoo

**Notebook:** [`notebooks/04_mechanism_zoo.ipynb`](../../notebooks/04_mechanism_zoo.ipynb)
· **Runtime:** ~6 s · **Prerequisites:** tutorials 01 and 03

## Learning objectives

1. describe the three-method mechanism contract and why everything else belongs to
   the engine;
2. enumerate the registry with `available_mechanisms()` and explain why a notebook
   must never hard-code mechanism names;
3. run every registered institution on one identical shock and compare shortage,
   welfare, plan fulfilment, price behaviour and planning effort;
4. explain the common-random-numbers argument, and *demonstrate* that the scenario id
   is part of the seed (so per-mechanism ids would break the pairing);
5. distinguish "fulfilling the plan" from "serving households", and "planning effort"
   from runtime;
6. read the welfare ranking without over-reading it: with a concave welfare
   functional, the number of starved sectors dominates total consumption.

## What it establishes (with the numbers)

* Six mechanisms, one cell, one seed: `lp_central` has the lowest mean shortage
  (0.0466) and the highest plan fulfilment (0.9885); `material_balance` has the
  highest mean shortage (0.0801) and the lowest fulfilment (0.7532); peak shortages
  range from 0.1220 (`lp_central`) to 0.2011 (`tatonnement`).
* At `t = 1`, **every mechanism observes bit-identical noisy demand** when they share
  the scenario id; a different id draws a different realisation — the pairing is a
  property of the id, not of the seed alone.
* With three seeds the ranking of `material_balance` and `lp_central` swaps relative
  to the single-seed ordering, while the seed noise (max sd 1.05) is much smaller
  than the between-mechanism spread (4.42).
* Price activity separates the institutions cleanly: `rationing_priority` and
  `lp_central` never move a price; `tatonnement` moves 51% of sector-quarters;
  `market` occasionally moves one by a factor of `e^3`.
* The welfare ranking correlates **-0.92** with the count of starved sector-quarters
  and only **-0.44** with mean consumption: with log welfare, *who* is served beats
  *how much*.

## Path through the notebook

0: the registry and the contract · 1: the cell · 2: run all six · 3: fulfilment vs
effort · 4: common random numbers (and the broken-pairing experiment) · 5: three
seeds · 6: prices · 7: what the table cannot prove (+ 7.1 why welfare is not a
consumption aggregate) · 8: exercises.

## Exercises

1. Hard-code `MECHANISMS = ["material_balance", "lp_central"]` and explain what the
   notebook would silently miss when a new institution lands.
2. Give each mechanism its own scenario id (`f"{CELL_ID}_{name}"`) and rerun §2. Did
   the ranking move? Is that noise or bias?
3. Set `noise = 0` and `sigma = 0`. Why does §4's assertion become trivially true?
4. Sweep `tatonnement`'s `price_gain` from 0.1 to 1.0. Where do you expect
   oscillation to start, given `g*(m) = 2 sin(pi/(2(2m+1)))` and `g = price_gain * b`?
5. Is a fair race possible? Every mechanism has at least one effort parameter. What
   does the pre-registration do about it, and why is "report the effect size"
   part of the answer?

## How to know you got it right

The final cell prints ten checks. The two that carry the identification argument are
`shared exogenous stream at t=1` (bit-identical) and `broken pairing changes the draw`.

## Common failure modes

* Comparing mechanisms across different scenario ids (destroys the pairing).
* Reading the welfare ranking as a ranking of economic systems (the information
  requirements differ, and the parameters are not tuned).
* Reporting plan fulfilment without shortage: a soft plan is easy to fulfil.
