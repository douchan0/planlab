# Tutorial 06 — reproducing a paper table (and a figure)

**Notebook:** [`notebooks/06_reproduce_a_paper_table.ipynb`](../../notebooks/06_reproduce_a_paper_table.ipynb)
· **Runtime:** ~4 s (longer if `results/` is large) · **Prerequisites:** tutorials 01–05, and a populated `results/`

## Learning objectives

1. inventory `results/` without assuming any filename;
2. state a table's **aggregation rule** before checking equality — including the
   burn-in, the filter that defines the sample, and what "mean" is a mean *of*;
3. recompute a published table from the rawest available records, in two steps:
   per-period → per-run → per-cell;
4. assert equality at the precision the artefact is *stored* with, and distinguish a
   rounding difference from a bug;
5. re-render a committed figure from `results/` and compare pixel arrays rather than
   bytes;
6. audit the manifest: package versions, seeds, completeness flags;
7. degrade gracefully when `results/` is absent, by demonstrating the same chain on
   data the notebook generates itself.

## What it establishes (with the numbers)

* Per-run metrics in `summary.csv` are reproduced from `runs.parquet` to **below
  1e-12** for every run (it is the same floats, re-aggregated).
* The publication table is reproduced **completely** from `summary.csv`: the two
  integer columns (`n_seeds`, `lag_effective = lag + 1`) match **exactly**, and all
  ten numeric columns agree within **1e-5 relative** — the precision at which the
  committed CSV stores its numbers.
* The *aggregation rule itself* is inferred from the artefact: the notebook searches
  (block × reference elasticity × gain filter) and keeps the combination that
  reproduces the table's `n_seeds` column exactly, rather than assuming one.
* §4.1 cross-checks the results-layer table against the paper's own
  `paper/tables/lag_grid_reference.csv`. For the four mechanisms whose gain is not a
  treatment, the two agree to **1.6e-16 relative**; for the two gain-swept ones they
  differ by up to **67.5%** (material_balance, lag 1: 0.0597 pooled vs 0.0356 at the
  reference gain). That is not a rounding difference: the two tables answer different
  questions, and the notebook says so instead of averaging it away.
* The source block and the reference elasticity are **identified by matching run
  counts**, not assumed: the notebook finds the block whose grouped counts reproduce
  the table's `n_seeds` column.
* The committed figure is regenerated from `results/` and matches **pixel for pixel**
  (`max |dRGB| = 0`), after applying the figure module's own `_style()`.
* The manifest's recorded package versions equal the running kernel's; `cells_failed
  = 0` and `runs_completed == runs_planned`.
* The self-contained demo shows the aggregation chain end to end and that dropping the
  burn-in instead of applying it moves a cell mean by 57% of its value.

## Path through the notebook

1: inventory · 2: pick a table and write down its rule · 3: per-period recomputation ·
4: the whole table · 5: the figure · 6: self-contained demo · 7: manifest · 8: what to
do when `results/` is missing, plus exercises.

## Exercises

1. Reproduce a *different* table (e.g. the shock-propagation one). What is its
   aggregation rule? Which block does it come from? (You will have to identify the
   block the same way §2 does.)
2. Record the worst relative difference in §4. Raise the stored precision from 6 to 15
   significant digits in the table writer (in `src/planlab/experiments/`, not here) and
   say what the tolerance should become.
2b. Reproduce the paper's `lag_grid_reference.csv` *from the results layer* by adding the
   `gain == 0.30` filter. Do the pooled and sliced tables then agree for all six
   mechanisms? Which one belongs in a caption that says "at the reference gain"?
3. Compare "mean over periods, then over runs" with "mean over all period-run rows".
   When do they differ materially?
4. What would a data-only comparison of the figure miss? (Answer: axis limits, legends,
   colours, titles — a wrong `ylim` changes the picture without changing the data.)
5. Tighten §4's tolerance to `1e-12` and rerun. It fails. Explain why that failure is
   the *correct* behaviour, and what it teaches about writing assertions.

## How to know you got it right

Two statuses are printed at the end: the hard checks (always required) and the
*path-dependent* items. `FULL PATH` means the table and the figure were reproduced;
`GRACEFUL PATH` means `results/` was not ready and the notebook told you which
artefact was missing — neither is faked.

## Common failure modes

* Hard-coding the burn-in: different blocks use different ones. Infer it from
  `n_periods_used`.
* Comparing to 1e-12 what was stored at 6 significant digits.
* Regenerating a figure without the module's `_style()`: the sizes do not even match
  (and you would wrongly conclude the figure is stale).
* Comparing a gain-**pooled** results table with a reference-gain **slice** and calling
  the difference a bug (or, worse, hiding it by widening the tolerance).
* Writing the regenerated figure over the committed one: an audit must not modify
  what it audits.
