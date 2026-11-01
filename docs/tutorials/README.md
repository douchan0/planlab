# Tutorials — the PlanLab teaching notebooks

One short page per notebook: what it establishes, what you need before starting, the
path to follow, and the exercises. The notebooks themselves are in
[`notebooks/`](../../notebooks/), and [`notebooks/README.md`](../../notebooks/README.md)
is the index (including the suggested three-hour teaching sequence).

| page | notebook | runtime |
|---|---|---|
| [01 — input–output foundations](01_io_foundations.md) | `01_io_foundations.ipynb` | ~8 s |
| [02 — the delay theorem](02_delay_theorem.md) | `02_delay_theorem.ipynb` | ~5 s |
| [03 — one period, opened up](03_one_period_walkthrough.md) | `03_one_period_walkthrough.ipynb` | ~13 s |
| [04 — the mechanism zoo](04_mechanism_zoo.md) | `04_mechanism_zoo.ipynb` | ~6 s |
| [05 — dual track and shadow prices](05_dual_track_and_shadow_prices.md) | `05_dual_track_and_shadow_prices.ipynb` | ~14 s |
| [06 — reproducing a paper table](06_reproduce_a_paper_table.md) | `06_reproduce_a_paper_table.ipynb` | ~14 s |

## Order

**01 → 02 → 03 → 04 → 05 → 06.** Each one uses only what came before it:

* 01 gives you the technology and the accounting identity;
* 02 gives you the one closed-form result (it needs no PlanLab import at all);
* 03 gives you one period of the engine, with every step recomputed by hand;
* 04 uses 01–03 to compare all six institutions;
* 05 uses 04's registry, plus the LP of 03, to price the plan;
* 06 uses everything, and the `results/` pipeline.

You can read 01 and 02 independently. 03 is the hinge: if you only have time for one
notebook, make it 03.

## Background assumed

Undergraduate linear algebra (inverses, eigenvalues), the Leontief input–output model
at the level of a first macro course, and enough Python to read NumPy. Delay
difference equations and linear-programming duality are *not* assumed — 02 and 05
develop what they need.

## What the suite is for

The repository's claim is that the institution is a replaceable module and the
physics is shared. The tutorials are where a reader can check that division for
themselves: 01–03 teach the shared physics, 04–05 the institutions, 06 how the
numbers in the paper are produced and how to falsify them.

## A note on honesty

Every notebook ends with an assertion ledger and fails if a number does not close.
The tutorials therefore do not say "the notebook should print roughly 0.19"; they say
what the number *is* and what would make it change. If your run disagrees with a
tutorial, the tutorial is the wrong thing to edit: check the substrate
(`data/processed/`, `data/metadata/manifest.json`) and the reference run described in
[`notebooks/README.md`](../../notebooks/README.md#where-the-numbers-come-from).
