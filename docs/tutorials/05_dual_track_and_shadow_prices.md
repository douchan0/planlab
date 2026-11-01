# Tutorial 05 — dual track and shadow prices

**Notebook:** [`notebooks/05_dual_track_and_shadow_prices.ipynb`](../../notebooks/05_dual_track_and_shadow_prices.ipynb)
· **Runtime:** ~14 s · **Prerequisites:** tutorials 03 (the programme) and 04 (the registry)

## Learning objectives

1. write out the central planner's LP in standard form and say what each block means;
2. explain the dual of a material-balance constraint as the marginal value of one
   more unit of that good — Kantorovich's resolving multiplier;
3. extract the duals with `scipy.optimize.linprog` and check them against the prices
   `lp_central` reports (they should be identical up to normalisation);
4. read a capacity bound's dual as an endogenous investment signal, and know its
   units (the *linear* welfare proxy, not log welfare);
5. sweep the dual-track share `theta` and trace the welfare–volatility frontier;
6. state honestly what a single-seed frontier can and cannot say about H5.

## What it establishes (with the numbers)

* The independently written LP (40 variables, 20 balance rows, 40 bounds) reproduces
  `lp_central`'s plan exactly, and its normalised duals equal the reported prices to
  **4e-16** in the peak period and 0.0 in the calm one.
* In the calm period the shadow prices are **1.1538** for essential goods and
  **0.9615** for the rest — i.e. the welfare weights, rescaled; dispersion
  `std(log p) = 0.073`.
* At the peak, `c12`'s shadow price is **11.19** (9.7× its calm value) and it is the
  only sector whose capacity binds; dispersion rises to **0.857**.
* The `theta` sweep (11 points): welfare falls from **247.44** at `theta = 0` to
  **247.19** at `theta = 1`, while price volatility falls from 0.0194 to 0.0. In this
  cell the welfare curve is **monotone**, so the single-seed `argmax` is at the
  boundary — the pattern that would falsify H5 if it survived 20 seeds.
* The mechanism's own diagnostics show the realised plan-track share exceeding
  `theta` at the peak (0.529 at `theta = 0.5`): when supply shrinks, the plan track's
  *relative* weight rises.

## Path through the notebook

1: the LP · 2: duals vs reported prices · 3: which constraint binds · 4: two cautions ·
5: the `theta` sweep · 6: demonstration vs test · 7: summary and exercises.

## Exercises

1. Change `output_penalty` from 1e-3 to 1e-1 and rerun §2. Do relative prices move?
   Does the set of binding constraints change?
2. Multiply your price vector by 2 and rerun the assertion. What does that tell you
   about whether the test is about absolute or relative prices?
3. In §3's right panel, which sector has the largest capacity value? Is it the shocked
   one? Why not?
4. Compare `theta = 1` with `material_balance` and `theta = 0` with `market`. They are
   *not* the same mechanism: `theta` changes allocation, not planning. Explain.
5. Rerun §5 with three seeds and error bars. Is the interior optimum still there? If
   the endpoint difference is smaller than the seed spread, what can you claim?
6. Test the dual theorem numerically: raise `c12`'s capacity by 1% at the peak period
   and compare the welfare gain with the capacity dual. Why do they differ?

## How to know you got it right

The final cell prints nine checks. The load-bearing one is
`duals equal the reported prices` in both periods; if that fails, your LP is not the
LP the mechanism solves (start with the demand definition: household + government +
investment, plus carried backlog).

## Common failure modes

* Forgetting that the mechanism provisions for *all* final uses, not just households.
* Comparing absolute price levels: the mechanism normalises to mean one, so only
  ratios are meaningful.
* Reading a boundary `argmax` as evidence about H5 — one seed is a demonstration.
