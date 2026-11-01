# Tutorial 01 — input–output foundations

**Notebook:** [`notebooks/01_io_foundations.ipynb`](../../notebooks/01_io_foundations.ipynb)
· **Runtime:** ~2 s · **Prerequisites:** none

## Learning objectives

By the end you should be able to

1. read a symmetric input–output table and explain what `A`, `x` and `y` are, and why
   all three are in constant-price value units;
2. verify the accounting identity `x = A x + y` and say what the size of its residual
   measures (storage precision, not model error);
3. state and check the three equivalent productivity conditions for `A` (column sums
   below one, spectral radius below one, a finite non-negative Leontief inverse);
4. compute the Leontief inverse, read it as "all production chains summed", and use
   its column sums as output multipliers;
5. explain why a *capacity* shock is amplified only if the shocked sector has no
   final-demand buffer to absorb it — and quantify that with the static
   short-side programme;
6. compare the static amplification with what the dynamic engine produces, and say
   which channel makes the dynamic loss larger.

## What it establishes (with the numbers)

* 20 sectors, 2020, Eurostat `naio_10_cp1700`, €25.1 bn of gross output; the identity
  holds to **1.2e-7** relative — the precision at which `A` is stored in the
  committed CSV.
* `max column sum = 0.675`, `rho(A) = 0.468`, Neumann series converges to
  `(I-A)^{-1}` to 9e-16.
* Output multipliers (column sums of `L`) run from **1.45 to 2.36**.
* A 50% capacity cut applied to one sector at a time: **12 of 20 sectors absorb it**
  (aggregate output loss equals the direct loss, and *other* sectors' final
  deliveries actually rise); **8 transmit it**, with amplification from 1.48 up to
  **73.9** for `c02`, whose output is 91% intermediate.
* `c12`: direct capacity loss 1.75% of base output → static output loss **8.23%**
  (4.70×), static final-delivery shortfall 2.26×; the dynamic engine gives **10.0%**
  output loss (5.72×) and a peak shortage of **19.05%** (10.89×).

## Path through the notebook

0–1: load and inspect · 2: the identity · 3: productivity · 4: the Leontief inverse ·
5: final-demand multiplier · 6: capacity shock and the two regimes · 7: static vs
dynamic · 8: summary ledger and exercises.

## Exercises

1. Hand-check the multiplier bound: `1/(1-0.675) = 3.08` versus the largest actual
   column sum, 2.36. Why is the bound not attained?
2. Rerun §6 with a 30% cut. Which sectors still have `amp_output == 1`? Explain the
   pattern with `final_share_y/x` (hint: a sector that sells mostly to final demand
   can absorb the cut in its own deliveries).
3. Build an `A` whose largest column sum is 0.99 (legal but nearly singular) and count
   how many Neumann terms you need for `1e-10`. Compare with `rho(A)^k`.
4. Add inventories (`inventory = 0.1 * x0`) to the §6 programme. Is `amp_output` still
   1 for the absorbing group? Why?
5. Show that imposing `y >= 0` adds no constraint to the short-side programme, because
   `y = x - Ax >= 0` is exactly `(A-I)x <= 0`.

## How to know you got it right

The final cell prints eleven checks and `ALL 11 CHECKS PASSED`. The two amplification
regimes must both appear (12 absorbed / 8 transmitting); if every sector shows
amplification 1.00, the capacity upper bound is not binding and you are solving a
different programme.

## Common failure modes

* Reading `residual` as model error: it is CSV storage precision.
* Reporting only aggregate output: §6 shows why the distribution is the object.
* Forgetting that the static model has *no* inventory and *no* time — which is exactly
  why §7's comparison matters.
