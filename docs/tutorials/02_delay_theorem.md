# Tutorial 02 — the delay theorem

**Notebook:** [`notebooks/02_delay_theorem.ipynb`](../../notebooks/02_delay_theorem.ipynb)
· **Runtime:** ~5 s · **Prerequisites:** tutorial 01 is *not* needed; only linear algebra

## Learning objectives

1. write the feedback loop `u_{t+1} = u_t - g u_{t-m}` and say what `u`, `g` and `m`
   mean in the model (`g = eta * b`, `m = lag + 1`);
2. derive the characteristic polynomial `Q_m(z) = z^{m+1} - z^m + g` and reproduce the
   unit-circle proof of `g*(m) = 2 sin(pi/(2(2m+1)))`;
3. confirm the closed form numerically by bisecting on the spectral radius, to 1e-13;
4. test the same claim in the time domain with an envelope-growth estimator;
5. measure the boundary oscillation period `2(2m+1)` by counting sign changes;
6. explain precisely why "each extra period of delay halves the admissible gain" is
   wrong, and what the correct asymptotic statement is;
7. see what happens when the loop is *positive* (`b < 0`), which is the linearised
   shadow of "taut plans are self-defeating" (H1′).

## What it establishes (with the numbers)

* Boundary confirmed for `m = 0..14` to **5.5e-15** (the requirement was 1e-13); at
  `g = g*(m)` a conjugate pair sits on the unit circle at angle exactly `pi/(2m+1)`
  (checked to 1.3e-15) and every other root is strictly inside.
* Envelope growth changes sign at the boundary for `m = 1..6`.
* Period measured by sign changes agrees with `2(2m+1)` to within **0.08 quarters**,
  and at `g = g*` the amplitude neither grows nor decays (drift 1.000000).
* `g*(m)/g*(m-1)` rises from 0.50 (m=1) to 0.88 (m=8): the "halving" story is a
  small-`m` artefact. `g*(m) ~ pi/(2m+1)`, so **doubling** the delay halves the gain.
* The positive loop has **no** stable gain: any `g > 0` diverges monotonically.

## Path through the notebook

1: simulate · 2: characteristic polynomial · 3: the proof, and the exact root check ·
4: numerical boundary · 5: time domain · 6: period · 7: what the theorem does *not*
say, and the positive loop · 8: summary ledger and exercises.

## Exercises

1. Derive `m = 2` by hand: `theta_0 = pi/5`, `g*(2) = 2 sin(pi/10) = (sqrt(5)-1)/2`.
   Verify that `z = e^{i pi/5}` solves `z^3 - z^2 + g = 0`.
2. Show `Q_0(z) = z - (1-g)` and the classical `0 < g < 2`. Why does `char_poly` need
   an explicit `m = 0` branch?
3. Explain why the phase condition's `j = 0` branch gives the smallest `g` (consider
   `sin(theta/2)` on `(0, 2pi)`).
4. Add partial adjustment: `u_{t+1} = (1-a) u_t - g u_{t-m}`. Find the new boundary
   numerically and explain why `a > 0` enlarges the stable region.
5. Read `docs/notes_delay_theorem.md` §5 and explain why the gain `b` must be measured
   in the full model rather than derived from `A`.
6. If `results/` exists, plot the measured critical `price_gain` for `tatonnement`
   against `g*(k+1)/b_p`. The pre-registered falsification rule (protocol §6) is "more
   than one lag period of disagreement".

## How to know you got it right

The final ledger prints seven checks; the boundary error must be below 1e-13. If you
see the boundary correct but the period wrong, you are looking at a different
mechanism than the one the theorem describes — which is exactly what the period is
there to detect.

## Common failure modes

* Using the model's `lag = k` as `m`: the effective delay is `k + 1`, because a
  realised outcome can only be reported after the period in which it happened.
* Quoting "each extra period halves the gain" (wrong for `m > 2`).
* Forgetting that this is a *local* statement about a linearised loop.
