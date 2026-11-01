# Derivation notes: delay-induced instability of administratively revised signals

**For the paper's Section 4.** These notes contain the derivation, the numerical
verification recipe, and the exact mapping from the model's parameters to the
theorem's. The writer must check every step independently and is free to improve
the presentation; what must not change is the substance.

---

## 1. The feedback loop

Consider any signal that a coordinator revises in proportion to a gap it has
*already observed*. Two instances in PlanLab:

* **plan revision** (`material_balance`): output targets move with the observed
  shortage,
  `T_t = T_{t-1} (1 + eta * s_{t-1-k})`;
* **administered price revision**: `log p_t = log p_{t-1} + eta_p * s_{t-1-k}`.

In both cases `s` is a shortage *rate* (a relative gap), the revision coefficient
is `eta`, and the observation is stale by `k` reporting lags. Because a realised
outcome can only be reported after the period in which it happened, the **effective
delay is `m = k + 1`** periods: even a perfectly informed coordinator reacts to
last period's gap. PlanLab's `info.lag = k` therefore corresponds to
`m = k + 1` in the theory, and every figure must state which of the two is plotted.

## 2. Linearisation

Let `u_t = log T_t - log T*` (or `log p_t - log p*`) be the deviation of the
instrument from its stationary value, and let the gap respond locally as

    s_t = - b * u_t + O(u_t^2),        b > 0

(a higher planned output, or a higher price, reduces the observed gap). Substituting,

    u_{t+1} = u_t - g * u_{t-m},        g := eta * b > 0,   m >= 1.        (★)

`g` is the **loop gain**: the product of how hard the coordinator pushes (`eta`)
and how strongly the gap reacts to the instrument (`b`). `b` is not a free parameter
of the model — it is an equilibrium object that must be *measured* in the
simulation (see §5).

## 3. Proposition (stability boundary)

> **Proposition 1.** The zero solution of (★) is locally asymptotically stable if
> and only if `0 < g < g*(m)`, where
>
>     g*(m) = 2 sin( pi / (2(2m+1)) ).
>
> Numerically `g*(0) = 2`, `g*(1) = 1`, `g*(2) = 0.61803... = 1/phi`,
> `g*(3) = 0.44504...`, and `g*(m) ~ pi/(2m+1)` for large `m`. At `g = g*(m)` the
> system has a pair of roots on the unit circle at angle
> `theta = pi/(2m+1)`, so the oscillation period is `2pi/theta = 2(2m+1)` periods.

**Proof.** The characteristic polynomial of (★) is

    Q_m(z) = z^{m+1} - z^m + g.

*Sufficiency at small g.* `Q_m` is a Schur-stable perturbation of `z^{m+1} - z^m =
z^m(z-1)` only up to the point where a root crosses the unit circle, and roots
depend continuously on `g`, so instability sets in exactly when `|z| = 1` first
becomes possible. Hence it suffices to find the smallest `g > 0` for which
`Q_m(z) = 0` has a root on the unit circle.

*Roots on the unit circle.* Put `z = e^{i theta}`, `theta` real. Then

    0 = e^{i(m+1)theta} - e^{im theta} + g
      => e^{im theta} (e^{i theta} - 1) = -g.                                 (1)

Taking moduli, and using `|e^{i theta} - 1| = 2|sin(theta/2)|`,

    g = 2 |sin(theta/2)|.                                                     (2)

For `theta in (0, 2pi)` we may write `e^{i theta} - 1 = 2 i sin(theta/2) e^{i theta/2}`,
so (1) becomes

    2 sin(theta/2) * e^{i( m theta + theta/2 + pi/2 )} = -g,

and since the right-hand side is real and negative, the phase must satisfy

    (m + 1/2) theta + pi/2 = pi + 2 pi j,   j in Z,
    => theta = (pi/2 + 2 pi j) / (m + 1/2),  j in Z.

The admissible angles are `theta_m = (pi/2 + 2 pi j)/(m + 1/2)`; writing
`x_m = theta_m/2` these are `x_m = pi(1 + 4j)/(2(2m+1))` for `0 <= j <= m`, and
`x_m` runs from `x_0 = pi/(4m+2)` up to `x_m = pi - x_0`. Since `sin` increases on
`[0, pi/2]` and is symmetric about `pi/2`, `sin x_m >= sin x_0` with equality at
`j = 0` and `j = m`, so the *smallest* gain at which a root reaches the unit circle
is attained at `j = 0`. (Excluding `j >= 1` by asserting that `j = 0` gives the
"smallest angle" would be a gap: `theta_j` exceed `pi`, where `sin(theta/2)`
decreases again.) This gives the boundary angle

    theta_0 = (pi/2) / (m + 1/2) = pi / (2m + 1).

Substituting into (2):

    g*(m) = 2 sin( theta_0 / 2 ) = 2 sin( pi / (2(2m+1)) ).   ∎

*Necessity.* At `g = g*(m)` a conjugate pair lies exactly on the unit circle, so
the zero solution is not asymptotically stable; by continuity of the roots in `g`
the moduli exceed one for `g > g*(m)`.

*Corollary (period).* At the boundary the complex pair is `e^{± i theta_0}` with
`theta_0 = pi/(2m+1)`, hence the period is `2 pi / theta_0 = 2(2m+1)` periods, i.e.
`~ 4m`. **This is a second, independent testable prediction**: not only *where* the
system destabilises, but *how fast it oscillates* when it does. A system that goes
unstable with the right threshold but the wrong period would indicate a different
mechanism at work.

*Remark (no delay).* For `m = 0`, i.e. `u_{t+1} = u_t - g u_t`, the condition is
`0 < g < 2`: the classic `eta in (0, 2)` range of proportional feedback. The
admissible gain decays like `pi/(2m+1)`, i.e. like `1/m`, so **the delay-gain
product is bounded**: `m * g*(m)` increases to `pi/2`. Note that the *first* period
of delay is the most expensive (2 -> 1, a halving), but the marginal cost of a
further period falls rather than staying proportional: it is the *product* of gain
and lag, not the gain alone, that delay constrains. Delay is not a nuisance, it is
a budget.

## 4. Why this is the right object for the model

The model's period map is: observe `(k)` -> plan -> claims -> **feasibility
programme** -> ration -> record. Two lemmas make the reduction to (★) legitimate:

> **Lemma 1 (monotonicity of the short side).** Let `R(ub, r, reserve)` be the
> maximal feasible output of the goods-balance programme
> `max 1'x  s.t.  A x - x <= r - reserve, 0 <= x <= ub`. Then `R` is monotone
> non-decreasing in `ub` and in `r`. Consequently the realised output is a monotone
> function of the planned targets, which is exactly the property needed for the
> planner's revision loop to be a monotone feedback system.

*Proof.* Both `ub` and `r` enter only through the right-hand side and the upper
bounds of the programme; enlarging either enlarges the feasible set, so the optimal
value of a maximisation problem cannot fall. ∎

> **Lemma 2 (rationing is monotone and conservative).** For fixed claims and tiers
> the priority-filling rule `G` is monotone non-decreasing in supply and satisfies
> `sum_j G_j(s) <= s` componentwise.

*Proof.* Each claimant receives `min(1, remaining/claimed) * claimed` on a
good-by-good basis, and `remaining` decreases by exactly what is granted. ∎

Together, Lemmas 1-2 imply that the composition
"plan revision -> short side -> rationing -> observed gap" is a monotone map on the
compact box `[0, (1+eta) * target_max]`, so by Tarski it has a greatest fixed point
(hence a stationary plan exists), and its linearisation around that fixed point is
(★).

> **Proposition 2 (existence of a stationary plan).** The one-period map of the
> model with a proportional revision rule maps the complete lattice
> `[0, T_max]^n` into itself monotonically, hence possesses a greatest fixed point.

## 5. How to test it in the simulation

1. **Estimate `b`.** Hold the shock fixed, set `eta = 0` (no revision), perturb the
   steady-state price (or target) of one sector by `+/- 5%`, run to the steady state,
   and measure `ds/dlog T`. `b` is the negative of that slope. Do this per sector and
   use the mean.
2. **Predict.** For each `m = k+1`, compute `g*(m)` and the predicted critical gain
   `eta*(m) = g*(m)/b`.
3. **Measure.** For a grid of `eta` and `k`, run the shock scenario with many seeds
   and record the oscillation amplitude (std of the detrended log price level) and
   the lag-1..4 autocorrelations. Declare "unstable" when the amplitude exceeds the
   `eta = 0` amplitude by at least 3 standard deviations and the lag-1
   autocorrelation of the detrended series exceeds 0.5 (pre-registered).
4. **Compare** the measured `eta*(m)` with the predicted one, and the measured
   period with `2(2m+1)`.

### Exact numerical verification of the proposition itself (no economics)
Iterate `u_{t+1} = u_t - g u_{t-m}` from a small perturbation for `m = 1..8` and a
grid of `g`; the empirical stability boundary must match `2 sin(pi/(2(2m+1)))` to
within the grid resolution. This belongs in `verification/` as a *verification* of
the mathematics, separate from the *validation* of the economics.

| m | g*(m) | period 2(2m+1) | m·g*(m) |
|---|---|---|---|
| 0 | 2.00000 | 2 | 0.000 |
| 1 | 1.00000 | 6 | 1.000 |
| 2 | 0.61803 | 10 | 1.236 |
| 3 | 0.44504 | 14 | 1.335 |
| 4 | 0.34730 | 18 | 1.389 |
| 5 | 0.28463 | 22 | 1.423 |
| 6 | 0.24107 | 26 | 1.446 |

The product `m·g*(m)` increases towards `pi/2 = 1.5708`. An earlier version of this
table paired `g*(0..5)` with the periods of `m = 1..6` and said the gain "halves
every period"; both were wrong and were caught by the independent verifier
(`verification/check1_theorem.py`). The paper (`paper/main.md` §4) always had the
correct version.

## 6. Statement hygiene for the paper

* Say "local linearisation", "small-gain", "delay-difference equation"; do not claim
  a global result.
* Report `k` (reporting lag) and `m = k+1` (effective delay) explicitly in every
  table, and label which one a figure's axis shows.
* The theorem is about a *linear* loop; the simulation is nonlinear. Agreement is a
  hypothesis test, disagreement must be reported as a finding, not buried.
