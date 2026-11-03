# Derivation notes: delay-induced instability of administratively revised signals

The notation here uses `m` for effective delay, corresponding to `k` in the
paper. If the realised signal is drawn from period `t-1-ell`, then
`m = ell + 1`.

## 1. Signed feedback

Let `u_t` be the deviation of a scalar tool from equilibrium. Suppose the
reported error can have either sign and locally equals
`s_t = -b u_t + O(u_t^2)` with `b > 0`. Linearising proportional
adjustment with coefficient `eta > 0` gives

    u[t+1] = u[t] - g u[t-m],    g = eta b.

The equilibrium must admit a two-sided derivative to interpret this as an
ordinary scalar linearisation.

## 2. Interpretation

The factor `g` combines the policy adjustment coefficient and the local
response of the signed error. For the scalar recurrence the proof below is
exact. For a nonlinear system this is a local prediction conditional on
the stated linearisation.

## 3. Proposition (stability boundary)

> **Proposition 1.** The zero solution of (★) is locally asymptotically stable if
> and only if `0 < g < g*(m)`, where
>
>     g*(m) = 2 sin( pi / (2(2m+1)) ).
>
> Numerically `g*(0) = 2`, `g*(1) = 1`, `g*(2) = 0.61803... = 1/phi`,
> `g*(3) = 0.44504...`, and `g*(m) ~ pi/(2m+1)` for large `m`. At `g = g*(m)` the
> system has a conjugate pair on the unit circle for `m >= 1`, at angles
> `+/- pi/(2m+1)`. A nonzero critical mode has period `2(2m+1)`.
> For `m = 0` the single critical root is -1 and the period is 2.

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

## 4. Relation to the simulator

The implemented tâtonnement reacts to nonnegative unmet demand. Its raw
administered price only rises or stays flat. Changes in the amplitude of
its detrended price are descriptive. They cannot locate the crossing of a
two-sided linear recurrence. Sector-wise price probes describe sensitivity
but do not determine the eigenvalues of the full dynamic system.

The short-side programme maximises total output under goods, capacity and
labour constraints. Enlarging the feasible set weakly raises the optimal
total. Individual entries of the optimal vector can fall. For example, with
labour coefficients (2,1), available labour 2 and bounds (1,0.5), the solution
is (0.75,0.5). Relaxing the second bound to 2 changes it to (0,2). This prevents
using the componentwise monotonicity argument for a greatest stationary plan.
The paper provides the full counterexample and states the corrected linear
description of feasible delivery.

## 5. Verification

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
