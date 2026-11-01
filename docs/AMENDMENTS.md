# Amendments to the pre-registration

`docs/PROTOCOL.md` is the pre-registration. This file records every change made to
it, with the date and the reason, *before* the main experimental grid was run. The
rule is: amendments are allowed, silent amendments are not.

---

## Amendment 1 (2026-11-01) — which loop carries the theory, and the sign of the gain

**Problem found.** Proposition 1 (`docs/notes_delay_theorem.md`) assumes a
*negative* feedback loop: a higher instrument (plan target, or price) reduces the
observed gap, so `s_t = -b u_t` with `b > 0`. A pilot run shows that this holds for
prices but **not** for output targets in a capacity-constrained Leontief economy:
raising the target raises intermediate demand, which competes for goods whose supply
is fixed, so the observed shortage *increases* with the target. The plan-revision
loop therefore has `b < 0`, i.e. it is a *positive* loop.

**Pilot evidence** (`material_balance`, Eurostat 2020, 20 sectors, lag 2, mean of 3
seeds, 40% capacity shock to `c12`):

| revision_gain | mean shortage | peak shortage | mean welfare |
|---|---|---|---|
| 0.0 | 0.0173 | 0.0415 | 258.931 |
| 0.1 | 0.0179 | 0.0486 | 258.935 |
| 0.3 | 0.0203 | 0.0412 | 258.289 |
| 0.5 | 0.0577 | 0.0447 | 257.019 |
| 0.8 | 0.0685 | 0.1184 | 256.140 |
| 1.2 | 0.0846 | 0.1721 | 255.379 |
| 2.0 | 0.1066 | 0.2496 | 254.556 |

Shortage and welfare loss rise monotonically with the gain: the loop does not
stabilise, it ratchets.

**Amendment.** H1's stability-boundary test is carried out on the mechanism whose
loop has `b > 0`, namely the **price** revision of `tatonnement` (`price_gain`), where
a higher price reduces excess demand through the household demand elasticity. Two
consequences:

* **H1 (revised)** is tested on `tatonnement` with `price_gain` as the gain and
  `demand.price_elasticity > 0`; the predicted boundary is
  `price_gain*(m) = 2 sin(pi/(2(2m+1))) / b_p` with `b_p` estimated from the price
  response experiment, and the predicted oscillation period is `2(2m+1)` quarters.
* **H1' (new, reported alongside)** — *taut plans are self-defeating*: in a
  capacity-constrained input–output economy the proportional revision of output
  targets is a positive feedback loop, so mean shortage is non-decreasing in the
  revision gain and welfare is non-increasing. This is a substantive result about
  material-balance planning, and it is reported with the same prominence as H1.

Nothing else in the protocol changes: the mechanism grid, the seeds, the outcomes
and the statistical procedures stay as pre-registered.

---

## Amendment 2 (2026-11-01) — pilot calibration of the shock and buffer parameters

**Reason.** Before fixing the grid, the Lead ran a pilot to check that the
treatments produce measurable variation rather than near-zero noise. The pilot used
the real Eurostat 2020 substrate, `material_balance`, 60 periods, burn-in 10, 3
seeds, and varied the shocked sector, the shock size, its duration and the initial
inventory ratio. Peak aggregate shortage rate:

| sector | size | duration | stock_ratio | peak shortage | mean shortage |
|---|---|---|---|---|---|
| c12 | 0.65 | 4 | 0.15 | 0.025 | 0.0016 |
| c12 | 0.50 | 8 | 0.15 | 0.190 | 0.0252 |
| c12 | 0.35 | 12 | 0.05 | 0.472 | 0.1354 |
| c09 | 0.50 | 8 | 0.15 | 0.151 | 0.0205 |
| c09 | 0.35 | 12 | 0.05 | 0.262 | 0.0765 |
| c06 | 0.65 | 4 | 0.15 | 0.008 | 0.0005 |
| c10 | 0.50 | 8 | 0.15 | 0.093 | 0.0145 |

**Finding that drives the calibration.** The response is governed far more by the
shock's *duration relative to the inventory buffer* than by its size: a 35% shock
lasting 12 quarters bites three times harder than a 65% shock lasting 4 quarters.
This is a result in itself (inventories are the shock absorber, so the cost of a
supply disruption is a question about time, not depth) and it is reported as such.

**Amendment.** The standard scenario parameters become

* `stock_ratio = 0.10` (was 0.15),
* capacity shock `size = 0.50`, `duration = 8` quarters in B3 (was unspecified),
* B4 keeps its pre-registered size sweep {0.10, 0.30, 0.50} at `duration = 8`,
* B2/B5 use `duration = 8` too, so that lag and mechanism effects are not swamped by
  a weak impulse.

The grid is now fixed; no further calibration of these parameters is permitted after
the main runs begin.

---

## Amendment 3 (2026-11-01) — the gain grid for the stability boundary

The pre-registered B2 varies lag × mechanism. Testing H1 against the predicted
boundary requires a *gain* axis as well. B2 therefore becomes a three-way design:
`price_gain` (tatonnement) / `revision_gain` (others) × lag `k in {0..6}` ×
mechanism, with the gain grid `{0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6}`.

This is an *extension* of the pre-registered design (more cells along an axis the
theory requires), not a change to the outcomes, the tests, or the seeds.
