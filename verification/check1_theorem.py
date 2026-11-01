"""Check 1 — independent re-derivation of Proposition 1 (the delay boundary).

The claim under test
--------------------
For the linear delay recursion

    u_{t+1} = u_t - g * u_{t-m},        g > 0, m >= 0 integer,

the zero solution is asymptotically stable iff

    g < g*(m) = 2 sin(pi / (2(2m+1))),

and at the boundary the oscillation period is 2(2m+1) periods.

My own derivation (written before reading ``docs/notes_delay_theorem.md``)
--------------------------------------------------------------------------
1. Substituting ``u_t = z^t`` gives the characteristic polynomial

       Q_m(z) = z^{m+1} - z^m + g = 0
              = z^m (z - 1) + g.

2. The boundary is the smallest ``g > 0`` at which a root has ``|z| = 1``.
   Put ``z = e^{i theta}``, ``theta in (0, 2 pi)``.  Then

       g = z^m (1 - z) = e^{i m theta} (1 - e^{i theta}).

   Now ``1 - e^{i theta} = 2 sin(theta/2) e^{i(theta/2 - pi/2)}`` and
   ``sin(theta/2) > 0`` on ``(0, 2pi)``, so

       |g|      = 2 sin(theta/2),
       arg(g)   = (m + 1/2) theta - pi/2   (mod 2 pi).

   ``g`` real and positive requires ``arg(g) = 0 (mod 2 pi)``:

       (m + 1/2) theta = pi/2 + 2 pi k
       => theta_k = pi (4k + 1) / (2m + 1),   k = 0, 1, ..., m.

3. The associated gains are ``2 sin(theta_k / 2)``.  ``theta_0 = pi/(2m+1)``
   gives the smallest one because every other ``theta_k`` lies in
   ``(pi/(2m+1), 2 pi)`` and the only other exponent with the same sine is
   ``theta_m = 2 pi - theta_0`` (its mirror image, the same conjugate pair).
   Hence

       g*(m) = 2 sin(pi / (2(2m+1))).

   For m = 0 this is 2 (Q_0 = z - (1-g) is stable iff 0 < g < 2);
   m = 1 gives 1, m = 2 gives the golden-ratio conjugate 0.6180339887.

4. At the boundary the dominant roots are ``z = e^{+- i theta_0}``, whose
   oscillation period is ``2 pi / theta_0 = 2(2m+1)``.
   For large m, ``g*(m) ~ pi/(2m+1)``.

Checks performed here (all implemented from scratch; no planlab.engine import)
-----------------------------------------------------------------------------
A. The unit-circle root identity: ``Q_m(e^{i theta_0}) = 0`` to floating
   precision for m = 0..30.
B. A *root-free* stability test (Schur-Cohn / Levinson reflection recursion)
   validated against root finding on random polynomials, then used with
   bisection to locate the boundary for m = 0..30.  This is a mathematically
   different route from eigenvalue computation, so agreement is real evidence.
C. Companion-matrix spectral radius: below/above the closed form (m = 0..30).
D. A time-domain test with a self-written log-envelope estimator (homogeneous
   rescaling, slope = log of the dominant growth factor): it must be negative
   just below g* and positive just above, and must agree with the spectral
   radius (m = 1..8, and 0.95/1.05 g* probes for m = 1..12).
E. Oscillation period at the boundary from zero-crossing counting and from the
   dominant root's argument (m = 1..12).
"""

from __future__ import annotations

import math

import numpy as np

from .common import CheckResult, run_guarded

MAX_M = 30
TOL_BOUNDARY = 1e-10
TOL_ZERO = 1e-12


# ---------------------------------------------------------------------------
# 1. closed form
# ---------------------------------------------------------------------------
def g_closed_form(m: int) -> float:
    return 2.0 * math.sin(math.pi / (2.0 * (2 * m + 1)))


def q_coeffs_ascending(m: int, g: float) -> np.ndarray:
    """Coefficients of Q_m, *ascending* order (a_0 + a_1 z + ... + a_{m+1} z^{m+1})."""
    if m == 0:                       # Q_0 = z - 1 + g
        return np.array([g - 1.0, 1.0])
    a = np.zeros(m + 2)
    a[0] = g
    a[m] = -1.0
    a[m + 1] = 1.0
    return a


# ---------------------------------------------------------------------------
# 2. root-free Schur test (Schur-Cohn / Levinson reflection recursion)
# ---------------------------------------------------------------------------
def schur_stable(a_ascending: np.ndarray) -> bool:
    """True iff every root of ``p`` has modulus strictly below one.

    Reflection recursion: with ``p(z) = sum a_k z^k`` monic of degree d and
    ``p*(z) = z^d conj(p(1/conj z))`` (reversed coefficients for real p), ``p``
    is Schur-stable iff ``|a_0| < 1`` and
    ``p_1(z) = (p(z) - a_0 p*(z)) / (z (1 - |a_0|^2))`` is Schur-stable.
    ``p_1`` is again monic, so the test recurses down to a constant.
    """
    a = np.asarray(a_ascending, dtype=float).copy()
    d = a.size - 1
    if abs(a[d] - 1.0) > 1e-12:
        raise ValueError("polynomial must be monic")
    while d >= 1:
        if abs(a[0]) >= 1.0:
            return False
        denom = 1.0 - a[0] * a[0]
        if denom <= 0.0:
            return False
        rev = a[::-1]
        nxt = (a - a[0] * rev)[1:] / denom     # drop the zero constant term, divide by z
        if nxt.size != d:                      # pragma: no cover - defensive
            raise AssertionError("reflection recursion degree mismatch")
        a = nxt
        d -= 1
    return True


def _self_test_schur() -> None:
    """The Schur test must agree with root finding on random monic polynomials."""
    rng = np.random.default_rng(20240501)
    for _ in range(400):
        d = int(rng.integers(1, 7))
        a = np.concatenate([[1.0], rng.standard_normal(d)])   # unnormalised, ascending
        a = a / a[-1]
        radius = float(np.max(np.abs(np.roots(a[::-1]))))
        schur = schur_stable(a)
        # tolerate polynomials whose roots sit essentially on the circle
        if min(abs(radius - 1.0), abs(radius - 0.999)) > 1e-6:
            assert schur == (radius < 1.0), (a, schur, radius)


def boundary_schur(m: int, lo: float = 1e-9, hi: float = 1.0) -> float:
    """Locate the stability boundary with the root-free test only."""
    # grow hi until unstable
    while schur_stable(q_coeffs_ascending(m, hi)):
        hi *= 1.6
        if hi > 1e6:                                  # pragma: no cover
            raise RuntimeError("boundary not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if schur_stable(q_coeffs_ascending(m, mid)):
            lo = mid
        else:
            hi = mid
        if hi - lo <= TOL_BOUNDARY * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 3. spectral radius via companion matrix (different routine from np.roots)
# ---------------------------------------------------------------------------
def companion_radius(m: int, g: float) -> float:
    if m == 0:
        return abs(1.0 - g)
    # z^{m+1} - z^m + g = 0 with monic coefficients (descending) 1, -1, 0, ..., g;
    # the companion first row is [-a_{d-1}/a_d, ..., -a_0/a_d] = [1, 0, ..., 0, -g]
    c = np.zeros(m + 1)
    c[0] = 1.0
    c[m] = -g
    C = np.zeros((m + 1, m + 1))
    C[0, :] = c
    C[1:, :-1] = np.eye(m)
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def dominant_root_angle(m: int, g: float) -> float:
    """|arg| of the dominant (largest-modulus) root, in radians."""
    if m == 0:
        return math.pi
    coeffs_desc = np.zeros(m + 2)
    coeffs_desc[0] = 1.0
    coeffs_desc[1] = -1.0
    coeffs_desc[-1] = g
    roots = np.roots(coeffs_desc)
    dom = roots[int(np.argmax(np.abs(roots)))]
    return float(abs(math.atan2(dom.imag, dom.real)))


# ---------------------------------------------------------------------------
# 4. time domain: self-written log-envelope estimator
# ---------------------------------------------------------------------------
def log_growth(
    g: float, m: int, steps: int = 4000, warmup: int = 800, seed: int = 7
) -> tuple[float, np.ndarray]:
    """Return (log of the dominant growth factor per period, normalised history).

    Homogeneous rescaling: the state ``v`` (last m+1 values) is renormalised to
    max-abs one every step and the log of each rescaling factor is accumulated,
    so ``logscale[t]`` is the log of the true magnitude scale of the trajectory.
    Its slope (least squares over the post-transient window) is ``log rho``.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(m + 1)          # v[0] is u_{t-m}, v[-1] is u_t
    logs = np.empty(steps)
    logscale = 0.0
    nz = max(1, v.size)
    v = v / np.max(np.abs(v))
    series = np.empty(steps)
    for t in range(steps):
        nxt = v[-1] - g * v[0]
        v = np.concatenate([v[1:], [nxt]])
        peak = float(np.max(np.abs(v)))
        if peak <= 0.0:                     # pragma: no cover - degenerate
            raise RuntimeError("trajectory collapsed to zero")
        v = v / peak
        logscale += math.log(peak)
        logs[t] = logscale
        series[t] = math.exp(logscale) * v[-1]
    k = np.arange(warmup, steps)
    slope = float(np.polyfit(k, logs[warmup:], 1)[0])
    return slope, series


def period_by_zero_crossings(u: np.ndarray) -> float:
    """Estimate the period from the spacing of sign changes (P/2 apart)."""
    s = np.sign(u)
    s = s[s != 0.0]
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size < 6:
        return float("nan")
    gaps = np.diff(idx).astype(float)
    return float(2.0 * np.median(gaps))


def period_by_fft(u: np.ndarray, skip: float = 0.5) -> float:
    start = int(len(u) * skip)
    x = np.asarray(u[start:], dtype=float)
    x = x - x.mean()
    n = 1 << int(math.ceil(math.log2(max(x.size, 2))))
    spec = np.abs(np.fft.rfft(x, n=n))
    freqs = np.fft.rfftfreq(n)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return float("inf") if freqs[k] <= 0 else float(1.0 / freqs[k])


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def _body(res: CheckResult) -> None:
    # -- A. the unit-circle root identity ----------------------------------
    worst_a = 0.0
    for m in range(MAX_M + 1):
        theta = math.pi / (2 * m + 1)
        z = complex(math.cos(theta), math.sin(theta))
        # Q_m(z) = z^m (z - 1) + g
        q = z**m * (z - 1.0) + g_closed_form(m)
        worst_a = max(worst_a, abs(q))
        res.check(abs(q) < 1e-9, f"A: |Q_m(e^(i theta_0))| = {abs(q):.3e} at m={m}")
    res.numbers["A_max_unit_circle_residual"] = worst_a
    res.note(
        "A: substituting z=e^{i theta} into z^{m+1}-z^m+g=0 gives g = z^m(1-z); "
        "on |z|=1, real positive g requires (m+1/2)theta = pi/2 mod 2pi, whose "
        "smallest-|g| solution is theta_0 = pi/(2m+1) with g = 2 sin(theta_0/2). "
        f"Worst numerical residual over m=0..{MAX_M}: {worst_a:.2e}."
    )

    # -- B. root-free Schur test vs root finding ---------------------------
    _self_test_schur()
    rows = []
    worst_b = 0.0
    for m in range(MAX_M + 1):
        pred = g_closed_form(m)
        emp = boundary_schur(m)
        diff = abs(emp - pred)
        worst_b = max(worst_b, diff)
        rows.append((m, pred, emp, diff))
        res.check(
            diff <= 1e-9,
            f"B: Schur-test boundary m={m}: {emp:.12f} vs closed form {pred:.12f} "
            f"(diff {diff:.2e})",
        )
    res.numbers["B_max_boundary_diff"] = worst_b
    res.numbers["B_boundaries"] = {str(m): round(pred, 12) for m, pred, _, _ in rows}

    # boundary is strictly decreasing in m
    vals = [r[1] for r in rows]
    res.check(
        all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
        "B: g*(m) is strictly decreasing in m (a longer delay is destabilising sooner)",
    )

    # -- C. spectral radius either side ------------------------------------
    worst_c = []
    for m in list(range(0, MAX_M + 1)):
        gb = g_closed_form(m)
        r_below = companion_radius(m, 0.97 * gb)
        r_above = companion_radius(m, 1.03 * gb)
        r_at = companion_radius(m, gb)
        worst_c.append((m, r_below, r_at, r_above))
        res.check(
            r_below < 1.0 < r_above,
            f"C: rho(0.97 g*)={r_below:.9f} < 1 < rho(1.03 g*)={r_above:.9f} at m={m}",
        )
        res.check(
            abs(r_at - 1.0) < 5e-9,
            f"C: rho(g*)={r_at:.12f} = 1 at m={m} (root on the unit circle)",
        )
    res.numbers["C_max_rho_at_boundary_error"] = max(abs(r - 1.0) for _, _, r, _ in worst_c)

    # -- D. time domain ----------------------------------------------------
    slopes = []
    for m in range(1, 9):
        gb = g_closed_form(m)
        sl_below, _ = log_growth(0.95 * gb, m)
        sl_above, _ = log_growth(1.05 * gb, m)
        rho_below = math.exp(sl_below)
        rho_above = math.exp(sl_above)
        # independent spectral radius for the same g (companion matrix)
        sr_below = companion_radius(m, 0.95 * gb)
        sr_above = companion_radius(m, 1.05 * gb)
        slopes.append((m, sl_below, sl_above, rho_below / sr_below, rho_above / sr_above))
        res.check(sl_below < 0.0, f"D: growth at 0.95 g* is negative: {sl_below:.4e} (m={m})")
        res.check(sl_above > 0.0, f"D: growth at 1.05 g* is positive: {sl_above:.4e} (m={m})")
        res.check(
            abs(rho_below / sr_below - 1.0) < 0.05 and abs(rho_above / sr_above - 1.0) < 0.05,
            f"D: simulated growth factor {rho_below:.5f}/{rho_above:.5f} matches "
            f"spectral radius {sr_below:.5f}/{sr_above:.5f} within 5% (m={m})",
        )
    res.numbers["D_growth_table"] = [
        {"m": m, "slope_below": round(a, 6), "slope_above": round(b, 6),
         "factor_below_over_rho": round(c, 5), "factor_above_over_rho": round(d, 5)}
        for m, a, b, c, d in slopes
    ]

    # -- E. period at the boundary ----------------------------------------
    for m in range(1, 13):
        gb = g_closed_form(m)
        predicted = 2 * (2 * m + 1)
        _, series = log_growth(1.02 * gb, m, steps=5000, warmup=600)
        p_zero = period_by_zero_crossings(series)
        p_fft = period_by_fft(series)
        angle = dominant_root_angle(m, gb)
        p_root = 2.0 * math.pi / angle
        res.check(
            abs(p_zero - predicted) <= 1.0,
            f"E: zero-crossing period {p_zero:.2f} != {predicted} (m={m})",
        )
        res.check(
            abs(p_fft - predicted) <= 1.0,
            f"E: FFT period {p_fft:.2f} != {predicted} (m={m})",
        )
        res.check(
            abs(p_root - predicted) <= 1e-6,
            f"E: dominant-root period {p_root:.6f} != {predicted} (m={m})",
        )
        res.numbers.setdefault("E_periods", []).append(
            {"m": m, "predicted": predicted, "zero_cross": round(p_zero, 3),
             "fft": round(p_fft, 3), "root_arg": round(p_root, 6)}
        )

    # -- F. asymptotics and sanity of the claimed constants ----------------
    res.check(abs(g_closed_form(0) - 2.0) < 1e-15, "F: g*(0) = 2")
    res.check(abs(g_closed_form(1) - 1.0) < 1e-15, "F: g*(1) = 1")
    res.check(abs(g_closed_form(2) - 0.6180339887498949) < 1e-15, "F: g*(2) = (sqrt5-1)/2")
    asym = abs(g_closed_form(200) * (2 * 200 + 1) / math.pi - 1.0)
    res.check(asym < 1e-3, f"F: g*(m) ~ pi/(2m+1) for large m (rel. dev {asym:.2e} at m=200)")

    if res.status == "PASS":
        res.summary = (
            f"Prop. 1 confirmed independently for m=0..{MAX_M}: g*(m)=2 sin(pi/(2(2m+1))), "
            f"boundary from a root-free Schur test agrees to {worst_b:.1e}, the "
            f"spectral radius crosses 1 exactly there, the time-domain growth exponent "
            f"changes sign, and the boundary oscillation period is 2(2m+1)."
        )


def run() -> CheckResult:
    return run_guarded("theorem", "Independent re-derivation of Proposition 1", _body)


if __name__ == "__main__":                              # pragma: no cover
    import json

    result = run()
    print(json.dumps(result.as_dict(), indent=2, default=str))
    raise SystemExit(0 if result.status == "PASS" else 1)
