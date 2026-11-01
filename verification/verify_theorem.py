"""Verification of Proposition 1 (the delay-instability boundary).

This script verifies the *mathematics*, not the economics: it checks that

    u_{t+1} = u_t - g * u_{t-m}

loses stability exactly at ``g*(m) = 2 sin(pi / (2(2m+1)))``, with oscillation
period ``2(2m+1)`` at the boundary.

Three independent checks are performed:

1. **Exact (spectral).**  The characteristic polynomial is
   ``Q_m(z) = z^{m+1} - z^m + g``; the system is asymptotically stable iff the
   spectral radius is below one.  The boundary is found by bisection on
   ``max|z| - 1`` and compared with the closed form.
2. **Time domain.**  Iterate the recursion and estimate the growth exponent of the
   envelope by a log-linear fit of its local maxima; it must be negative just below
   the boundary and positive just above it.
3. **Period.**  Just above the boundary, the dominant spectral peak of the simulated
   series must be at period ``2(2m+1)`` quarters.

Exit status is non-zero if any check fails, so this can be used in CI.

Run:  .venv/bin/python verification/verify_theorem.py
"""

from __future__ import annotations

import sys

import numpy as np

TOL_BOUNDARY = 1e-9     # spectral check: agreement with closed form
TOL_PERIOD = 1.0        # periods: |measured - predicted| <= 1 quarter


def closed_form(m: int) -> float:
    """``g*(m) = 2 sin(pi/(2(2m+1)))``; equals 2 at m=0 (the no-delay case)."""
    return 2.0 * np.sin(np.pi / (2.0 * (2 * m + 1)))


def char_poly_coefficients(g: float, m: int) -> np.ndarray:
    """Coefficients (highest power first) of ``z^{m+1} - z^m + g``."""
    if m == 0:
        # z^1 - z^0 + g = z - (1 - g)
        return np.array([1.0, g - 1.0])
    coeffs = np.zeros(m + 2)
    coeffs[0] = 1.0
    coeffs[1] = -1.0
    coeffs[-1] = g
    return coeffs


def spectral_radius(g: float, m: int) -> float:
    return float(np.max(np.abs(np.roots(char_poly_coefficients(g, m)))))


def boundary_by_roots(m: int, hi: float = 5.0, tol: float = 1e-13) -> float:
    """Bisection on the spectral radius crossing one."""
    lo = 1e-12
    while spectral_radius(hi, m) <= 1.0 and hi < 1e4:
        hi *= 1.5
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid, m) > 1.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def simulate(g: float, m: int, periods: int = 6000, u0: float = 1.0) -> np.ndarray:
    u = np.zeros(periods)
    u[0] = u0
    for t in range(periods - 1):
        lag = u[t - m] if t - m >= 0 else 0.0
        u[t + 1] = u[t] - g * lag
    return u


def envelope_growth_rate(u: np.ndarray, n_peaks: int = 25) -> float:
    """Log-slope of the envelope over the first ``n_peaks`` local maxima of |u|.

    Using the first peaks (rather than the tail) keeps the estimate in a range where
    neither underflow nor overflow distorts it.
    """
    seg = np.abs(u)
    if seg.size < 10:
        return float("nan")
    interior = seg[1:-1]
    is_peak = (interior > seg[:-2]) & (interior >= seg[2:])
    peaks = interior[is_peak]
    peaks = peaks[peaks > 0.0][:n_peaks]
    if peaks.size < 5:
        return float("nan")
    k = np.arange(peaks.size)
    return float(np.polyfit(k, np.log(peaks), 1)[0])


def dominant_period(u: np.ndarray, window: int = 81) -> float:
    """Period of the strongest spectral peak after removing a moving average."""
    x = u - np.convolve(np.pad(u, (window // 2, window // 2), mode="edge"),
                        np.ones(window) / window, mode="valid")[: u.size]
    x = x[len(x) // 4:]
    n = 1 << int(np.ceil(np.log2(max(x.size, 2))))
    spec = np.abs(np.fft.rfft(x, n=n))
    freqs = np.fft.rfftfreq(n)
    spec[0] = 0.0
    idx = int(np.argmax(spec))
    return float("inf") if freqs[idx] <= 0 else float(1.0 / freqs[idx])


def main() -> int:
    failures: list[str] = []

    print("=" * 78)
    print("Check 1 (exact): spectral radius of Q_m(z) = z^{m+1} - z^m + g")
    print("=" * 78)
    print(f"{'m':>3} {'2sin(pi/(2(2m+1)))':>20} {'roots':>20} {'|diff|':>10} {'max|z| at g*':>13}")
    for m in range(0, 15):
        pred = closed_form(m)
        emp = boundary_by_roots(m)
        diff = abs(emp - pred)
        rad = spectral_radius(pred, m)
        flag = "" if diff <= TOL_BOUNDARY else "  <-- FAIL"
        if diff > TOL_BOUNDARY:
            failures.append(f"spectral boundary m={m}: {emp:.12f} != {pred:.12f}")
        print(f"{m:3d} {pred:20.12f} {emp:20.12f} {diff:10.2e} {rad:13.9f}{flag}")

    print()
    print("=" * 78)
    print("Check 2 (time domain): envelope growth must change sign at g*")
    print("=" * 78)
    print(f"{'m':>3} {'g*':>10} {'growth @0.9g*':>15} {'growth @1.1g*':>15}")
    for m in range(1, 7):
        gb = closed_form(m)
        below = envelope_growth_rate(simulate(0.9 * gb, m, periods=600))
        above = envelope_growth_rate(simulate(1.1 * gb, m, periods=600))
        ok = below < 0.0 < above
        if not ok:
            failures.append(f"time domain m={m}: growth {below:.3e} / {above:.3e}")
        print(f"{m:3d} {gb:10.6f} {below:15.3e} {above:15.3e}{'' if ok else '  <-- FAIL'}")

    print()
    print("=" * 78)
    print("Check 3: oscillation period at g = 1.02 g* must be 2(2m+1) quarters")
    print("=" * 78)
    print(f"{'m':>3} {'predicted':>10} {'measured':>10} {'|diff|':>8}")
    for m in range(1, 8):
        gb = closed_form(m)
        u = simulate(1.02 * gb, m, periods=8000)
        measured = dominant_period(u)
        predicted = 2 * (2 * m + 1)
        diff = abs(measured - predicted)
        ok = diff <= TOL_PERIOD
        if not ok:
            failures.append(f"period m={m}: {measured:.2f} != {predicted}")
        print(f"{m:3d} {predicted:10d} {measured:10.2f} {diff:8.2f}{'' if ok else '  <-- FAIL'}")

    print()
    if failures:
        print(f"VERIFICATION FAILED ({len(failures)} checks):")
        for f in failures:
            print("  -", f)
        return 1
    print("VERIFICATION PASSED: the closed form, the sign change and the period all agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
