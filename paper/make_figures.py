#!/usr/bin/env python3
"""Generate the analytic figures and tables of the PlanLab methodology paper.

Everything produced here is *mathematics*, not simulation output: it depends only
on Proposition 1 (the delay-stability boundary) and its corollary, and is
therefore reproducible before any experiment has been run.

    source scripts/env.sh && .venv/bin/python paper/make_figures.py

Outputs (all under ``paper/``)::

    figures/stability_boundary.png   Figure 1
    figures/delay_response.png       Figure 2
    tables/stability_boundary.csv    Table 1  (+ .cap caption)
    tables/*.json                    the numbers quoted in the text
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
FIG = PAPER / "figures"
TAB = PAPER / "tables"

# --------------------------------------------------------------------------- #
# The analysis, implemented once and reused everywhere
# --------------------------------------------------------------------------- #


def boundary_gain(k: int) -> float:
    """Critical gain g*(k) = 2 sin(pi / (2(2k+1)))."""
    return 2.0 * np.sin(np.pi / (2.0 * (2 * k + 1)))


def boundary_angle(k: int) -> float:
    """Argument theta_0 = pi/(2k+1) of the unit-circle root at g = g*(k)."""
    return np.pi / (2 * k + 1)


def boundary_period(k: int) -> float:
    """Oscillation period in quarters at the boundary: 2 pi / theta_0 = 2(2k+1)."""
    return 2.0 * np.pi / boundary_angle(k)


def char_poly_roots(k: int, g: float) -> np.ndarray:
    """Roots of z^{k+1} - z^k + g."""
    coeffs = np.zeros(k + 2)
    coeffs[0] = 1.0            # z^{k+1}
    coeffs[1] = -1.0           # z^k        (for k = 0 this is the constant term)
    coeffs[k + 1] += g         # + g z^0
    return np.roots(coeffs)


def spectral_radius(k: int, g: float) -> float:
    return float(np.max(np.abs(char_poly_roots(k, g))))


def impulse_response(k: int, g: float, horizon: int = 120) -> np.ndarray:
    """u_{t+1} = u_t - g u_{t-k} from u_{-k..0} = (1, 0, ..., 0)."""
    u = np.zeros(horizon + k + 1)
    u[k] = 1.0
    for t in range(k, horizon + k):
        u[t + 1] = u[t] - g * u[t - k]
    return u[k:k + horizon]


# --------------------------------------------------------------------------- #
# Figure 1 — the boundary and the spectral-radius landscape
# --------------------------------------------------------------------------- #


def figure_boundary() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Charter", "Iowan Old Style", "Palatino", "Georgia", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.55), constrained_layout=True)

    # ---- (a) g*(k) and the asymptotic slope -------------------------------
    ks = np.arange(0, 13)
    gs = boundary_gain(ks)
    kk = np.linspace(0.4, 12.6, 400)

    ax1.plot(kk, np.pi / (2 * kk + 1), ls=(0, (4, 2)), lw=1.0, color="#8a8a8a",
             label=r"asymptote $\pi/(2k+1)$")
    ax1.plot(ks[1:], gs[1:], "-o", ms=3.4, lw=1.25, color="#7a1f2b",
             label=r"$g^\star(k)=2\sin\!\left[\pi/(2(2k+1))\right]$")
    ax1.plot(ks[:1], gs[:1], "o", ms=4.6, mfc="white", mew=1.4, color="#7a1f2b")

    ph = (np.sqrt(5) - 1) / 2
    ax1.annotate(r"$g^\star(2)=1/\varphi\approx0.618$", xy=(2, gs[2]),
                 xytext=(3.4, 1.15), fontsize=7.6, color="#7a1f2b",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#7a1f2b",
                                 shrinkA=0, shrinkB=3))
    ax1.annotate(r"$g^\star(0)=2$", xy=(0, 2.0), xytext=(0.35, 1.72), fontsize=7.6,
                 color="#7a1f2b")
    ax1.set_xlabel(r"information lag $k$ (quarters)")
    ax1.set_ylabel(r"critical gain $g^\star(k)$")
    ax1.set_xlim(-0.3, 12.6)
    ax1.set_ylim(0, 2.18)
    ax1.legend(loc="upper right", fontsize=7.4, handlelength=2.0)
    ax1.text(0.985, 0.04, "(a)", transform=ax1.transAxes, fontsize=9, fontweight="bold", ha="right")

    # ---- (b) spectral radius over the (k, g) plane -------------------------
    k_grid = np.arange(0, 9)
    g_grid = np.linspace(0.02, 2.2, 320)
    R = np.array([[spectral_radius(int(k), float(g)) for g in g_grid] for k in k_grid])
    R = np.clip(R, 0.0, 1.6)

    mesh = ax2.pcolormesh(g_grid, k_grid, R, cmap="RdYlBu_r", vmin=0.0, vmax=1.6,
                          shading="nearest", rasterized=True)
    ax2.plot(gs[:9], k_grid, "-", lw=1.4, color="#141d2b")
    ax2.plot(gs[:9], k_grid, "o", ms=2.2, color="white", mew=0)
    cb = fig.colorbar(mesh, ax=ax2, pad=0.02, fraction=0.06)
    cb.set_label(r"spectral radius $\rho(g,k)$", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)
    ax2.set_xlabel(r"proportional gain $g$")
    ax2.set_ylabel(r"lag $k$")
    ax2.set_yticks(k_grid)
    ax2.set_ylim(-0.5, 8.5)
    ax2.text(0.03, 0.06, "stable\n" + r"$\rho<1$", transform=ax2.transAxes, fontsize=7.4,
             va="bottom", color="#12305c")
    ax2.text(0.97, 0.95, "unstable " + r"$\rho>1$", transform=ax2.transAxes, fontsize=7.4,
             color="#4a0d0d", ha="right", va="top")
    ax2.text(0.985, 0.04, "(b)", transform=ax2.transAxes, fontsize=9, fontweight="bold", ha="right")

    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "stability_boundary.png", bbox_inches="tight")
    import matplotlib.pyplot as plt2

    plt2.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — impulse responses below, at, and above the boundary
# --------------------------------------------------------------------------- #


def figure_response() -> None:
    """(a) damped impulse responses; (b) the sharpness of the spectral crossing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Charter", "Iowan Old Style", "Palatino", "Georgia", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "legend.frameon": False,
        "savefig.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)

    # ---- (a) damped responses: the period is the observable ----------------
    horizon = 108
    colours = {1: "#1f4e79", 2: "#b3760a", 4: "#7a1f2b"}
    for k in (1, 2, 4):
        u = impulse_response(k, 0.90 * boundary_gain(k), horizon)
        ax1.plot(np.arange(horizon), u / np.max(np.abs(u)), lw=1.1, color=colours[k],
                 label=rf"$k={k}$  (period {2 * (2 * k + 1)} quarters)")
    ax1.axhline(0.0, lw=0.6, color="#999999")
    ax1.set_xlabel("quarter $t$")
    ax1.set_ylabel(r"$u_t / \max_t |u_t|$")
    ax1.set_xlim(0, horizon - 1)
    ax1.set_title(r"$g = 0.90\,g^\star(k)$: damped oscillation", fontsize=8.2, pad=3)
    ax1.legend(loc="upper right", fontsize=6.9, handlelength=1.6)
    ax1.text(0.985, 0.04, "(a)", transform=ax1.transAxes, fontsize=9, fontweight="bold", ha="right")

    # ---- (b) the crossing is smooth in g/g* but steepens with k ------------
    ratios = np.linspace(0.80, 1.30, 220)
    for k in (1, 2, 4, 8):
        rho = np.array([spectral_radius(k, float(r) * boundary_gain(k)) for r in ratios])
        ax2.plot(ratios, rho - 1.0, lw=1.1, color=colours.get(k, "#2f6b3f"))
        ax2.annotate(rf"$k={k}$", xy=(1.30, rho[-1] - 1.0), xytext=(2, -1),
                     textcoords="offset points", fontsize=6.9,
                     color=colours.get(k, "#2f6b3f"), va="center")
    ax2.axhline(0.0, lw=0.9, color="#141d2b")
    ax2.axvline(1.0, lw=0.7, ls=(0, (3, 2)), color="#8a8a8a")
    ax2.text(1.006, 0.055, r"$g=g^\star(k)$", fontsize=7.0, color="#555555")
    ax2.set_xlabel(r"gain relative to the boundary, $g/g^\star(k)$")
    ax2.set_ylabel(r"$\rho(g,k) - 1$")
    ax2.set_xlim(0.80, 1.36)
    ax2.set_title("spectral radius crosses one at the boundary", fontsize=8.2, pad=3)
    ax2.text(0.985, 0.04, "(b)", transform=ax2.transAxes, fontsize=9, fontweight="bold", ha="right")

    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "delay_response.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Table 1 — the boundary, its angle, its period, its asymptotic approximation
# --------------------------------------------------------------------------- #


def table_boundary() -> dict:
    rows = []
    for k in range(0, 13):
        g = boundary_gain(k)
        th = boundary_angle(k)
        rows.append({
            "k": k,
            "g_star": g,
            "theta0_over_pi": th / np.pi,
            "period_quarters": round(boundary_period(k), 6),
            "asymptote_pi_over_2k1": np.pi / (2 * k + 1),
            "ratio_to_asymptote": g / (np.pi / (2 * k + 1)),
        })

    TAB.mkdir(exist_ok=True)
    header = "k, g*(k), theta_0/pi, period (quarters), pi/(2k+1), ratio"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['k']}, {r['g_star']:.6f}, {r['theta0_over_pi']:.6f}, "
            f"{r['period_quarters']:.0f}, {r['asymptote_pi_over_2k1']:.6f}, "
            f"{r['ratio_to_asymptote']:.4f}"
        )
    (TAB / "stability_boundary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (TAB / "stability_boundary.cap").write_text(
        r"Local stability boundary of the administrative price-adjustment rule "
        r"$u_{t+1}=u_t-g\,u_{t-k}$ (Proposition 1). For each information lag $k$ the "
        r"critical gain is $g^\star(k)=2\sin[\pi/(2(2k+1))]$, attained at a conjugate pair "
        r"of roots on the unit circle with argument $\theta_0=\pi/(2k+1)$, so the "
        r"oscillation at the boundary has period $2(2k+1)$ quarters (Corollary 1). The "
        r"penultimate column gives the large-$k$ asymptote $\pi/(2k+1)$ and the last "
        r"column its ratio to the exact value.",
        encoding="utf-8",
    )
    return {"rows": rows}


# --------------------------------------------------------------------------- #


def main() -> int:
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    table = table_boundary()
    figure_boundary()
    figure_response()

    numbers = {
        "g_star": {str(k): boundary_gain(k) for k in range(0, 13)},
        "period": {str(k): boundary_period(k) for k in range(0, 13)},
        "rho_at_boundary_check": {
            str(k): {
                "below": spectral_radius(k, 0.999 * boundary_gain(k)),
                "above": spectral_radius(k, 1.001 * boundary_gain(k)),
            }
            for k in range(0, 9)
        },
        "g_star_2_exact": (np.sqrt(5) - 1) / 2,
        "asymptote_ratio_k12": table["rows"][12]["ratio_to_asymptote"],
        "half_gain_lag": {
            str(k): float(_half_gain_lag(k)) for k in (0, 1, 2, 4, 6)
        },
    }
    (TAB / "analytic_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n",
                                               encoding="utf-8")
    print("wrote figures/stability_boundary.png, figures/delay_response.png")
    print("wrote tables/stability_boundary.csv, tables/analytic_numbers.json")
    print(f"g*(2) = {numbers['g_star_2_exact']:.10f}")
    print(f"rho at 0.999 g* and 1.001 g*: "
          f"{numbers['rho_at_boundary_check']['4']}")
    return 0


def _half_gain_lag(k: int) -> float:
    """Lag at which the boundary gain halves the k=0 value: illustrative scaling."""
    return boundary_gain(k) / 2.0


if __name__ == "__main__":
    raise SystemExit(main())
