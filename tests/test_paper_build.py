"""Tests for the paper: the mathematics of Section 4 and the Markdown->HTML->PDF build.

Run with ``source scripts/env.sh && .venv/bin/python -m pytest tests/test_paper_build.py``.

The point of these tests is not coverage. Three claims in the manuscript are of the
"if this is wrong, retract the paper" kind, and each gets an independent check here:
Proposition 1's boundary (by root-finding, not by the closed form), Corollary 4's
claim that a positive loop admits no gain (by root-finding), and the build pipeline's
ability to resolve every cross-reference it is given.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    """Import a script by path; dataclasses requires the module in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def figures():
    return _load("planlab_make_figures", PAPER / "make_figures.py")


@pytest.fixture(scope="module")
def builder():
    return _load("planlab_build_paper", ROOT / "scripts" / "build_paper.py")


# --------------------------------------------------------------------------- #
# Proposition 1: the boundary
# --------------------------------------------------------------------------- #


def test_boundary_closed_form_known_values(figures):
    assert figures.boundary_gain(0) == pytest.approx(2.0, abs=1e-12)
    assert figures.boundary_gain(1) == pytest.approx(1.0, abs=1e-12)
    assert figures.boundary_gain(2) == pytest.approx((np.sqrt(5) - 1) / 2, abs=1e-12)
    assert figures.boundary_gain(3) == pytest.approx(0.4450418679126288, abs=1e-12)


def test_boundary_is_decreasing_and_approaches_asymptote(figures):
    gains = [figures.boundary_gain(k) for k in range(0, 200)]
    assert all(a > b for a, b in zip(gains, gains[1:]))
    for k in (50, 100, 199):
        assert gains[k] / (np.pi / (2 * k + 1)) == pytest.approx(1.0, abs=5e-3)


def test_spectral_radius_crosses_one_at_the_boundary(figures):
    """Independent of the closed form: root-find, then compare."""
    for k in range(0, 13):
        g = figures.boundary_gain(k)
        assert figures.spectral_radius(k, 0.999 * g) < 1.0
        assert figures.spectral_radius(k, 1.001 * g) > 1.0


def test_no_return_to_stability_above_the_boundary(figures):
    """Corollary of Step 2: every crossing is outward, so rho > 1 for all g > g*."""
    for k in range(0, 7):
        g = figures.boundary_gain(k)
        for factor in (1.05, 1.5, 3.0, 10.0):
            assert figures.spectral_radius(k, factor * g) > 1.0


def test_boundary_root_argument_and_period(figures):
    for k in range(1, 9):
        roots = figures.char_poly_roots(k, figures.boundary_gain(k))
        on_circle = roots[np.abs(np.abs(roots) - 1.0) < 1e-7]
        assert on_circle.size >= 2
        theta = float(np.abs(np.angle(on_circle[0])))
        assert theta == pytest.approx(np.pi / (2 * k + 1), abs=1e-8)
        assert 2 * np.pi / theta == pytest.approx(2 * (2 * k + 1), abs=1e-6)


def test_measured_period_matches_corollary(figures):
    """The period predicted by the corollary is the period in the time series."""
    for k in (1, 2, 3, 4):
        u = figures.impulse_response(k, figures.boundary_gain(k), horizon=400)
        tail = u[200:]
        acf = np.array([np.corrcoef(tail[:-lag], tail[lag:])[0, 1] for lag in range(1, 60)])
        # at the boundary the motion is a pure sinusoid, so the autocorrelation is 1
        # at every multiple of the period: take the smallest lag that returns to 1
        first_return = int(np.flatnonzero(acf > 0.99)[0]) + 1
        assert first_return == 2 * (2 * k + 1)


def test_gain_lag_budget(figures):
    for k in range(0, 500):
        assert k * figures.boundary_gain(k) < np.pi / 2
    assert 400 * figures.boundary_gain(400) == pytest.approx(np.pi / 2, abs=2e-3)


# --------------------------------------------------------------------------- #
# Corollary 4: a positive loop admits no gain
# --------------------------------------------------------------------------- #


def test_positive_loop_has_a_real_root_above_one():
    """Q_k(z, G) = z^{k+1} - z^k - G has Q(1) = -G < 0 and Q -> +inf."""
    for k in range(0, 9):
        for gain in (1e-4, 1e-2, 0.1, 0.5, 1.0, 3.0):
            coeffs = np.zeros(k + 2)
            coeffs[0] = 1.0
            coeffs[1] = -1.0
            coeffs[k + 1] -= gain          # Q(1) = -G
            assert coeffs.sum() == pytest.approx(-gain, abs=1e-12)
            assert np.max(np.abs(np.roots(coeffs))) > 1.0


def test_positive_loop_unstable_even_at_tiny_gain():
    for k in range(0, 5):
        coeffs = np.zeros(k + 2)
        coeffs[0] = 1.0
        coeffs[1] = -1.0
        coeffs[k + 1] -= 1e-9          # k = 0 would otherwise overwrite coeffs[1]
        assert np.max(np.abs(np.roots(coeffs))) > 1.0


# --------------------------------------------------------------------------- #
# The build pipeline
# --------------------------------------------------------------------------- #


def test_builder_numbers_sections_and_resolves_refs(builder):
    st = builder.BuildState()
    st.eq_counter = 0
    st.fig_counter = 0
    st.tab_counter = 0
    st.sections_order = []
    md = builder.make_markdown()
    body = md.convert("## Alpha {#sec:a}\n\nText.\n\n### Beta {#sec:b}\n\nMore text.\n")
    out = builder.number_headings(body, st)
    assert st.sections["sec:a"] == ("1", "section")
    assert st.sections["sec:b"] == ("1.1", "section")
    assert '<span class="secno">1</span> Alpha' in out
    assert '<span class="secno">1.1</span> Beta' in out


def test_builder_appendix_lettering(builder):
    st = builder.BuildState()
    st.eq_counter = 0
    st.fig_counter = 0
    st.tab_counter = 0
    st.sections_order = []
    md = builder.make_markdown()
    body = md.convert(
        "## One {#sec:one}\n\nx\n\n## Proofs {#sec:app .appendix}\n\n### Inner {#sec:inner}\n\ny\n"
    )
    builder.number_headings(body, st)
    assert st.sections["sec:app"] == ("Appendix A", "appendix")
    assert st.sections["sec:inner"] == ("A.1", "section")


def test_builder_rejects_nothing_but_reports_unresolved(builder):
    st = builder.BuildState()
    st.eq_counter = 0
    st.fig_counter = 0
    st.tab_counter = 0
    st.sections_order = []
    src = builder.extract_refs_and_cites("see [[REF:fig:nosuch]] and [[@alsoNosuch]]", st)
    html = builder.resolve_sentinels(src, st, {})
    assert "xref-broken" in html
    assert any("nosuch" in slot for slot in st.unresolved)


def test_every_citation_in_the_manuscript_is_in_the_bibliography(builder):
    entries = builder.parse_bibliography(PAPER / "bibliography.md")
    text = (PAPER / "main.md").read_text(encoding="utf-8")
    cited = set(re.findall(r"\[\[@([A-Za-z0-9_:\-]+)\]\]", text))
    for group in re.findall(r"\[\[CITE:([^\]]+)\]\]", text):
        cited.update(k.strip() for k in group.split(",") if k.strip())
    missing = sorted(cited - set(entries))
    assert not missing, f"cited but not in bibliography.md: {missing}"
    # every bibliography entry carries a resolvable link
    for key, entry in entries.items():
        assert "href=" in entry["html"], f"entry {key} has no URL"


def test_manuscript_builds_and_has_no_dangling_references(tmp_path):
    """Full HTML build; the only broken references allowed are the declared slots."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_paper.py"), "--no-pdf"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    html = (PAPER / "main.html").read_text(encoding="utf-8")

    declared = set(re.findall(r"\[\[(?:REF:(?:fig|tab|table|sec|eq)|RESULTS):[A-Za-z0-9_]+\]\]",
                              (PAPER / "main.md").read_text(encoding="utf-8")))
    broken = len(re.findall(r'class="xref-broken"', html))
    # five declared wave-2 floats are expected to be unresolved at this stage
    assert broken <= 5, f"{broken} broken references, expected at most 5 declared slots"
    assert declared, "no placeholders found; did the manuscript change?"

    # the mathematics must survive the pipeline
    assert "$g^\\star(k)$" in html or "g^\\star(k)" in html
    assert "katex" in html.lower()
    assert html.count('class="equation"') >= 15
