"""Structural tests for the teaching notebook suite.

These tests do **not** execute the notebooks (that is ``notebooks/execute_all.py``'s
job, and it takes about a minute).  They verify that what is *committed* is what a
reader will open:

* the six notebooks exist, validate under nbformat, and carry a kernelspec;
* no code cell is empty and every code cell produced output — so a stored notebook
  is a fresh execution rather than a skeleton someone forgot to run;
* no cell failed and no output is an error traceback;
* ``notebooks/execution_status.json`` records a successful run of *this exact
  revision* of each notebook (its sha256), which is what stops a stale output from
  being passed off as a current one;
* the committed output stays a sensible size;
* the bilingual convention holds (Chinese prose, then an italic English summary).

Run with:  .venv/bin/python -m pytest tests/test_nb_notebooks.py -q
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nbformat
import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
STATUS_PATH = NB_DIR / "execution_status.json"

EXPECTED = [
    "01_io_foundations.ipynb",
    "02_delay_theorem.ipynb",
    "03_one_period_walkthrough.ipynb",
    "04_mechanism_zoo.ipynb",
    "05_dual_track_and_shadow_prices.ipynb",
    "06_reproduce_a_paper_table.ipynb",
]

#: total committed size of the executed notebooks (they embed base64 PNGs)
MAX_TOTAL_BYTES = 3_000_000
#: per-notebook ceiling, so one runaway figure cannot blow up the repository
MAX_NOTEBOOK_BYTES = 800_000

CJK = re.compile(r"[\u4e00-\u9fff]")
ITALIC = re.compile(r"(?<![\w*])\*[^*\n]{10,}\*(?![\w*])")


def _paths() -> list[Path]:
    return sorted(NB_DIR.glob("*.ipynb"))


def _read(path: Path):
    return nbformat.read(path, as_version=4)


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def _markdown_cells(nb):
    return [c for c in nb.cells if c.cell_type == "markdown"]


def _status() -> dict:
    assert STATUS_PATH.exists(), (
        "notebooks/execution_status.json is missing: run "
        "`.venv/bin/python notebooks/execute_all.py`"
    )
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------
def test_all_six_notebooks_are_committed():
    names = [p.name for p in _paths()]
    assert names == EXPECTED, f"expected {EXPECTED}, found {names}"


def test_sources_and_builder_are_present():
    for name in EXPECTED:
        assert (NB_DIR / "nb_sources" / (Path(name).stem + ".py")).exists(), name
    for name in ("build_notebooks.py", "execute_all.py", "README.md"):
        assert (NB_DIR / name).exists(), name


# ---------------------------------------------------------------------------
# structure (parameterised over the committed notebooks)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", EXPECTED)
def test_nbformat_validates(name):
    nb = _read(NB_DIR / name)
    nbformat.validate(nb)
    assert nb.nbformat == 4


@pytest.mark.parametrize("name", EXPECTED)
def test_kernelspec_and_source_metadata(name):
    nb = _read(NB_DIR / name)
    assert nb.metadata["kernelspec"]["name"] == "python3"
    source = nb.metadata.get("planlab", {}).get("source", "")
    assert source.endswith(".py"), f"{name} does not record its authoring source"
    assert (NB_DIR.parent / source).exists(), (
        f"{name} points at {source!r}, which does not exist"
    )


@pytest.mark.parametrize("name", EXPECTED)
def test_no_empty_cells(name):
    nb = _read(NB_DIR / name)
    for index, cell in enumerate(nb.cells):
        assert cell.source.strip(), f"{name}: cell {index} is empty"
    assert len(_code_cells(nb)) >= 5, f"{name}: suspiciously few code cells"


@pytest.mark.parametrize("name", EXPECTED)
def test_every_code_cell_has_output(name):
    """A cell that prints nothing is a cell a reader cannot verify.

    The authoring rule (see build_notebooks.py) is that every code cell ends by
    printing something, which makes "has this notebook been executed?" visible at a
    glance.  This test enforces it.
    """
    nb = _read(NB_DIR / name)
    silent = [i for i, c in enumerate(_code_cells(nb)) if not c.get("outputs")]
    assert not silent, f"{name}: code cells {silent} produced no output"


@pytest.mark.parametrize("name", EXPECTED)
def test_no_error_or_stderr_output(name):
    nb = _read(NB_DIR / name)
    for index, cell in enumerate(_code_cells(nb)):
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", (
                f"{name}: code cell {index} raised {out.get('ename')}"
            )
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = out.get("text", "")
                # warnings in committed output look like an unfinished notebook
                assert "Warning" not in text and "Traceback" not in text, (
                    f"{name}: code cell {index} wrote to stderr: {text[:200]}"
                )


@pytest.mark.parametrize("name", EXPECTED)
def test_bilingual_markdown(name):
    """Chinese prose first, then an italic English summary (the teaching convention)."""
    nb = _read(NB_DIR / name)
    md = _markdown_cells(nb)
    assert md, f"{name}: no markdown cells"
    without_cjk = [i for i, c in enumerate(md) if not CJK.search(c.source)]
    assert not without_cjk, f"{name}: markdown cells {without_cjk} have no Chinese prose"
    assert any(ITALIC.search(c.source) for c in md), (
        f"{name}: no italic English summary found in any markdown cell"
    )


@pytest.mark.parametrize("name", EXPECTED)
def test_committed_output_is_not_huge(name):
    size = (NB_DIR / name).stat().st_size
    assert size < MAX_NOTEBOOK_BYTES, f"{name} is {size/1024:.0f} kB, over the ceiling"


def test_total_output_size_is_modest():
    total = sum(p.stat().st_size for p in _paths())
    assert total < MAX_TOTAL_BYTES, (
        f"the executed notebooks total {total/1e6:.2f} MB; keep the repository light"
    )


# ---------------------------------------------------------------------------
# the status report: are the stored outputs the ones for *these* files?
# ---------------------------------------------------------------------------
def test_status_report_covers_every_notebook():
    status = _status()
    recorded = status["notebooks"]
    for name in EXPECTED:
        assert name in recorded, f"{name} is not in execution_status.json"
        assert recorded[name]["ok"] is True, f"{name} failed: {recorded[name].get('error')}"


def test_status_hashes_match_the_committed_notebooks():
    """The guard against stale outputs: the recorded sha256 must be of this file."""
    status = _status()
    for name in EXPECTED:
        entry = status["notebooks"][name]
        actual = _sha256(NB_DIR / name)
        assert entry["sha256"] == actual, (
            f"{name} was modified after it was executed "
            f"(recorded {entry['sha256'][:12]}, actual {actual[:12]}); "
            "re-run `.venv/bin/python notebooks/execute_all.py`"
        )


def test_each_notebook_runs_in_under_a_minute():
    status = _status()
    slow = {name: e["seconds"] for name, e in status["notebooks"].items()
            if name in EXPECTED and e["seconds"] > 60}
    assert not slow, f"notebooks over the ~60 s budget: {slow}"


# ---------------------------------------------------------------------------
# teaching-specific invariants
# ---------------------------------------------------------------------------
def test_delay_theorem_notebook_does_not_import_the_verification_script():
    """Notebook 02 must rebuild the theorem, not call the script that checks it."""
    nb = _read(NB_DIR / "02_delay_theorem.ipynb")
    code = "\n".join(c.source for c in _code_cells(nb))
    assert "verify_theorem" not in code.replace("verification/verify_theorem.py", ""), (
        "notebook 02 imports or calls verification/verify_theorem.py; it must re-derive it"
    )
    assert "import planlab" not in code, "notebook 02 should need no PlanLab import at all"


def test_mechanism_notebook_uses_the_registry():
    """Notebook 04 must enumerate the registry, so it survives mechanisms landing."""
    nb = _read(NB_DIR / "04_mechanism_zoo.ipynb")
    code = "\n".join(c.source for c in _code_cells(nb))
    assert "available_mechanisms()" in code


def test_walkthrough_notebook_asserts_the_balance_identity():
    """Notebook 03 is the auditable one: it must contain explicit balance checks."""
    nb = _read(NB_DIR / "03_one_period_walkthrough.ipynb")
    code = "\n".join(c.source for c in _code_cells(nb))
    for token in ("reconcile_by_hand", "S_end", "ledger"):
        assert token in code, f"notebook 03 lost its {token!r} audit machinery"
    text = "\n".join(
        out.get("text", "")
        for c in _code_cells(nb)
        for out in c.get("outputs", [])
        if out.get("output_type") == "stream"
    )
    assert "ALL" in text and "CHECKS PASSED" in text, (
        "notebook 03 did not report a passing audit ledger in its stored output"
    )
