"""Shared plumbing for the independent verification suite.

Nothing in here imports :mod:`planlab` except for path discovery: every check
implements its own arithmetic so that a bug in the project cannot validate
itself.  The only allowed dependency on the project is *data*: the technology
matrix (which is the object under test anyway) and the scenario YAML files.

Every check module exposes ``run() -> CheckResult``; ``verification/run_all.py``
collects them, prints a summary and writes ``verification/report.md``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCENARIOS = ROOT / "scenarios"
DATA = ROOT / "data" / "processed"
PAPER = ROOT / "paper"
TABLES = RESULTS / "tables"

sys.path.insert(0, str(ROOT / "src"))


@dataclass
class CheckResult:
    """Outcome of one verification check."""

    key: str
    title: str
    status: str = "PASS"          # PASS | FAIL | SKIP | ERROR
    summary: str = ""
    numbers: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)
        self.status = "FAIL"

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.fail(message)
        return bool(ok)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "numbers": self.numbers,
            "failures": self.failures,
            "notes": self.notes,
        }


def run_guarded(key: str, title: str, fn) -> CheckResult:
    """Run a check body, converting an unexpected exception into ERROR (not PASS)."""
    res = CheckResult(key=key, title=title)
    try:
        fn(res)
    except Exception as exc:  # noqa: BLE001 - a crash must never look like a pass
        res.status = "ERROR"
        res.fail(f"{type(exc).__name__}: {exc}")
        res.failures.append(traceback.format_exc()[-4000:])
    return res


# --------------------------------------------------------------------------
# result-file discovery (the experiments layer is built in parallel, so every
# check tolerates a missing/partial results/ directory and says so explicitly)
# --------------------------------------------------------------------------
def results_ready() -> bool:
    return RESULTS.exists() and any(RESULTS.iterdir())


def find_block_dirs() -> list[Path]:
    if not RESULTS.exists():
        return []
    out = []
    for p in sorted(RESULTS.iterdir()):
        if p.is_dir() and (p / "runs.parquet").exists():
            out.append(p)
    return out


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def env_python() -> str:
    return os.environ.get("PLANLAB_PY", sys.executable)
