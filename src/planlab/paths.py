"""Filesystem helpers.

Every path in the project is derived from the package location so that no module
contains a hard-coded absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "project_root",
    "data_dir",
    "raw_dir",
    "processed_dir",
    "metadata_dir",
    "scenarios_dir",
    "results_dir",
    "figures_dir",
    "paper_dir",
    "cache_dir",
    "ensure_dir",
]


def project_root() -> Path:
    """Root of the checkout (the directory that contains ``pyproject.toml``)."""
    env = os.environ.get("PLANLAB_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _sub(name: str) -> Path:
    p = project_root() / name
    return p


def data_dir() -> Path:
    return _sub("data")


def raw_dir() -> Path:
    return _sub("data") / "raw"


def processed_dir() -> Path:
    return _sub("data") / "processed"


def metadata_dir() -> Path:
    return _sub("data") / "metadata"


def scenarios_dir() -> Path:
    return _sub("scenarios")


def results_dir() -> Path:
    return _sub("results")


def figures_dir() -> Path:
    return _sub("figures")


def paper_dir() -> Path:
    return _sub("paper")


def cache_dir() -> Path:
    base = os.environ.get("PLANLAB_CACHE")
    if base:
        return Path(base)
    return project_root() / ".tools" / "cache"


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
