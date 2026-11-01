#!/usr/bin/env python
"""Thin wrapper around the data-layer build CLI.

Kept deliberately dumb: the actual work lives in :mod:`planlab.data.build`, so
``scripts/fetch_data.py`` and ``python -m planlab.data.build`` always behave
identically.

Usage::

    source scripts/env.sh
    python scripts/fetch_data.py                 # build both substrates
    python scripts/fetch_data.py --source eurostat
    python scripts/fetch_data.py --force         # re-download and rebuild
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from planlab.data.build import main  # noqa: E402  (needs the path shim above)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
