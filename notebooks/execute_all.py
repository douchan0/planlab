"""Execute every teaching notebook headlessly and rewrite its outputs in place.

Usage
-----
::

    source scripts/env.sh
    python notebooks/execute_all.py                # all notebooks
    python notebooks/execute_all.py --only 02_delay_theorem.ipynb

The script executes each ``notebooks/*.ipynb`` with :class:`nbclient.NotebookClient`
against an **ephemeral kernelspec that points at the interpreter running this
script**, so the notebook always sees the same virtual environment — no ``PATH``
or ``jupyter kernelspec install`` gymnastics required, and nothing is written
outside the workspace.

On success the executed notebook is written back over the original file, which is
what makes the committed notebooks carry their outputs.  A machine-readable
record is written to ``notebooks/execution_status.json``; it stores the sha256 of
each notebook *after* execution, so ``tests/test_nb_notebooks.py`` can verify
that the pending outputs really belong to the committed file rather than to some
earlier revision.

Exit status is non-zero if any notebook fails, so this is usable in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS_PATH = HERE / "execution_status.json"
KERNEL_NAME = "planlab-venv"
TIMEOUT = 600


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_kernel() -> str:
    """Register a kernelspec for *this* interpreter, inside a temp directory."""
    tmp = Path(tempfile.mkdtemp(prefix="planlab-kernels-"))
    kdir = tmp / "kernels" / KERNEL_NAME
    kdir.mkdir(parents=True)
    (kdir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "PlanLab (project venv)",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    os.environ["JUPYTER_DATA_DIR"] = str(tmp)
    os.environ.setdefault("IPYTHONDIR", str(tmp / "ipython"))
    return str(tmp)


def _strip_outputs(nb) -> None:
    """Guarantee the committed outputs are exactly the ones produced here."""
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def execute_one(path: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient

    started = time.perf_counter()
    nb = nbformat.read(path, as_version=4)
    _strip_outputs(nb)
    record: dict = {"ok": False, "seconds": 0.0, "cells": len(nb.cells),
                    "kernel": KERNEL_NAME, "error": None}
    try:
        client = NotebookClient(
            nb,
            timeout=TIMEOUT,
            kernel_name=KERNEL_NAME,
            resources={"metadata": {"path": str(HERE)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.validate(nb)
        nbformat.write(nb, path)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - we want the message, not the type
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
        # write what we have so the failed cell is inspectable
        try:
            nbformat.write(nb, path)
        except Exception:  # pragma: no cover
            pass
    record["seconds"] = round(time.perf_counter() - started, 2)
    record["sha256"] = _sha256(path)
    record["n_code_cells"] = sum(1 for c in nb.cells if c.get("cell_type") == "code")
    record["n_outputs"] = sum(len(c.get("outputs", [])) for c in nb.cells
                              if c.get("cell_type") == "code")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", default=None,
                        help="execute only these notebook file names (repeatable)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    _install_kernel()
    notebooks = sorted(HERE.glob("*.ipynb"))
    if args.only:
        wanted = set(args.only)
        notebooks = [p for p in notebooks if p.name in wanted]
    if not notebooks:
        print("no notebooks found (build them first: python notebooks/build_notebooks.py)",
              file=sys.stderr)
        return 1

    status: dict = {
        "generated_by": "notebooks/execute_all.py",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "nbclient_timeout_s": TIMEOUT,
        "notebooks": {},
    }
    failures = 0
    total = 0.0
    for path in notebooks:
        if not args.quiet:
            print(f"executing {path.name} ...", end="", flush=True)
        record = execute_one(path)
        status["notebooks"][path.name] = record
        total += record["seconds"]
        if record["ok"]:
            if not args.quiet:
                print(f" ok  {record['seconds']:6.2f}s  "
                      f"({record['n_code_cells']} code cells, {record['n_outputs']} outputs)")
        else:
            failures += 1
            print(f" FAILED  {record['seconds']:6.2f}s\n  {record['error']}", file=sys.stderr)

    status["total_seconds"] = round(total, 2)
    status["failures"] = failures
    STATUS_PATH.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"\n{len(notebooks) - failures}/{len(notebooks)} notebooks executed in "
              f"{total:.1f}s -> {STATUS_PATH.relative_to(HERE.parent)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
