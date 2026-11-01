"""Authoring tool for the PlanLab teaching notebooks.

The six ``*.ipynb`` files in this directory are the **deliverable**, and they are
committed *with their executed outputs* (see ``notebooks/README.md``).  Their
prose and code are kept in reviewable plain-Python form under ``nb_sources/`` so
that a diff of a teaching notebook is a diff of text rather than a diff of JSON.

Workflow
--------
::

    python notebooks/build_notebooks.py            # rewrite the .ipynb skeletons
    python notebooks/execute_all.py                # execute them in place
    python -m pytest tests/test_nb_notebooks.py -q # verify outputs are committed

``build_notebooks.py`` alone destroys stored outputs; ``--keep-outputs`` carries
the previous outputs over whenever a cell's source is unchanged, which is what
you want when only a markdown cell was edited.

Writing rule (enforced by ``tests/test_nb_notebooks.py``): every code cell must
produce output, so that a reader can tell whether they are reading a fresh
execution or a stale one.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "nb_sources"

#: notebook file name -> module under ``nb_sources/`` that defines ``CELLS``
NOTEBOOKS = [
    ("01_io_foundations.ipynb", "nb01_io_foundations"),
    ("02_delay_theorem.ipynb", "nb02_delay_theorem"),
    ("03_one_period_walkthrough.ipynb", "nb03_one_period_walkthrough"),
    ("04_mechanism_zoo.ipynb", "nb04_mechanism_zoo"),
    ("05_dual_track_and_shadow_prices.ipynb", "nb05_dual_track"),
    ("06_reproduce_a_paper_table.ipynb", "nb06_reproduce"),
]

KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def _load_source(module_name: str):
    path = SOURCE_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_nb_source_{module_name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load notebook source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _previous_outputs(path: Path) -> dict[str, list]:
    """Map code-cell source -> outputs from an already executed notebook."""
    if not path.exists():
        return {}
    try:
        old = nbformat.read(path, as_version=4)
    except Exception:  # pragma: no cover - a corrupt file is simply replaced
        return {}
    out: dict[str, list] = {}
    for cell in old.cells:
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            out[cell["source"]] = cell["outputs"]
    return out


def build_one(filename: str, module_name: str, *, keep_outputs: bool) -> Path:
    module = _load_source(module_name)
    cells = list(module.CELLS)
    title = getattr(module, "TITLE", filename)
    target = HERE / filename
    previous = _previous_outputs(target) if keep_outputs else {}

    nb = nbformat.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": KERNELSPEC,
            "language_info": {"name": "python", "version": "3.10.13",
                              "mimetype": "text/x-python", "file_extension": ".py"},
            "title": title,
            # the notebooks are generated from nb_sources/ so that their diffs are readable
            "planlab": {"source": f"notebooks/nb_sources/{module_name}.py",
                        "builder": "notebooks/build_notebooks.py"},
        }
    )
    for kind, text in cells:
        text = text.strip("\n")
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code":
            cell = nbformat.v4.new_code_cell(text)
            if text in previous:
                cell["outputs"] = previous[text]
            nb.cells.append(cell)
        else:  # pragma: no cover - authoring error
            raise ValueError(f"unknown cell kind {kind!r} in {module_name}")
    nbformat.validate(nb)
    nbformat.write(nb, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-outputs", action="store_true",
                        help="carry stored outputs over for unchanged code cells")
    parser.add_argument("--only", default=None,
                        help="build a single notebook by file name (e.g. 01_io_foundations.ipynb)")
    args = parser.parse_args(argv)

    built = 0
    missing = 0
    for filename, module_name in NOTEBOOKS:
        if args.only and filename != args.only:
            continue
        if not (SOURCE_DIR / f"{module_name}.py").exists():
            print(f"skip   {filename}  (no source: nb_sources/{module_name}.py)", file=sys.stderr)
            missing += 1
            continue
        path = build_one(filename, module_name, keep_outputs=args.keep_outputs)
        print(f"built {path.relative_to(HERE)}")
        built += 1
    if not built:
        print("nothing built", file=sys.stderr)
        return 1
    if missing and args.only:
        return 1
    print("\nNow run:  python notebooks/execute_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
