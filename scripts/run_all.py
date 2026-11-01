#!/usr/bin/env python
"""Run the PlanLab experiment blocks in order, resumably.

    source scripts/env.sh
    python scripts/run_all.py                  # every block, cpu_count-1 workers
    python scripts/run_all.py -b B1 B1b B2     # a subset, in the order given
    python scripts/run_all.py --workers 4 --verbose
    python scripts/run_all.py --force          # ignore existing shards

Every block writes ``results/<block>/`` (see ``planlab.experiments.runner``) and a
``results/manifest.json`` is written at the end with the wall time, the run counts
and the versions of everything that can move a number.  A block whose mechanism is
not registered yet is skipped with a warning and reported in its manifest, so the
same command can be re-run after new mechanisms land: completed cells are skipped,
missing ones run.

Exit status is 0 only if every planned cell of every requested block completed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
if str(_THIS.parents[1] / "src") not in sys.path:  # allow running without env.sh
    sys.path.insert(0, str(_THIS.parents[1] / "src"))

from planlab.experiments import grid, runner  # noqa: E402


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_all.py",
        description="Run (or resume) the pre-registered PlanLab experiment blocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-b",
        "--blocks",
        nargs="*",
        default=None,
        metavar="BLOCK",
        help=f"blocks to run, in order (default: all of {', '.join(grid.BLOCK_ORDER)})",
    )
    p.add_argument("-w", "--workers", type=int, default=None,
                   help="worker processes (default: cpu_count - 1)")
    p.add_argument("--force", action="store_true", help="re-run cells even if a shard exists")
    p.add_argument("--results-dir", default=None, help="override results/ (tests)")
    p.add_argument("--quiet", action="store_true", help="only print the final summary")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    blocks = list(args.blocks) if args.blocks else list(grid.BLOCK_ORDER)
    unknown = [b for b in blocks if b not in grid.BLOCK_ORDER]
    if unknown:
        print(f"unknown block(s): {', '.join(unknown)}; known: {', '.join(grid.BLOCK_ORDER)}")
        return 2

    planned = grid.planned_run_count(blocks)
    present = grid.available_mechanisms_safe()
    print("PlanLab experiment run")
    print(f"  blocks        : {', '.join(blocks)}")
    print(f"  mechanisms    : {', '.join(present) if present else 'NONE REGISTERED'}")
    print(f"  planned runs  : {planned['total']}  " + "  ".join(f"{b}={planned[b]}" for b in blocks))
    print(f"  grid hash     : {grid.grid_fingerprint()}")
    if args.dry_run:
        for block in blocks:
            cells = grid.build_cells(block)
            runnable, skipped = grid.split_runnable(cells, present)
            print(
                f"  {block:4s} cells={len(cells):4d} runnable={len(runnable):4d} "
                f"seeds/cell={sorted({len(c.seeds) for c in cells})} skipped={len(skipped)}"
            )
        return 0

    started = time.perf_counter()
    summaries: list[dict] = []
    for block in blocks:
        summaries.append(
            runner.run_block(
                block,
                workers=args.workers,
                force=args.force,
                results_dir=args.results_dir,
                verbose=not args.quiet,
            ).as_dict()
        )
    wall = time.perf_counter() - started

    root = runner.results_root(args.results_dir)
    index = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "blocks_requested": blocks,
        "blocks_completed": [s["block"] for s in summaries if s["cells_failed"] == 0],
        "total_wall_time_s": round(wall, 2),
        "planned_runs": planned["total"],
        "runs_completed": int(sum(s["runs_completed"] for s in summaries)),
        "sum_cell_wall_time_s": round(
            float(
                sum(
                    (json.loads((Path(s["manifest"]).read_text(encoding="utf-8"))).get(
                        "sum_cell_wall_time_s", 0.0
                    )
                     if Path(s["manifest"]).exists()
                     else 0.0)
                    for s in summaries
                )
            ),
            2,
        ),
        "cells_failed": int(sum(s["cells_failed"] for s in summaries)),
        "mechanisms_present": present,
        "grid_fingerprint": grid.grid_fingerprint(),
        "code_fingerprint": runner.code_fingerprint(),
        **runner.git_state(),
        "package_versions": runner.package_versions(),
        "environment": dict(runner.BLOCK_ENVIRONMENT),
        "blocks": summaries,
        "how_to_reproduce": "bash scripts/reproduce.sh   # or: make reproduce",
    }
    (root / "manifest.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    print("\nblock   cells   runs   failed   skipped(missing mech)   wall_s")
    for s in summaries:
        print(
            f"{s['block']:5s} {s['cells_planned']:6d} {s['runs_completed']:6d} "
            f"{s['cells_failed']:7d} {s['cells_skipped_existing']:6d}"
            f"                    {s['wall_s']:7.1f}"
        )
    print(f"\ntotal wall time: {wall:.1f}s   runs: {index['runs_completed']}   "
          f"failures: {index['cells_failed']}")
    print(f"manifest: {root / 'manifest.json'}")
    return 0 if index["cells_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
