"""PlanLab experiments: the pre-registered grid, the Monte Carlo runner, the tables
and the publication figures.

The layer is split so that each stage is independently testable and re-runnable:

``grid``
    Pure data.  Builds every cell of blocks B1–B6 (plus the exploratory ladder
    block and the two gain probes) from the frozen design, with the common random
    numbers shared across mechanisms.  No I/O, no simulation.

``runner``
    Executes cells with multiprocessing, resumably, and writes
    ``results/<block>/{runs.parquet, summary.csv, aggregate.csv, sector_panel.parquet,
    manifest.json}``.  Skips mechanisms that are not registered yet, with a
    warning; re-running picks them up.

``aggregate``
    Turns the results into the paper's tables (``results/tables/*.csv`` and
    ``*.tex``) and into the machine-readable test statistics
    (``results/tables/tests.csv``): paired Wilcoxon with bootstrap CIs,
    Kruskal–Wallis with Holm correction, the lag regressions with mechanism fixed
    effects and cluster-robust SEs, and the empirical stability boundary against
    ``2 sin(pi/(2(2m+1)))/b``.

``figures``
    Every figure of the paper, generated from the results files, recorded in
    ``results/figures_index.csv``.

Typical use::

    from planlab.experiments import grid, runner
    cells = grid.build_cells("B1")
    runner.run_block("B1")
"""

from __future__ import annotations

from .grid import (  # noqa: F401
    BLOCKS,
    BLOCK_ORDER,
    EXPLORATORY_BLOCKS,
    GAINS,
    LAGS,
    MECHANISMS,
    Cell,
    block_run_count,
    build_cells,
    build_grid,
    planned_run_count,
)
from .runner import BlockResult, run_block, run_cells  # noqa: F401

__all__ = [
    "BLOCKS",
    "BLOCK_ORDER",
    "EXPLORATORY_BLOCKS",
    "GAINS",
    "LAGS",
    "MECHANISMS",
    "Cell",
    "BlockResult",
    "block_run_count",
    "build_cells",
    "build_grid",
    "planned_run_count",
    "run_block",
    "run_cells",
]
