"""Command-line build entry point: ``python -m planlab.data.build``.

Examples
--------
Build everything from the network (idempotent — a second run reuses the cached
raw payload and the committed processed artefacts)::

    python -m planlab.data.build --source eurostat

Rebuild the synthetic substrate from the committed Eurostat aggregate::

    python -m planlab.data.build --source synthetic

Force a fresh download and full rebuild::

    python -m planlab.data.build --source both --force

The command is safe to re-run: it verifies the cached raw payload's sha256 and
only rewrites artefacts whose inputs changed (``--force`` overrides).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .eurostat import DEFAULT_GEO, DEFAULT_STK_FLOW, DEFAULT_UNIT, EurostatError
from .loader import DataError, build_all, build_eurostat, build_synthetic, write_manifest

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the build CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m planlab.data.build",
        description="Fetch and process the PlanLab input-output substrates.",
    )
    parser.add_argument(
        "--source",
        choices=("eurostat", "synthetic", "both"),
        default="both",
        help="which substrate to build (default: both)",
    )
    parser.add_argument("--year", type=int, default=2020, help="reference year (default: 2020)")
    parser.add_argument("--n-sectors", type=int, default=20, help="target sectors (default: 20)")
    parser.add_argument("--geo", default=DEFAULT_GEO, help=f"Eurostat geo (default: {DEFAULT_GEO})")
    parser.add_argument("--unit", default=DEFAULT_UNIT, help=f"Eurostat unit (default: {DEFAULT_UNIT})")
    parser.add_argument(
        "--stk-flow",
        default=DEFAULT_STK_FLOW,
        choices=("DOM", "TOTAL"),
        help="DOM = domestic uses (default), TOTAL = domestic + imported",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download the raw payload and rewrite all artefacts",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print errors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the build CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on a data-layer error, 2 on a
        network failure.
    """
    args = build_parser().parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print
    try:
        if args.source == "both":
            summary = build_all(
                year=args.year, n_sectors=args.n_sectors, force=args.force
            )
            euro, syn = summary["eurostat"], summary["synthetic"]
            manifest = summary["manifest"]
        elif args.source == "eurostat":
            euro = build_eurostat(
                year=args.year,
                n_sectors=args.n_sectors,
                geo=args.geo,
                unit=args.unit,
                stk_flow=args.stk_flow,
                force=args.force,
            )
            manifest = write_manifest(year=args.year, n_sectors=args.n_sectors)
            syn = None
        else:
            syn = build_synthetic(
                year=args.year, n_sectors=args.n_sectors, force=args.force
            )
            manifest = write_manifest(year=args.year, n_sectors=args.n_sectors)
            euro = None
    except EurostatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if euro is not None:
        meta = euro["meta"]
        say(
            f"eurostat: {euro['key']}  source sectors={meta['provenance']['n_source_sectors']}  "
            f"sectors={meta['n_sectors']}  gross output={sum(meta['gross_output']):.1f} MIO EUR"
        )
        say(
            f"          A column sums in [{meta['checks']['a_column_sums_min']:.4f}, "
            f"{meta['checks']['a_column_sums_max']:.4f}] (all < 1: "
            f"{meta['checks']['all_column_sums_below_one']})"
        )
        say(
            f"          identity max |A x + y - x| = "
            f"{meta['checks']['identity_max_abs_residual_n65']:.4g} MIO EUR"
        )
    if syn is not None:
        meta = syn["meta"]
        say(
            f"synthetic: {syn['key']}  seed={meta['synthetic']['seed']}  "
            f"gross output={sum(meta['gross_output']):.1f} MIO EUR"
        )
    say(f"manifest: data/metadata/manifest.json  substrates={sorted(manifest['substrates'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
