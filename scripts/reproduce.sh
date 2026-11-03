#!/usr/bin/env bash
# Reproduction entry point: results/ + tables/ + figures/ from a clean checkout.
#
#   bash scripts/reproduce.sh              # clean re-run, default worker count
#   WORKERS=4 bash scripts/reproduce.sh    # fewer workers
#   CLEAN=0 bash scripts/reproduce.sh      # resume instead of starting from scratch
#   PAPER=1 bash scripts/reproduce.sh      # also build the PDF (needs the paper toolchain)
#   SKIP_TESTS=1 bash scripts/reproduce.sh
#
# Everything is seeded and deterministic; the only non-determinism is wall time,
# which is recorded in results/<block>/manifest.json.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/env.sh
source scripts/env.sh

WORKERS="${WORKERS:-}"
CLEAN="${CLEAN:-1}"
PAPER="${PAPER:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
START=$(date +%s)

echo "== PlanLab reproduction =============================================="
echo "  root      : $PLANLAB_ROOT"
echo "  python    : $PLANLAB_PY"
echo "  workers   : ${WORKERS:-auto (cpu_count-1)}"
echo "  clean     : $CLEAN"
echo "  started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ "$CLEAN" = "1" ]; then
  echo "-- cleaning results/ and figures/ (except .gitkeep)"
  find results -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
  find figures -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true
fi

echo "-- 1/4 simulation blocks"
if [ -n "$WORKERS" ]; then
  "$PLANLAB_PY" scripts/run_all.py --workers "$WORKERS"
else
  "$PLANLAB_PY" scripts/run_all.py
fi

echo "-- 2/4 tables and pre-registered tests"
"$PLANLAB_PY" -m planlab.experiments.aggregate

echo "-- 3/4 figures"
"$PLANLAB_PY" -m planlab.experiments.figures

if [ "$SKIP_TESTS" != "1" ]; then
  echo "-- 4/4 tests"
  "$PLANLAB_PY" -m pytest tests -q
else
  echo "-- 4/4 tests skipped (SKIP_TESTS=1)"
fi

if [ "$PAPER" = "1" ]; then
  echo "-- building the paper"
  "$PLANLAB_PY" scripts/paper_tables.py
  "$PLANLAB_PY" scripts/build_paper.py
  "$PLANLAB_PY" scripts/build_paper.py --lang zh
fi

END=$(date +%s)
echo "== done in $((END - START)) s ========================================="
echo "  results : $PLANLAB_ROOT/results"
echo "  tables  : $PLANLAB_ROOT/results/tables"
echo "  figures : $PLANLAB_ROOT/figures"
