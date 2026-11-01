#!/usr/bin/env bash
# Source this file before working in the PlanLab repo:  source scripts/env.sh
# It keeps every tool's cache inside the workspace (the sandbox cannot write to $HOME).
export PLANLAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PLANLAB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PLANLAB_ROOT/.tools/mpl"
export XDG_CACHE_HOME="$PLANLAB_ROOT/.tools/cache"
export PLANLAB_PY="$PLANLAB_ROOT/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export PATH="$PLANLAB_ROOT/.venv/bin:$PATH"
