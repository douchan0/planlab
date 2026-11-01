# PlanLab — reproduction targets.
#
#   make data         rebuild the input-output substrates under data/processed
#   make test         run the test suite
#   make experiments  run the pre-registered blocks B1-B8 (+ the two probes)
#   make tables       rebuild results/tables from results/
#   make figures      rebuild figures/ from results/
#   make paper        rebuild paper/main.pdf
#   make reproduce    everything, from a clean results/ and figures/
#   make check        the full definition-of-done: tests + experiments + tables + figures
#
# Every target sources scripts/env.sh first (PYTHONPATH, MPLCONFIGDIR, the venv).

SHELL := /bin/bash
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
RUN = cd "$(ROOT)" && source scripts/env.sh &&

.PHONY: help data test experiments tables figures paper reproduce check clean clean-results clean-figures

help:
	@sed -n '3,12p' Makefile

data:
	@$(RUN) python -m planlab.data.build

test:
	@$(RUN) python -m pytest tests -q

experiments:
	@$(RUN) python scripts/run_all.py

tables:
	@$(RUN) python -m planlab.experiments.aggregate

figures:
	@$(RUN) python -m planlab.experiments.figures

paper:
	@$(RUN) python scripts/build_paper.py

reproduce:
	@bash scripts/reproduce.sh

check: test experiments tables figures
	@echo "== all targets complete =="

clean-results:
	@find results -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true

clean-figures:
	@find figures -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true

clean: clean-results clean-figures
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
