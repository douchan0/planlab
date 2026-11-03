# PlanLab

PlanLab compares six coordination rules in a shared input–output simulation. It uses a 20-sector network derived from Eurostat's 2020 EU-27 domestic table and records output, household delivery, shortage, welfare and inventory. The model fixes production recipes and uses simulated quarterly timing.

**Read the paper** · [中文论文](paper/main_zh.pdf) · [English paper](paper/main.pdf) · [中文审计报告](docs/项目审计报告.md)

The revised papers start with two-product examples and require only introductory calculus and linear algebra. The exact delay threshold applies to a signed scalar feedback rule. The administered price mechanism in the simulation raises prices in response to nonnegative shortage, so its detrended amplitude cannot identify that threshold. The papers distinguish the mathematical result from the mechanism comparisons and report paired shock and welfare results at their stated calibrations.

## Quick start

Run these commands from Bash.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,notebooks]"
source scripts/env.sh
python -m planlab.cli list-mechanisms
python -m planlab.cli run scenarios/baseline_mb_k0.yaml --trace
make test
```

The committed processed input matrices allow offline simulation. To rebuild the source data, use `make data`. The raw response is cached locally and its SHA-256 is recorded in [the data manifest](data/metadata/manifest.json). The source table is annual and measured at current basic prices. The simulation uses it as a fixed-price scale for model periods. [Data notes](docs/DATA_NOTES.md)

## Reproduce

```bash
python scripts/run_all.py --dry-run
CLEAN=0 WORKERS=6 bash scripts/reproduce.sh
python -m verification.run_all
make paper
```

- The grid contains 7,364 treatment runs across 11 blocks. B3 also runs no-shock controls with matched seeds. Each block commits its cell definitions, run summaries, sector panel, manifest and a sample of accounting records.
- Per-period `runs.parquet` and per-cell shards are generated locally. Verification lists panel checks that are skipped when those files are absent. Run summaries can still be compared against sampled re-runs.
- `make paper` builds both languages and regenerates the paper tables from the analysis outputs. The PDF builder uses headless Microsoft Edge, with `PLANLAB_EDGE` available to select an executable. The HTML build is available with `python scripts/build_paper.py --no-pdf` and the same command with `--lang zh`.
- [Experiment results](results/) · [Test log](results/tables/tests.csv) · [Hypothesis adjudication](results/tables/t00_hypotheses.csv) · [Verification report](verification/report.md)

## Read the model

| Component | Location |
| --- | --- |
| Period order, balance equation and feasibility | [Architecture](docs/ARCHITECTURE.md) |
| Original hypotheses and subsequent design changes | [Protocol](docs/PROTOCOL.md) · [Amendments](docs/AMENDMENTS.md) · [Deviations](docs/DEVIATIONS.md) |
| Mathematical derivation | [Delay theorem](docs/notes_delay_theorem.md) |
| Guided executable examples | [Teaching notebooks](notebooks/README.md) |
| Research history before the audit revision | [Process report](paper/process_report_zh.pdf) · [Reference run](reference_results/CROSSCHECK.md) |

## Licence and citation

[MIT licence](LICENSE) · [Citation metadata](CITATION.cff)
