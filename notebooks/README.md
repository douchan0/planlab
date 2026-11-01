# PlanLab teaching notebooks

Six executed notebooks that rebuild PlanLab from the data up, so that a graduate
student can reconstruct the model, break it, and see why the theory holds — using
nothing but this directory and the repository it sits in.

**These notebooks are a deliverable in their own right.** Every one of them is
committed *with the outputs of a real execution*, executes end to end in well under
a minute, needs no network, and ends with a ledger of assertions that either pass
or fail the notebook. A stored notebook whose numbers do not close is a bug report,
not a draft.

| # | notebook | what it establishes | runtime |
|---|---|---|---|
| 01 | [01_io_foundations.ipynb](01_io_foundations.ipynb) | the substrate: accounting identity, productivity of `A`, the Leontief inverse, and how much a sectoral capacity shock is amplified | ~3 s |
| 02 | [02_delay_theorem.ipynb](02_delay_theorem.ipynb) | the delay theorem from scratch: characteristic polynomial, unit-circle proof, numerical boundary, oscillation period | ~2 s |
| 03 | [03_one_period_walkthrough.ipynb](03_one_period_walkthrough.ipynb) | one period of the engine opened up and re-derived by hand: claims, tiers, the feasibility programme, rationing, and the balance identity | ~3 s |
| 04 | [04_mechanism_zoo.ipynb](04_mechanism_zoo.ipynb) | every registered mechanism on one identical shock, with the common-random-numbers argument made testable | ~4 s |
| 05 | [05_dual_track_and_shadow_prices.ipynb](05_dual_track_and_shadow_prices.ipynb) | the central LP's dual variables as shadow prices, and the welfare–volatility frontier of the dual track | ~3 s |
| 06 | [06_reproduce_a_paper_table.ipynb](06_reproduce_a_paper_table.ipynb) | reproducibility: recompute a paper table from the rawest records and re-render a paper figure, asserting equality | ~4 s |

(Runtimes are wall-clock, rounded up, on an otherwise idle machine; the exact
recorded figures are in `execution_status.json`. They move with system load, so treat
the ~60 s budget as the constraint and these numbers as a guide.)

Companion pages with objectives, prerequisites and exercises for each notebook live
in [`docs/tutorials/`](../docs/tutorials/).

---

## Before you start

```bash
cd <repo>                       # the directory containing pyproject.toml
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[dev]"
source scripts/env.sh           # PLANLAB_ROOT, PYTHONPATH, MPLCONFIGDIR, PLANLAB_PY
```

You need `numpy`, `scipy`, `pandas`, `matplotlib`, `nbclient`, `nbformat` and
`ipykernel` — all of which `pip install -e ".[dev]"` brings in. Nothing else: the
notebooks never touch the network, because `data/processed/` is committed.

The first notebook prints the interpreter path it is running, so you can tell at a
glance whether you are inside the project virtual environment.

### Run them

```bash
# execute every notebook headlessly and rewrite its outputs in place (~1 min)
.venv/bin/python notebooks/execute_all.py

# one notebook at a time
.venv/bin/python notebooks/execute_all.py --only 03_one_period_walkthrough.ipynb

# open them interactively
.venv/bin/python -m jupyter notebook notebooks/
```

`execute_all.py` registers an *ephemeral* kernelspec pointing at its own
interpreter, so the notebook can never silently run under a different Python. It
writes `execution_status.json`, which records the wall time and the sha256 of each
notebook **after** execution; `tests/test_nb_notebooks.py` uses that hash to prove
the stored outputs belong to the committed file.

### Verify what is committed

```bash
.venv/bin/python -m pytest tests/test_nb_notebooks.py -q
```

---

## How the notebooks teach

Four conventions run through all six, and they are worth knowing before you read the
first cell:

1. **Every number is computed twice.** Wherever a notebook reports a quantity that
   the engine also computes, it recomputes it in front of you with independent code
   and asserts agreement. Notebook 03 does this for an entire period and gets
   bit-identical arrays; notebook 05 does it for the dual variables of the central
   LP; notebook 06 does it for a published table.
2. **The last cell is a receipt.** Each notebook ends with a table of checks with
   the *measured* error in each, and `assert all(...)`. If a number does not close,
   the notebook fails — it does not round.
3. **Bilingual prose.** Markdown is Chinese first (the intended audience), then a
   one- or two-sentence italic English summary. Code, comments and every matplotlib
   label are English, so no CJK font is needed to execute them.
4. **Every code cell prints something.** A cell that prints nothing is a cell you
   cannot verify; the authoring rule and the test suite both enforce this, which
   makes "has this notebook been executed?" visible at a glance.

### Why the reproducible-looking structure matters

The repository's claim is that *the institution is the treatment and the physics is
shared*. The notebooks are where that claim becomes checkable: notebooks 01–03 teach
the physics (technology, accounting, feasibility, rationing), 04–05 teach the
institutions (six mechanisms, dual track, shadow prices), and 06 teaches how the
paper's numbers are produced from the raw records.

---

## A suggested three-hour teaching sequence

This is a design for one sitting with a graduate student who has read
`docs/PROTOCOL.md` and the paper's introduction. It assumes the notebooks are open
interactively and that the student types along.

| time | what | why |
|---|---|---|
| 0:00–0:15 | **01 §1–§3**: load the table, verify `x = A x + y`, check that `A` is productive | establishes the substrate and the one identity everything else relies on |
| 0:15–0:35 | **01 §4–§6**: Leontief inverse, output multipliers, the capacity-shock table | the student discovers that amplification depends on the shocked sector's *position*, not the shock's size |
| 0:35–0:40 | **01 §7 + exercise 2**: static vs dynamic amplification | motivates the simulation before showing it |
| 0:40–1:10 | **02 in full**: simulate the delay equation, find the boundary, measure the period | the only closed-form result in the project; here it is derived, not quoted |
| 1:10–1:20 | **02 §7 + exercise 6**: the positive loop, and why the theorem applies to prices, not to output targets | connects the maths to `docs/AMENDMENTS.md` (H1′) |
| 1:20–1:40 | *break* | |
| 1:40–2:20 | **03 in full**: the 3-sector hand calculation, then the real period | the flagship: every step of a period, auditable, with the ledger at the end |
| 2:20–2:30 | **03 §9 exercise 1**: switch to the `consumption_first` ladder and watch the engine need several passes | the ladder is a treatment, not a detail; this is where it becomes obvious |
| 2:30–2:50 | **04**: six mechanisms on one shock, then §4 (common random numbers) | the identification strategy, demonstrated rather than asserted |
| 2:50–3:05 | **05 §1–§3**: write the central LP out, extract the duals, compare with the mechanism's prices | what a shadow price actually is, computed rather than described |
| 3:05–3:20 | **05 §5–§6** + **06**: the dual-track frontier, then reproduce a paper table | ends with "and here is how you check we did not make it up" |

If you have only **one hour**, do 01 §1–§3, 02 §1–§3, 03 §6, and 06. That is the
identity, the theorem, the audit, and the reproducibility check.

If you have **a full day**, add: 01 exercise 4 (inventories), 02 exercise 5
(estimate the loop gain `b`), 03 exercise 2 (the stock ratio), 04 exercise 4
(price gain vs the boundary), 05 exercise 5 (is the interior optimum robust to
seeds?), and 06 exercise 1 (reproduce a *different* table).

---

## Reading the notebooks without running them

The committed outputs are meant to be read. Each notebook's story is visible in its
stored output even in a static viewer:

* **01** ends with the amplification table: 8 of 20 sectors transmit a 50% capacity
  cut, the other 12 absorb it in their own final deliveries; `c12` costs 4.70 times
  the direct loss in output.
* **02** prints the boundary table to `1e-15` and the measured period to within
  0.2 quarters, and it shows why "every extra period halves the gain" is wrong
  asymptotically (`g*(m) ~ π/(2m+1)`: doubling the delay halves the gain).
* **03** prints the complete account of one period twice — once for a three-sector
  economy you can check on paper, once for the real 20-sector table — and ends with
  26/26 checks.
* **04** prints six institutions on one shock, and the correlation that shows why
  the welfare ranking is driven by *how many* sectors are starved rather than by
  total consumption.
* **05** prints the shadow prices: `c12` rises from 1.15 to 11.19 at the peak, and
  the duals equal the mechanism's own prices to 4e-16.
* **06** either reproduces a paper table to the precision it is stored with, or
  stops with an executable instruction and demonstrates the procedure on its own
  data.

---

## Authoring pipeline (for maintainers)

The notebooks are generated from plain-Python cell lists so that a diff of a
teaching notebook is a diff of text rather than a diff of JSON:

```
notebooks/nb_sources/NN_*.py     -- the cells (markdown + code), one per notebook
        |  build_notebooks.py    -- writes the .ipynb skeletons
        v
notebooks/NN_*.ipynb             -- the deliverable (executed, outputs committed)
        |  execute_all.py        -- runs them headlessly, rewrites outputs, records status
        v
tests/test_nb_notebooks.py       -- asserts the committed state is the executed state
```

```bash
.venv/bin/python notebooks/build_notebooks.py                    # rebuild all six
.venv/bin/python notebooks/build_notebooks.py --keep-outputs      # prose-only edit
.venv/bin/python notebooks/build_notebooks.py --only 04_mechanism_zoo.ipynb
.venv/bin/python notebooks/execute_all.py                        # execute all six
.venv/bin/python -m pytest tests/test_nb_notebooks.py -q          # verify the commit
```

`build_notebooks.py` without `--keep-outputs` discards stored outputs; always follow
it with `execute_all.py` before committing.

---

## If a number does not close

The notebooks are written so that a disagreement is a *failure*, not a footnote.
Notebook 03 §8 lists the five places where the accounts usually fail to close
(intermediate requirements computed on targets instead of realised output;
forgetting the `min(·, demand)` cap; treating `S_end` as a residual rather than a
line item; float-comparing tiers; dropping the labour constraint).

If none of those explains a mismatch between the engine and an explicit
recomputation, that is a genuine bug in the model: report it rather than adjusting
the prose, because the whole point of the exercise is that the prose is downstream
of the arithmetic.

---

## Where the numbers come from

* Substrate: Eurostat `naio_10_cp1700`, symmetric input–output table at basic
  prices, EU27, 2020, million EUR, aggregated 64 → 20 sectors. Committed under
  `data/processed/`, with provenance and hashes in `data/metadata/manifest.json`
  (© European Union, CC-BY 4.0; the aggregation is ours).
* Reference run used by notebooks 01, 03 and 05: 50% capacity cut to sector `c12`
  for 8 quarters from t = 40, inventory ratio 0.10, 60 periods, `material_balance`
  with default parameters → peak aggregate shortage **0.1905**, aggregate output
  loss **0.1000**. If those two numbers move, something in the engine or the data
  changed, and the notebooks will fail rather than silently drift.
* Paper artefact used by notebook 06: `results/` produced by
  `bash scripts/reproduce.sh` (tables under `results/tables/`, figures under
  `figures/`).

---

## Licence

Code and prose: MIT (see `LICENSE`). The derived input–output data: CC-BY 4.0, as
recorded in `data/metadata/manifest.json`.
