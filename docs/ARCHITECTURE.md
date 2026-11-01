# PlanLab — Architecture & Interface Contract

**Status:** frozen contract (v1). Every module and experiment is built against this document.
**Owner:** Lead. Teammates may *not* change these interfaces without asking the Lead.

PlanLab is a mechanism-switchable simulator of a planned economy built on an
input–output (Leontief) substrate. The scientific claim it supports: *the
institution* (who sets targets, who rations scarce goods, how prices move) is a
replaceable module; the *physics* (technology, accounting, capacity, inventories)
is shared and auditable.

```
data layer -> technology -> mechanism (institution) -> engine (physics) -> metrics -> experiments -> paper
```

---

## 1. Repository layout

```
planlab/
├── src/planlab/
│   ├── core/            # Lead-owned. Frozen interfaces.
│   │   ├── state.py     # dataclasses: Technology, State, Observation, Plan, Allocation, PeriodRecord
│   │   ├── io_model.py  # Technology construction/validation, I-O identity checks
│   │   ├── engine.py    # period loop, feasibility fixed point, accounting
│   │   ├── metrics.py   # outcome metrics + welfare
│   │   ├── shocks.py    # demand/capacity shock processes
│   │   └── scenarios.py # YAML scenario loading + validation
│   ├── mechanisms/      # one file per institution; registry in __init__.py
│   │   ├── base.py      # CoordinationMechanism ABC (Lead-owned, frozen)
│   │   ├── material_balance.py   (reference impl, Lead)
│   │   ├── lp_central.py, tatonnement.py, rationing_priority.py,
│   │   │   market.py, hybrid_dual_track.py   (T1)
│   ├── data/            # T2: fetch/parse/aggregate I-O tables, synthetic generator
│   └── experiments/     # T3: grid, runner, aggregate, figures
├── scenarios/           # T3: 16 standard scenario YAML files (SCENARIO_SPEC.md)
├── data/                # T2: raw/ processed/ metadata/
├── results/             # T3: run outputs (csv/parquet + manifest.json)
├── figures/             # T3: publication figures (png + pdf)
├── paper/               # T4: manuscript, tables/, figures/ (symlinked copies)
├── notebooks/           # T5: teaching notebooks (executed)
├── tests/               # everyone; file naming convention below
├── verification/        # T6: independent verification scripts + report
├── docs/                # this file, PROTOCOL.md, DATA_SCHEMA.md, SCENARIO_SPEC.md
└── scripts/             # thin CLI wrappers
```

**File-ownership convention for tests:** `tests/test_core_*.py` (Lead),
`tests/test_mech_*.py` (T1), `tests/test_data_*.py` (T2),
`tests/test_exp_*.py` (T3), `tests/test_verify_*.py` (T6).
Nobody edits another owner's files; report problems in the shared task instead.

---

## 2. Units and accounting identity (non-negotiable)

* All sectoral quantities are in **constant-price value units** (millions of the
  base-year currency unit, from the source I-O table, basic prices). A sector's
  "output" is its value added at constant prices; real-volume heterogeneity is
  out of scope and stated as a limitation. Never mix units.
* One period = one **quarter**. `T` periods per run.
* Per-sector balance must hold to within `1e-8 * scale` every period:

  `S0_i + x_i + M_i = sum_j Z_ij + C_i + G_i + I_i + S_end_i + S_i^{spoil}`

  with `M_i` imports (0 unless the scenario enables trade), `I_i` investment demand,
  and `S_i^{spoil}` **spoilage**: output that is neither used nor storable. Storage is
  finite — `S_end_i <= max_stock_ratio * base_output_i` (default 2.0, see
  `docs/AMENDMENTS.md` Amendment 4) — and the excess is recorded, not silently
  accumulated. The audit is implemented once, in
  `planlab.core.engine.balance_residual`; import it rather than re-deriving the
  identity. This is asserted by the engine every period (raises `AccountingError`).

---

## 3. Core dataclasses (`planlab.core.state`)

```python
@dataclass(frozen=True)
class Technology:
    sectors: tuple[str, ...]        # (n,)
    A: np.ndarray                   # (n,n) direct input coefficients, a[i,j] = Z_ij / x_j
    labor: np.ndarray               # (n,) labour per unit of gross output
    kappa: np.ndarray               # (n,) capital-output ratio; capacity = K / kappa
    consumption_weights: np.ndarray # (n,) welfare weights w_i
    essentials: np.ndarray          # (n,) 1.0 essential, 0.0 non-essential (priority tier)
    base_output: np.ndarray         # (n,) base-year gross output (calibration anchor)
    base_final: np.ndarray          # (n,) base-year final consumption
    meta: Mapping[str, Any]         # provenance dict from the data layer

    @property n: int

@dataclass
class State:                        # mutable, one instance per run
    t: int
    x: np.ndarray            # (n,) gross output in t-1 (realised)
    inventory: np.ndarray    # (n,) S0 for the next period
    capital: np.ndarray      # (n,)
    capacity: np.ndarray     # (n,) = capital / kappa, after shocks
    price: np.ndarray        # (n,) >= 1e-9, index with base = 1.0
    backlog: np.ndarray      # (n,) unfilled household demand carried over
    gov_demand: np.ndarray   # (n,) exogenous government/other final demand level

@dataclass(frozen=True)
class Observation:          # what the coordinator may see (delay + noise applied)
    t: int
    lag: int                # k actually applied
    demand: np.ndarray      # (n,) estimated household final demand
    capacity: np.ndarray    # (n,) estimated capacity
    inventory: np.ndarray   # (n,) estimated beginning inventory
    price: np.ndarray       # (n,) estimated prices
    backlog: np.ndarray     # (n,) estimated backlog
    labour: float           # estimated available labour

@dataclass
class Plan:
    targets: np.ndarray             # (n,) gross output targets x*
    final_targets: np.ndarray       # (n,) intended final deliveries y*
    priority: np.ndarray | None     # (n,) optional priority weights in [0,1]
    meta: dict                      # diagnostics (solver status, iterations, ...)

@dataclass
class Allocation:
    claims: np.ndarray        # (m, n): rows 0..n-1 = sector input requirements,
                              # rows n.. = final claimants (labels below)
    tiers: np.ndarray         # (m,) priority, lower is served first
    labels: tuple[str, ...]   # "inter:j", "final:essential", "final:other",
                              # "gov:...", "invest:...", "stock:..."
    meta: dict
```
`Allocation.build(...)` is the canonical constructor used by the shipped
mechanisms; `Allocation.split_grants(grants)` maps grants back to
`intermediates / final / gov / invest / stock`.  The default priority ladder is
intermediates (0) -> essential final (1) -> other final (2) -> gov (3) -> invest (4)
-> stock (5); see the note in section 5.

```python
@dataclass
class PeriodRecord:
    t: int
    x: np.ndarray; Z: np.ndarray; C: np.ndarray; G: np.ndarray; Invest: np.ndarray
    inventory_end: np.ndarray; demand: np.ndarray; unmet: np.ndarray
    price: np.ndarray; capacity_util: np.ndarray; plan_targets: np.ndarray
    plan_status: str; iterations: int
    metrics: dict[str, float]
```

`Plan.meta` and `Allocation.meta` are free-form; the engine only reads the keys
`"status"` and `"iterations"` (both optional).

---

## 4. Mechanism interface (`planlab.mechanisms.base`)

An institution is a class implementing three decisions. **The engine owns all
accounting and feasibility**; mechanisms must never mutate `State` and must never
look at `PeriodRecord`-level truth from the future (no look-ahead; only `Observation`).

```python
class CoordinationMechanism(ABC):
    name: str                      # registry key, e.g. "material_balance"
    def __init__(self, tech: Technology, params: Mapping[str, float]): ...
    def form_plan(self, obs: Observation, state: State) -> Plan: ...
    def allocate(self, obs: Observation, state: State, plan: Plan,
                 supply: np.ndarray) -> Allocation: ...
    def update_prices(self, obs: Observation, state: State, plan: Plan,
                      alloc: Allocation, realised: "Realisation") -> np.ndarray: ...
```

* `supply[i]` = nominal supply of good *i* available this period
  (`state.inventory[i] + planned output of good i`). Mechanisms split it among
  claimants. The engine verifies feasibility and will *scale down* over-claims,
  setting `alloc.meta["scale_applied"] = True`; shipped mechanisms must not
  trigger that in calm conditions (asserted in tests).
* `update_prices` returns the price vector for the *next* period.
* `Realisation` is a small read-only view: `x`, `C`, `unmet`, `Z`, `excess_demand`.

Registry: `from planlab.mechanisms import get_mechanism; get_mechanism(name, tech, params)`.

Shipped mechanisms (name → file):

| name | institution |
|---|---|
| `lp_central` | omniscient central LP (welfare max subject to `(I-A)x >= y`, capacity) — upper bound |
| `material_balance` | Gosplan-style resource/need balancing with iterative adjustment |
| `rationing_priority` | administrative quotas by priority tier, proportional within tier |
| `tatonnement` | Lange/shadow-price board: price-mediated allocation + tâtonnement updates |
| `market` | competitive tâtonnement inside the period (Walrasian benchmark) |
| `hybrid_dual_track` | plan track at admin prices for share θ, market track for 1-θ |

---

## 5. Engine period order (`planlab.core.engine`)

```
                       ┌───────────────────────────────────────────┐
period t:              │ 1 shock draw        cap_t, demand_t, gov_t │
                       │ 2 observation       obs_t = delayed+noisy  │
                       │ 3 form_plan         plan_t = mech(obs_t)   │
                       │ 4 nominal supply    S0 + plan_t.targets    │
                       │ 5 allocate          claims = mech.allocate │
                       │ 6 feasibility FP    x_t = short-side fixed │
                       │                     point over allocations │
                       │ 7 accounting        Z=A·x, C, G, I, S_end  │
                       │ 8 accounting audit  raise on residual      │
                       │ 9 prices            p_{t+1}=mech.update    │
                       │10 metrics + record                        │
                       └───────────────────────────────────────────┘
```

Step 6 detail (the short-side allocation).  See `planlab.core.feasibility` and
`Simulation._reconcile`.  Given capacity, inventories and the claims that outrank
production inputs, the feasible output solves

```
maximise  sum_j x_j
s.t.      A x - x <= r - reserve        (goods balance, per good)
          0 <= x <= min(plan.targets, capacity)
          labour . x <= L
```

This is the standard non-substitution / Dreze-Benassy short-side allocation.  A
*proportional cascade* was tried first and rejected: in a capacity-constrained
input-output economy a cascade that shrinks every sector proportionally cannot
rebalance the production mix, and converges to zero rather than to the rebalanced
plan.  The programme is solved only when the target vector is infeasible, so calm
periods cost one vector comparison.

After the programme, the residual supply `r + x - A x` is rationed across the final
claimants by the mechanism's priority tiers (`engine.ration_by_priority`).
`PeriodRecord.iterations` = number of programme/ration passes needed (1 in calm
periods, more when the tier structure binds).

**Priority ladder.** `Allocation.build` assigns tiers; the default is

```
0  intermediate inputs   (production must not be starved)
1  final:essential       (food, energy, health, transport)
2  final:other
3  gov
4  invest
5  stock
```

`material_balance` accepts `ladder: production_first | consumption_first`; the
consumption-first ladder (essentials at tier 0, intermediates at tier 1) is a
treatment, not a default, because fixed final claims that outrank production inputs
can starve production and produce a self-reinforcing collapse.  That contrast is a
result; the engine reproduces it.

## 6. Metrics (`planlab.core.metrics`)

Per period and per run:

* `shortage_rate = sum_i max(0, d_i - C_i) / sum_i d_i`
* `plan_fulfilment_i = x_i / plan_targets_i` (unweighted mean and sector table)
* `inventory_ratio_i = inventory_i / demand_i`
* `welfare = sum_i w_i * log(C_i + c0)` with `c0 = 1e-3 * base_final_i`
* `penalised_welfare = welfare - l1*shortage_rate - l2*inventory_cost
  - l3*plan_deviation - l4*inequality - l5*adjustment_cost`
  (the weights are scenario parameters, never "true" social welfare)
* `output_volatility = std(log(x_t))` per sector and aggregate
* `price_dispersion = std(log p)` across sectors; `price_volatility = std(Δlog p)`
* `plan_deviation = mean_i |x_i/x*_i - 1|`
* `capital_utilisation = mean(x / capacity)`
* `amplification`: for shock experiments, `Amp = (peak shortfall share in
  downstream sectors) / (direct capacity shock share)`; run-level only.
* Run-level: cumulative discounted welfare, mean/worst-period shortage,
  recovery half-life `h` (periods until shortage rate returns to within 10% of
  baseline), and `stability_index = 1 / (1 + cv(shortage_rate))`.

All metric dicts are flat `dict[str, float]` so they concatenate into parquet.

---

## 7. Scenario file schema (`planlab.core.scenarios`) — see `docs/SCENARIO_SPEC.md`

```yaml
id: baseline_20s
name: "Baseline: perfect information, material balance"
mechanism: material_balance
mechanism_params: {balance_rounds: 3, priority_tier_first: true}
technology: {source: eurostat, year: 2020, n_sectors: 20}   # resolved by data layer
periods: 40
seeds: [0, 1, 2]           # Monte Carlo replicates (T3 expands)
shocks: []                 # list of shock specs, see shocks.py
info: {lag: 0, noise: 0.0, observe_capacity: true}
demand: {growth: 0.0, ar1: 0.0, sigma: 0.0}
objective: {weights: {shortage: 1.0, inventory: 0.1, plan_deviation: 0.5,
                      inequality: 0.5, adjustment: 0.1}}
tags: [core]
```

Every scenario is fully self-describing (no hidden defaults beyond documented
ones), and `planlab run <scenario.yaml>` must be able to run it standalone.

---

## 8. Results schema (T3)

`results/<experiment_id>/`
* `runs.parquet` — one row per (scenario, seed, period) with all metrics.
* `summary.csv` — one row per (scenario, seed): run-level metrics.
* `aggregate.csv` — one row per scenario: mean/sd/CI across seeds.
* `manifest.json` — git commit (or content hash), python/numpy/scipy versions,
  scenario hashes, timestamp, wall time, seeds.
* Recipe: every reported number in the paper must be regenerable by
  `make reproduce` (or `scripts/reproduce.sh`) without manual steps.

---

## 9. Definition of done (per module)

1. Public functions documented (docstring with Args/Returns).
2. `pytest tests/` green, including the owner's tests.
3. `ruff`-clean not required, but no dead code, no debug prints, no bare excepts.
4. No hard-coded absolute paths; all paths via `planlab.paths` helpers.
5. Numerical results deterministic given `seed` (`numpy.random.default_rng`).
