"""The experimental grid: every cell of the pre-registered design, as data.

A *cell* is one treatment combination (block, mechanism, gain, lag, shock, ...)
together with its seed list.  Cells are pure data: the runner turns each one into
a ``Scenario``, runs every seed and writes the numbers.  Nothing in this module
touches the filesystem or the clock, so the grid can be inspected, hashed and
tested without running anything (``pytest tests/test_exp_grid.py``).

Design sources
--------------
* ``docs/PROTOCOL.md``      — the pre-registration (blocks B1–B6, seeds, outcomes)
* ``docs/AMENDMENTS.md``    — the authoritative amendments (shock parameters,
  the gain axis, and the switch of the H1 stability test from plan targets to
  tatonnement prices)
* ``docs/notes_delay_theorem.md`` — ``m = lag + 1``, ``g*(m) = 2 sin(pi/(2(2m+1)))``

Common random numbers
---------------------
Every cell of a block shares one ``scenario.id`` (the *common-random-number id*).
The engine derives its RNG stream from ``(scenario.id, seed)``, so giving the six
mechanisms of a block the same scenario id makes the demand-noise and
measurement-noise paths identical across mechanisms, across gains and across
lags.  Mechanism comparisons are therefore paired by construction, which is what
the protocol requires ("identical seed -> identical exogenous path").  The cell's
own identity (``cell_id``) is separate from the scenario id and is what appears
in ``results/``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "PERIODS",
    "BURN_IN",
    "SHOCK_START",
    "SHOCK_DURATION",
    "SHOCK_SIZE",
    "STOCK_RATIO",
    "STANDARD_SECTOR",
    "STANDARD_SIGMA_DEMAND",
    "STANDARD_AR1",
    "STANDARD_SIGMA_OBS",
    "GAINS",
    "LAGS",
    "MECHANISMS",
    "GAIN_PARAMS",
    "REFERENCE_GAIN",
    "SEEDS",
    "Cell",
    "BLOCKS",
    "build_cells",
    "build_grid",
    "block_run_count",
    "available_mechanisms_safe",
    "split_runnable",
    "perturb_technology",
    "cell_from_dict",
]

# ---------------------------------------------------------------------------
# design constants  (docs/PROTOCOL.md §4 as amended by docs/AMENDMENTS.md)
# ---------------------------------------------------------------------------

#: periods per run.  The protocol's 60 quarters are extended to 80 in
#: ``docs/DEVIATIONS.md`` §1: an 8-quarter shock starting at t=40 plus the
#: pre-registered 20-period detrending window and the predicted oscillation
#: period 2(2m+1) <= 26 need a post-shock window of at least two cycles.
PERIODS = 80

#: periods discarded from all run-level statistics
BURN_IN = 30

#: first period of the standard capacity shock (absolute period number)
SHOCK_START = 40

#: duration of the standard capacity shock, in quarters (Amendment 2)
SHOCK_DURATION = 8

#: capacity *multiplier* during the shock: 0.50 = lose 50% of capacity (Amendment 2)
SHOCK_SIZE = 0.50

#: initial inventory as a share of base-year gross output (Amendment 2)
STOCK_RATIO = 0.10

#: the sector the standard shock hits.  c12 has the highest forward linkage in
#: the Eurostat 2020 20-sector substrate and is the sector the Lead's pilot used.
STANDARD_SECTOR = "c12"

#: gain grid (Amendment 3): ``revision_gain`` for plan-revision mechanisms,
#: ``price_gain`` for tatonnement.
GAINS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6)

#: reporting lag ``k``.  The theory's effective delay is ``m = k + 1``.
LAGS: tuple[int, ...] = (0, 1, 2, 3, 4, 6)

#: pre-registered mechanism set (docs/PROTOCOL.md §3.1)
MECHANISMS: tuple[str, ...] = (
    "lp_central",
    "material_balance",
    "rationing_priority",
    "tatonnement",
    "market",
    "hybrid_dual_track",
)

#: which parameter carries the gain axis for each mechanism.  Mechanisms absent
#: from this map have no gain-parameterised feedback loop; they are run at the
#: reference gain only (Amendment 3: drop mechanism coverage in the high-gain
#: corners first, never seeds).
GAIN_PARAMS: Mapping[str, str] = {
    "material_balance": "revision_gain",
    "tatonnement": "price_gain",
}

#: Parameters applied to **every mechanism in every cell**.  They are household-side
#: or planning-side settings that are *not* institutional choices, and letting them
#: differ would confound the mechanism comparison (Lead, identification review):
#: ``backlog_carry`` re-injects last period's unmet demand into this period's claim,
#: which is a positive feedback channel unrelated to the coordination rule; a
#: differing ``stock_target_ratio`` changes the buffer the planner holds.  Only the
#: institutional knob (revision/price gain, plan share theta, quota speed) is
#: allowed to vary between cells.
COMMON_MECHANISM_PARAMS: Mapping[str, Any] = {
    "backlog_carry": 0.0,
    "stock_target_ratio": 0.10,
    "max_target_ratio": 2.5,
    "min_target_ratio": 0.25,
}

#: The documented **reference operating point** of every institution, pinned in every
#: cell of every block (Lead's specification).  Without this, "material balance" at
#: ``revision_gain = 0`` (its module default during development) and at ``0.30`` are
#: different institutions, and the paper's central ranking would move with a module
#: default.  The treatment axes override the relevant knob in the cells that sweep it
#: (``revision_gain`` in B2/B2L/B8, ``price_gain`` in B2/B7, ``theta`` in B5,
#: ``balance_rounds`` in B6/B8); elsewhere the pinned value *is* the mechanism.
#:
#: ``lp_central.use_shadow_prices = 0`` is not a treatment but an artefact switch: its
#: LP duals are Hilbert-space shadow prices, not the consumer prices of this demand
#: system, and leaving them on collapses the information-rich benchmark (see
#: ``docs/DEVIATIONS.md``).
MECHANISM_BASE_PARAMS: Mapping[str, Mapping[str, Any]] = {
    "lp_central": {"use_shadow_prices": 0.0},
    "material_balance": {"revision_gain": 0.30, "balance_rounds": 3},
    "rationing_priority": {"quota_speed": 0.50, "revision_interval": 1},
    "tatonnement": {"price_gain": 0.30},
    "market": {},
    "hybrid_dual_track": {"theta": 0.50, "market_gain": 0.25},
}

#: the reference gain used for the lag x mechanism grid and for every mechanism
#: that has no gain parameter.  0.30 is the shipped ``material_balance`` default.
REFERENCE_GAIN = 0.30

#: seeds per block (docs/PROTOCOL.md §4)
SEEDS: Mapping[str, tuple[int, ...]] = {
    "B1": (0,),
    "B1b": tuple(range(20)),
    "B2": tuple(range(20)),
    "B3": tuple(range(20)),
    "B4": tuple(range(20)),
    "B5": tuple(range(20)),
    "B6": tuple(range(20)),
    "B2L": tuple(range(20)),
    "B7": (0, 1, 2),
    "B8": tuple(range(16)),
    "P2": (0,),
}

#: Standard information regime (the "σ" of docs/PROTOCOL.md §3).  Calibrated
#: before the main runs so that the *calm* economy is quiet but the Monte Carlo
#: has dispersion: a no-shock run then has mean shortage ~0.9% (measurement noise
#: alone makes a planner mis-plan), the standard c12 shock peaks at ~0.19 with a
#: seed-to-seed sd of ~0.002, and the shocked economy's mean shortage rises
#: monotonically with the reporting lag (3.0% at k=0 to 5.1% at k=6), which is the
#: variation B2 is designed to measure.  Recorded in docs/EXPERIMENTS_RUN.md §2.
STANDARD_SIGMA_DEMAND = 0.01
STANDARD_AR1 = 0.4
STANDARD_SIGMA_OBS = 0.02

#: price elasticities.  0.4 is the standard environment (the tatonnement loop
#: needs a negative price->excess-demand feedback; Amendment 1); 0.8 is the
#: robustness level that should *halve* the critical price gain if the theory's
#: ``eta* = g*/b`` with ``b ~= elasticity`` holds.
STANDARD_ELASTICITY = 0.4
HIGH_ELASTICITY = 0.8

#: shock sizes of block B4 (probabilities of capacity loss are 1 - size)
B4_SIZES: tuple[float, ...] = (0.10, 0.30, 0.50)

#: dual-track plan shares (block B5)
THETAS: tuple[float, ...] = tuple(round(0.1 * i, 10) for i in range(11))

#: the three mechanisms of block B3 (propagation)
B3_MECHANISMS: tuple[str, ...] = ("material_balance", "rationing_priority", "tatonnement")

#: the two mechanisms of block B4 (rationing vs shadow price)
B4_MECHANISMS: tuple[str, ...] = ("rationing_priority", "tatonnement")

#: B1 scenarios: (cell id, mechanism)
B1_CASES: tuple[tuple[str, str], ...] = (
    ("baseline_lp_k0", "lp_central"),
    ("baseline_mb_k0", "material_balance"),
)

#: names of candidate parameters that carry the plan share of the dual track.
_THETA_KEYS: tuple[str, ...] = ("theta", "plan_share", "plan_share_theta", "dual_track_theta")


# ---------------------------------------------------------------------------
# scenario template
# ---------------------------------------------------------------------------
def _scenario(
    *,
    scenario_id: str,
    name: str,
    mechanism: str,
    seeds: Sequence[int],
    lag: int = 0,
    noise: float = 0.02,
    sigma: float = 0.01,
    ar1: float = 0.4,
    price_elasticity: float = STANDARD_ELASTICITY,
    shocks: Sequence[Mapping[str, Any]] = (),
    mechanism_params: Mapping[str, Any] | None = None,
    objective: Mapping[str, float] | None = None,
    periods: int = PERIODS,
    burn_in: int = BURN_IN,
    stock_ratio: float = STOCK_RATIO,
    technology: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
    labor_supply_factor: float = 1.15,
) -> dict[str, Any]:
    """Build a complete scenario dictionary (the schema of ``core.scenarios``).

    Every field is explicit; nothing is inherited from a mutable default, so the
    dictionary can be hashed, written to JSON and shipped in the manifest.
    """
    obj = {
        "shortage": 1.0,
        "inventory": 0.1,
        "plan_deviation": 0.5,
        "inequality": 0.0,
        "adjustment": 0.1,
    }
    obj.update(dict(objective or {}))
    return {
        "id": str(scenario_id),
        "name": str(name),
        "mechanism": str(mechanism),
        "technology": dict(technology or {"source": "eurostat", "year": 2020, "n_sectors": 20}),
        "periods": int(periods),
        "seeds": [int(s) for s in seeds],
        "burn_in": int(burn_in),
        "shocks": [dict(s) for s in shocks],
        "info": {
            "lag": int(lag),
            "noise": float(noise),
            "observe_capacity": True,
            "observe_inventory": True,
            "observe_backlog": True,
            "planning_horizon": 1,
        },
        "demand": {
            "growth": 0.0,
            "ar1": float(ar1),
            "sigma": float(sigma),
            "price_elasticity": float(price_elasticity),
            "income_elasticity": 1.0,
        },
        "objective": obj,
        "household": {
            "essential_priority": 0.0,
            "backlog_persistence": 0.5,
            "hoarding": 0.0,
        },
        "mechanism_params": dict(mechanism_params or {}),
        "stock_ratio": float(stock_ratio),
        "investment_share": 0.12,
        "gov_share": 0.10,
        "depreciation": 0.02,
        "discount": 0.98,
        "labor_supply_factor": float(labor_supply_factor),
        "indexation": 0.0,
        "price_bounds": [0.05, 20.0],
        "tags": [str(t) for t in tags],
    }


def _capacity_shock(
    sector: str | int,
    *,
    size: float = SHOCK_SIZE,
    start: int = SHOCK_START,
    duration: int = SHOCK_DURATION,
) -> dict[str, Any]:
    return {
        "type": "capacity",
        "sector": sector,
        "start": int(start),
        "duration": int(duration),
        "size": float(size),
    }


# ---------------------------------------------------------------------------
# the cell
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One row of the experimental design.

    Attributes
    ----------
    block:
        ``B1``..``B6``, the Lead's price-response block ``B7``, plus the
        exploratory extensions ``B2L`` (priority ladder) and ``P2`` (the ``b_T``
        sign probe behind H1').
    cell_id:
        Stable, human-readable identity of the treatment combination.  This is
        what ``results/`` is keyed on.
    mechanism:
        Registry id of the institution (reporting only: the runner resolves it
        and skips the cell with a warning if it is not registered).
    scenario:
        Complete scenario dictionary (see :func:`_scenario`).  Its ``id`` is the
        common-random-number id of the block, *not* the cell id.
    seeds:
        Replicate seeds.  Shared across every mechanism of a block.
    treatment:
        Flat dict of factor levels; becomes columns of ``summary.csv`` so that
        every table can pivot on it.
    tech:
        Technology spec: ``{"source", "year", "n_sectors"}`` plus optional
        ``a_perturb`` / ``perturb_seed``.
    probe:
        Optional probe descriptor (``target_offset`` / ``price_offset``); probes
        inject an instrument directly into a fresh mechanism instance instead of
        changing the scenario, and are used only to measure the local gain ``b``.
    notes:
        Free text recorded in the manifest.
    """

    block: str
    cell_id: str
    mechanism: str
    scenario: Mapping[str, Any]
    seeds: tuple[int, ...]
    treatment: Mapping[str, Any] = field(default_factory=dict)
    tech: Mapping[str, Any] = field(default_factory=dict)
    probe: Mapping[str, Any] | None = None
    notes: str = ""

    # ------------------------------------------------------------------ views
    @property
    def scenario_id(self) -> str:
        return str(self.scenario["id"])

    @property
    def lag(self) -> int:
        return int(self.scenario["info"]["lag"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "cell_id": self.cell_id,
            "mechanism": self.mechanism,
            "scenario_id": self.scenario_id,
            "seeds": list(self.seeds),
            "treatment": dict(_jsonable(self.treatment)),
            "tech": dict(_jsonable(self.tech)),
            "probe": None if self.probe is None else dict(_jsonable(self.probe)),
            "notes": self.notes,
        }

    def scenario_dict(self) -> dict[str, Any]:
        """The full scenario, JSON-serialisable (used by manifests and the CLI)."""
        return _jsonable(self.scenario)

    def full_dict(self) -> dict[str, Any]:
        """Cell *and* scenario: what the runner hashes and the manifest records."""
        return {**self.to_dict(), "scenario": self.scenario_dict()}

    def hash(self) -> str:
        """Content hash of the cell; changes if any design input changes."""
        payload = json.dumps(self.full_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cell_from_dict(data: Mapping[str, Any]) -> Cell:
    """Inverse of :meth:`Cell.to_dict` (round-trip used by tests and manifests)."""
    return Cell(
        block=str(data["block"]),
        cell_id=str(data["cell_id"]),
        mechanism=str(data["mechanism"]),
        scenario=dict(data["scenario"]),
        seeds=tuple(int(s) for s in data["seeds"]),
        treatment=dict(data.get("treatment") or {}),
        tech=dict(data.get("tech") or {}),
        probe=data.get("probe"),
        notes=str(data.get("notes", "")),
    )


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python objects."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# technology construction (substrate + A-perturbation)
# ---------------------------------------------------------------------------
def _standard_tech_spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"source": "eurostat", "year": 2020, "n_sectors": 20}
    spec.update(overrides)
    return spec


def perturb_technology(tech: Any, magnitude: float = 0.05, seed: int = 0) -> Any:
    """Return a copy of ``tech`` with ``A`` perturbed log-normally by ``magnitude``.

    ``A'_{ij} = A_{ij} * exp(magnitude * z_{ij})``, ``z ~ N(0, 1)``, columns whose
    sum would reach 1 are rescaled to stay productive, and ``base_final`` is
    recomputed as ``x - A' x`` (clipped at a strictly positive floor) so that the
    perturbed economy has a well-defined base-year final demand.  The random
    draw is a deterministic function of ``seed``.

    Args:
        tech: a ``planlab.core.state.Technology``.
        magnitude: log-scale standard deviation of the perturbation (0.05 = 5%).
        seed: RNG seed; different instances of the same perturbation.

    Returns:
        A new ``Technology`` with the perturbed ``A`` and ``base_final``.
    """
    from planlab.core.state import Technology

    rng = np.random.default_rng(seed)
    A = np.asarray(tech.A, dtype=float)
    z = rng.standard_normal(A.shape)
    A2 = A * np.exp(float(magnitude) * z)
    colsum = A2.sum(axis=0)
    scale = np.where(colsum >= 0.98, 0.98 / np.maximum(colsum, 1e-12), 1.0)
    A2 = A2 * scale.reshape(1, -1)
    base = np.asarray(tech.base_output, dtype=float)
    final2 = np.maximum(base - A2 @ base, 1e-3 * np.maximum(base, 1e-9))
    meta = dict(getattr(tech, "meta", {}) or {})
    meta["a_perturbation"] = {
        "magnitude": float(magnitude),
        "seed": int(seed),
        "max_rel_change": float(np.max(np.abs(A2 / np.maximum(A, 1e-12) - 1.0))),
    }
    return Technology(
        sectors=tech.sectors,
        A=A2,
        labor=tech.labor,
        kappa=tech.kappa,
        consumption_weights=tech.consumption_weights,
        essentials=tech.essentials,
        base_output=base,
        base_final=final2,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# mechanism discovery
# ---------------------------------------------------------------------------
def available_mechanisms_safe() -> list[str]:
    """Registered mechanism ids, or an empty list if the registry cannot load.

    The mechanism package is written in parallel with this module, so a broken
    or absent registry must degrade to "nothing is runnable yet", never to a
    traceback inside the experiment layer.
    """
    try:
        from planlab.mechanisms import available_mechanisms

        return sorted(available_mechanisms())
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"mechanism registry unavailable: {exc!r}", RuntimeWarning, stacklevel=2)
        return []


def split_runnable(
    cells: Iterable[Cell], mechanisms: Sequence[str] | None = None
) -> tuple[list[Cell], list[Cell]]:
    """Split cells into (runnable, skipped) given the registered mechanisms."""
    present = set(available_mechanisms_safe() if mechanisms is None else mechanisms)
    runnable: list[Cell] = []
    skipped: list[Cell] = []
    for cell in cells:
        (runnable if cell.mechanism in present else skipped).append(cell)
    return runnable, skipped


# ---------------------------------------------------------------------------
# block builders
# ---------------------------------------------------------------------------
def _cell_id(*parts: Any) -> str:
    """Readable, stable cell id: ``B2.k2.g0p3.eps0p4.material_balance``."""
    out = []
    for part in parts:
        if isinstance(part, float):
            out.append(("g%g" % part).replace(".", "p").replace("-", "m"))
        else:
            out.append(str(part))
    return ".".join(out)


def _gain_param(mechanism: str) -> str | None:
    return GAIN_PARAMS.get(mechanism)


def _theta_key() -> str | None:
    """Resolve the dual-track plan-share parameter name from the mechanism class.

    The mechanism is being written by another teammate; rather than hard-coding
    a guess that would silently be ignored (all 11 theta cells would then be
    identical), the key is read from the class's ``default_params`` when the
    class is available.  Returns ``None`` when the mechanism is not registered
    yet, in which case the preferred name ``theta`` is used.
    """
    try:
        from planlab.mechanisms import hybrid_dual_track as _hdt  # noqa: PLC0415

        cls = getattr(_hdt, "HybridDualTrack", None)
        defaults = dict(getattr(cls, "default_params", {}) or {})
    except Exception:
        return None
    for key in _THETA_KEYS:
        if key in defaults:
            return key
    return None


def _build_b1() -> list[Cell]:
    """B1 — deterministic calibration/validation on both substrates.

    Six mechanisms x two substrates (the 6-sector toy fixture and the real
    20-sector Eurostat table), one deterministic run each: no shock, no noise, no
    lag.  With perfect information every institution must reproduce the base year,
    so this block is the falsification test of the *implementation*: any departure
    from ``x/base = 1``, a non-zero shortage or a balance residual above ``1e-8``
    is a bug, not a result.
    """
    cells: list[Cell] = []
    substrates: tuple[tuple[str, dict[str, Any]], ...] = (
        ("eurostat", _standard_tech_spec()),
        ("toy6", {"toy": 6}),
    )
    for mechanism in MECHANISMS:
        for substrate, tech in substrates:
            cell_id = _cell_id("B1", mechanism, substrate)
            sc = _scenario(
                scenario_id="B1",
                name=f"B1 validation: {mechanism} on {substrate}, perfect information",
                mechanism=mechanism,
                seeds=SEEDS["B1"],
                lag=0,
                noise=0.0,
                sigma=0.0,
                ar1=0.0,
                price_elasticity=0.0,
                shocks=(),
                mechanism_params=_mech_params(mechanism, REFERENCE_GAIN),
                technology=dict(tech),
                tags=("B1", "validation", "deterministic", substrate),
            )
            cells.append(
                Cell(
                    block="B1",
                    cell_id=cell_id,
                    mechanism=mechanism,
                    scenario=sc,
                    seeds=SEEDS["B1"],
                    treatment={
                        "block": "B1",
                        "mechanism": mechanism,
                        "lag": 0,
                        "m": 1,
                        "gain": REFERENCE_GAIN,
                        "price_elasticity": 0.0,
                        "shock_size": 1.0,
                        "shock_sector": None,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": substrate,
                    },
                    tech=dict(tech),
                    notes="base-year reproduction: no shock, no noise, lag 0",
                )
            )
    return cells


def _mech_params(mechanism: str, gain: float, **extra: Any) -> dict[str, Any]:
    """The full ``mechanism_params`` of a cell.

    Order of precedence: the common identification block, then the pinned
    per-mechanism parameters (``use_shadow_prices``), then the institutional knob
    (``revision_gain`` / ``price_gain`` / ``theta``), then block-specific extras
    (the priority ladder, ``balance_rounds``).  Nothing else is allowed to differ
    between cells, so a mechanism comparison identifies the institution.
    """
    params: dict[str, Any] = dict(COMMON_MECHANISM_PARAMS)
    params.update(MECHANISM_BASE_PARAMS.get(mechanism, {}))
    key = _gain_param(mechanism)
    if key:
        params[key] = float(gain)
    params.update(extra)
    return params


def _build_b2() -> list[Cell]:
    """B2 — gain x lag x mechanism (Amendment 3), 20 seeds per cell."""
    cells: list[Cell] = []
    details: list[tuple[str, Any]] = []
    for mechanism in MECHANISMS:
        gain_param = _gain_param(mechanism)
        gains = GAINS if gain_param else (REFERENCE_GAIN,)
        elasticities = (
            (STANDARD_ELASTICITY, HIGH_ELASTICITY)
            if mechanism == "tatonnement"
            else (STANDARD_ELASTICITY,)
        )
        for gain in gains:
            for lag in LAGS:
                for eps in elasticities:
                    details.append((mechanism, gain, lag, eps))
    for mechanism, gain, lag, eps in details:
        cell_id = _cell_id("B2", f"k{lag}", gain, f"eps{eps}", mechanism)
        swept = _gain_param(mechanism) is not None
        sc = _scenario(
            scenario_id="B2",
            name=(
                f"B2 lag x mechanism: {mechanism}, k={lag} (m={lag + 1}), "
                f"gain={gain:g}, epsilon={eps:g}, 50% capacity shock to {STANDARD_SECTOR}"
            ),
            mechanism=mechanism,
            seeds=SEEDS["B2"],
            lag=lag,
            price_elasticity=eps,
            shocks=(_capacity_shock(STANDARD_SECTOR),),
            mechanism_params=_mech_params(mechanism, gain),
            tags=("B2", "lag", "gain"),
        )
        cells.append(
            Cell(
                block="B2",
                cell_id=cell_id,
                mechanism=mechanism,
                scenario=sc,
                seeds=SEEDS["B2"],
                treatment={
                    "block": "B2",
                    "mechanism": mechanism,
                    "lag": int(lag),
                    "m": int(lag) + 1,
                    "gain": float(gain),
                    "gain_param": _gain_param(mechanism) or "",
                    "gain_axis": "swept" if swept else "fixed",
                    "price_elasticity": float(eps),
                    "shock_size": SHOCK_SIZE,
                    "shock_duration": SHOCK_DURATION,
                    "shock_sector": STANDARD_SECTOR,
                    "stock_ratio": STOCK_RATIO,
                    "substrate": "eurostat",
                },
                tech=_standard_tech_spec(),
                notes="standard environment: stock_ratio 0.10, sigma_demand 0.01, ar1 0.4, sigma_obs 0.02",
            )
        )
    return cells


def _build_b1b() -> list[Cell]:
    """B1b — no shock, no lag, only measurement and demand noise.

    The paper's thesis in its purest form (requested by the Lead as an amendment
    extension): with perfect information every institution reproduces the base
    year exactly, so *any* departure here is the information rent of the
    institution.  Environment: measurement noise 0.03, AR(1) demand noise
    (ar1 0.3, sigma 0.02), no shock, lag 0, 20 seeds.
    """
    cells: list[Cell] = []
    for mechanism in MECHANISMS:
        cell_id = _cell_id("B1b", mechanism)
        sc = _scenario(
            scenario_id="B1b",
            name=f"B1b information rent: {mechanism}, noise only (no shock, no lag)",
            mechanism=mechanism,
            seeds=SEEDS["B1b"],
            lag=0,
            noise=0.03,
            sigma=0.02,
            ar1=0.3,
            shocks=(),
            mechanism_params=_mech_params(mechanism, REFERENCE_GAIN),
            tags=("B1b", "information_rent"),
        )
        cells.append(
            Cell(
                block="B1b",
                cell_id=cell_id,
                mechanism=mechanism,
                scenario=sc,
                seeds=SEEDS["B1b"],
                treatment={
                    "block": "B1b",
                    "mechanism": mechanism,
                    "lag": 0,
                    "m": 1,
                    "gain": REFERENCE_GAIN,
                    "price_elasticity": STANDARD_ELASTICITY,
                    "sigma_demand": 0.02,
                    "sigma_obs": 0.03,
                    "shock_size": 1.0,
                    "shock_sector": None,
                    "stock_ratio": STOCK_RATIO,
                    "substrate": "eurostat",
                },
                tech=_standard_tech_spec(),
                notes="no shock, no lag: the information rent of the institution",
            )
        )
    return cells


def _build_b3(mechanisms: Sequence[str] = B3_MECHANISMS) -> list[Cell]:
    """B3 — one cell per (shocked sector, mechanism), 10 seeds each."""
    sectors = _sector_names()
    cells: list[Cell] = []
    for mechanism in mechanisms:
        for sector in sectors:
            cell_id = _cell_id("B3", sector, mechanism)
            sc = _scenario(
                scenario_id="B3",
                name=f"B3 propagation: 50% capacity shock to {sector}, {mechanism}",
                mechanism=mechanism,
                seeds=SEEDS["B3"],
                lag=2,
                shocks=(_capacity_shock(sector),),
                mechanism_params=_mech_params(mechanism, REFERENCE_GAIN),
                tags=("B3", "propagation"),
            )
            cells.append(
                Cell(
                    block="B3",
                    cell_id=cell_id,
                    mechanism=mechanism,
                    scenario=sc,
                    seeds=SEEDS["B3"],
                    treatment={
                        "block": "B3",
                        "mechanism": mechanism,
                        "lag": 2,
                        "m": 3,
                        "gain": REFERENCE_GAIN,
                        "price_elasticity": STANDARD_ELASTICITY,
                        "shock_size": SHOCK_SIZE,
                        "shock_duration": SHOCK_DURATION,
                        "shock_sector": sector,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": "eurostat",
                    },
                    tech=_standard_tech_spec(),
                    notes="network amplification: perturb_shock_name recorded in treatment",
                )
            )
    return cells


def _build_b4() -> list[Cell]:
    """B4 — rationing vs shadow price at 10/30/50% capacity losses."""
    cells: list[Cell] = []
    for mechanism in B4_MECHANISMS:
        for size in B4_SIZES:
            loss = int(round((1.0 - size) * 100))
            cell_id = _cell_id("B4", f"loss{loss}", mechanism)
            sc = _scenario(
                scenario_id="B4",
                name=f"B4 rationing vs shadow price: {loss}% capacity loss, {mechanism}",
                mechanism=mechanism,
                seeds=SEEDS["B4"],
                lag=2,
                shocks=(_capacity_shock(STANDARD_SECTOR, size=size),),
                mechanism_params=_mech_params(mechanism, REFERENCE_GAIN),
                tags=("B4", "shock_size"),
            )
            cells.append(
                Cell(
                    block="B4",
                    cell_id=cell_id,
                    mechanism=mechanism,
                    scenario=sc,
                    seeds=SEEDS["B4"],
                    treatment={
                        "block": "B4",
                        "mechanism": mechanism,
                        "lag": 2,
                        "m": 3,
                        "gain": REFERENCE_GAIN,
                        "price_elasticity": STANDARD_ELASTICITY,
                        "shock_size": float(size),
                        "capacity_loss": loss,
                        "shock_duration": SHOCK_DURATION,
                        "shock_sector": STANDARD_SECTOR,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": "eurostat",
                    },
                    tech=_standard_tech_spec(),
                    notes="same shock sector and duration; only the size changes",
                )
            )
    return cells


def _build_b5() -> list[Cell]:
    """B5 — dual track: plan share theta x lag."""
    theta_key = _theta_key() or "theta"
    cells: list[Cell] = []
    for lag in (0, 3):
        for theta in THETAS:
            cell_id = _cell_id("B5", f"k{lag}", f"theta{theta:g}")
            sc = _scenario(
                scenario_id="B5",
                name=f"B5 dual track: theta={theta:g}, k={lag} (m={lag + 1})",
                mechanism="hybrid_dual_track",
                seeds=SEEDS["B5"],
                lag=lag,
                shocks=(_capacity_shock(STANDARD_SECTOR),),
                mechanism_params=_mech_params(
                    "hybrid_dual_track", REFERENCE_GAIN, **{theta_key: float(theta)}
                ),
                tags=("B5", "dual_track"),
            )
            cells.append(
                Cell(
                    block="B5",
                    cell_id=cell_id,
                    mechanism="hybrid_dual_track",
                    scenario=sc,
                    seeds=SEEDS["B5"],
                    treatment={
                        "block": "B5",
                        "mechanism": "hybrid_dual_track",
                        "theta": float(theta),
                        "theta_param": theta_key,
                        "lag": int(lag),
                        "m": int(lag) + 1,
                        "gain": REFERENCE_GAIN,
                        "price_elasticity": STANDARD_ELASTICITY,
                        "shock_size": SHOCK_SIZE,
                        "shock_duration": SHOCK_DURATION,
                        "shock_sector": STANDARD_SECTOR,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": "eurostat",
                    },
                    tech=_standard_tech_spec(),
                    notes="theta=0 is the pure market pole, theta=1 the pure plan pole",
                )
            )
    return cells


def _build_b6() -> list[Cell]:
    """B6 — twelve robustness cells around the B2/B3 reference condition."""
    base = dict(
        mechanism="material_balance",
        lag=3,
        gain=REFERENCE_GAIN,
        eps=STANDARD_ELASTICITY,
        sigma=STANDARD_SIGMA_DEMAND,
        noise=STANDARD_SIGMA_OBS,
        substrate="eurostat",
        weights={"shortage": 1.0, "inventory": 0.1},
        tech=_standard_tech_spec(),
        notes="reference condition = B2 material_balance k=3, gain 0.30",
    )
    variants: list[dict[str, Any]] = [
        {"label": "reference", "note": "the reference condition every other cell is compared with"},
        {"label": "synthetic", "substrate": "synthetic", "note": "synthetic substrate"},
        {"label": "eta_low", "gain": 0.15, "note": "revision gain 0.15"},
        {"label": "eta_high", "gain": 0.60, "note": "revision gain 0.60"},
        {"label": "sigma_low", "sigma": 0.005, "note": "demand noise sigma 0.005"},
        {"label": "sigma_high", "sigma": 0.06, "note": "demand noise sigma 0.06"},
        {"label": "obs_noise_zero", "noise": 0.0, "note": "perfect measurement"},
        {"label": "obs_noise_high", "noise": 0.10, "note": "measurement noise 0.10"},
        {
            "label": "a_perturb_1",
            "tech": _standard_tech_spec(a_perturb=0.05, perturb_seed=11),
            "note": "5% log-normal perturbation of A, instance 1",
        },
        {
            "label": "a_perturb_2",
            "tech": _standard_tech_spec(a_perturb=0.05, perturb_seed=29),
            "note": "5% log-normal perturbation of A, instance 2",
        },
        {
            "label": "balance_rounds_0",
            "extra_params": {"balance_rounds": 0},
            "note": "planning effort: no internal consistency pass (a naive taut plan)",
        },
        {
            "label": "balance_rounds_6",
            "extra_params": {"balance_rounds": 6},
            "note": "planning effort: six internal consistency passes",
        },
        *[
            {
                "label": f"labour{int(slack * 100)}_{cond}",
                "labour_supply_factor": float(slack),
                "no_shock": True,
                "extra_params": dict(params),
                "note": (
                    "labour slack robustness on the B8 decomposition: the taut plan "
                    f"(gain 1.0, rounds 3) at {slack:.2f}x labour supply"
                    if cond == "taut"
                    else f"control {cond} at {slack:.2f}x labour supply"
                ),
            }
            for slack in (1.05, 1.15, 1.40, 2.00)
            for cond, params in (
                ("taut", {"revision_gain": 1.0, "balance_rounds": 3}),
                ("rev_only", {"revision_gain": 1.0, "balance_rounds": 0}),
                ("no_rev", {"revision_gain": 0.0, "balance_rounds": 0}),
            )
        ],
        {
            "label": "weights_low",
            "weights": {"shortage": 0.5, "inventory": 0.05},
            "note": "shortage and inventory penalties at -50%",
        },
        {
            "label": "weights_high",
            "weights": {"shortage": 1.5, "inventory": 0.15},
            "note": "shortage and inventory penalties at +50%",
        },
    ]
    tech_by_label = {"eurostat": _standard_tech_spec(), "synthetic": _standard_tech_spec(source="synthetic")}
    cells: list[Cell] = []
    for variant in variants:
        label = str(variant["label"])
        substrate = str(variant.get("substrate", "eurostat"))
        tech = dict(variant.get("tech") or tech_by_label[substrate])
        gain = float(variant.get("gain", REFERENCE_GAIN))
        no_shock = bool(variant.get("no_shock", False))
        labour_factor = variant.get("labour_supply_factor")
        cell_id = _cell_id("B6", label)
        sc = _scenario(
            scenario_id="B6",
            name=f"B6 robustness: {label}",
            mechanism="material_balance",
            seeds=SEEDS["B6"],
            lag=int(variant.get("lag", 0 if no_shock else base["lag"])),
            noise=float(variant.get("noise", 0.03 if no_shock else base["noise"])),
            sigma=float(variant.get("sigma", 0.02 if no_shock else base["sigma"])),
            ar1=float(variant.get("ar1", 0.3 if no_shock else 0.4)),
            price_elasticity=float(variant.get("eps", base["eps"])),
            shocks=() if no_shock else (_capacity_shock(STANDARD_SECTOR),),
            mechanism_params=_mech_params(
                "material_balance", gain, **dict(variant.get("extra_params") or {})
            ),
            labor_supply_factor=float(labour_factor) if labour_factor is not None else 1.15,
            objective=dict(variant.get("weights", base["weights"])),
            technology={"source": tech.get("source", "eurostat"), "year": tech.get("year", 2020),
                        "n_sectors": tech.get("n_sectors", 20)},
            tags=("B6", "robustness", label),
        )
        cells.append(
            Cell(
                block="B6",
                cell_id=cell_id,
                mechanism="material_balance",
                scenario=sc,
                seeds=SEEDS["B6"],
                treatment={
                    "block": "B6",
                    "mechanism": "material_balance",
                    "variant": label,
                    "lag": int(sc["info"]["lag"]),
                    "m": int(sc["info"]["lag"]) + 1,
                    "gain": gain,
                    "price_elasticity": float(sc["demand"]["price_elasticity"]),
                    "sigma_demand": float(sc["demand"]["sigma"]),
                    "sigma_obs": float(sc["info"]["noise"]),
                    "shortage_weight": float(sc["objective"]["shortage"]),
                    "inventory_weight": float(sc["objective"]["inventory"]),
                    "a_perturb": float(tech.get("a_perturb", 0.0)),
                    "labor_supply_factor": float(
                        labour_factor if labour_factor is not None else 1.15
                    ),
                    "shock_scale": 0.0 if no_shock else SHOCK_SIZE,
                    "shock_size": SHOCK_SIZE,
                    "shock_duration": SHOCK_DURATION,
                    "shock_sector": STANDARD_SECTOR,
                    "stock_ratio": STOCK_RATIO,
                    "substrate": str(tech.get("source", "eurostat")),
                },
                tech=tech,
                notes=str(variant["note"]),
            )
        )
    return cells


def _build_b8() -> list[Cell]:
    """B8 — decomposing material balance: revision gain x balancing rounds.

    The Lead's decomposition experiment (noise-only environment, no shock, lag 0):
    neither the proportional revision alone nor the balancing alone costs output;
    the loss lives in their interaction, because the balancing pass acts on a noisy
    stock estimate and the resulting asymmetry (an over-estimate is harmless —
    capacity binds from above — while an under-estimate cuts the plan inside the
    period) turns symmetric measurement noise into a systematic output loss.
    """
    cells: list[Cell] = []
    for gain in (0.0, 0.1, 0.3, 1.0):
        for rounds in (0, 1, 3):
            cell_id = _cell_id("B8", f"g{gain:g}", f"r{rounds}")
            sc = _scenario(
                scenario_id="B8",
                name=(
                    f"B8 material-balance decomposition: revision_gain={gain:g}, "
                    f"balance_rounds={rounds}"
                ),
                mechanism="material_balance",
                seeds=SEEDS["B8"],
                lag=0,
                noise=0.03,
                sigma=0.02,
                ar1=0.3,
                shocks=(),
                mechanism_params=_mech_params(
                    "material_balance", gain, balance_rounds=int(rounds)
                ),
                tags=("B8", "decomposition", "information_rent"),
            )
            cells.append(
                Cell(
                    block="B8",
                    cell_id=cell_id,
                    mechanism="material_balance",
                    scenario=sc,
                    seeds=SEEDS["B8"],
                    treatment={
                        "block": "B8",
                        "mechanism": "material_balance",
                        "gain": float(gain),
                        "revision_gain": float(gain),
                        "balance_rounds": int(rounds),
                        "lag": 0,
                        "m": 1,
                        "price_elasticity": STANDARD_ELASTICITY,
                        "sigma_demand": 0.02,
                        "sigma_obs": 0.03,
                        "shock_size": 1.0,
                        "shock_sector": None,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": "eurostat",
                    },
                    tech=_standard_tech_spec(),
                    notes="no shock, no lag: revision x balancing interaction only",
                )
            )
    return cells


def _build_b2l() -> list[Cell]:
    """B2L (exploratory) — production-first vs consumption-first priority ladder."""
    cells: list[Cell] = []
    for ladder in ("production_first", "consumption_first"):
        for lag in (0, 1, 2, 4):
            cell_id = _cell_id("B2L", ladder.replace("_first", ""), f"k{lag}")
            params = _mech_params("material_balance", REFERENCE_GAIN, ladder=ladder)
            sc = _scenario(
                scenario_id="B2L",
                name=f"B2L ladder: {ladder}, k={lag}",
                mechanism="material_balance",
                seeds=SEEDS["B2L"],
                lag=lag,
                shocks=(_capacity_shock(STANDARD_SECTOR),),
                mechanism_params=params,
                tags=("B2L", "ladder", "exploratory"),
            )
            cells.append(
                Cell(
                    block="B2L",
                    cell_id=cell_id,
                    mechanism="material_balance",
                    scenario=sc,
                    seeds=SEEDS["B2L"],
                    treatment={
                        "block": "B2L",
                        "mechanism": "material_balance",
                        "ladder": ladder,
                        "lag": int(lag),
                        "m": int(lag) + 1,
                        "gain": REFERENCE_GAIN,
                        "price_elasticity": STANDARD_ELASTICITY,
                        "shock_size": SHOCK_SIZE,
                        "shock_duration": SHOCK_DURATION,
                        "shock_sector": STANDARD_SECTOR,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": "eurostat",
                    },
                    tech=_standard_tech_spec(),
                    notes="exploratory: the ladder is a treatment, not a default (ARCHITECTURE §5)",
                )
            )
    return cells


def _build_b7() -> list[Cell]:
    """B7 — price-response probe: measure ``b_p = -ds/dlog p`` sector by sector.

    The mechanism is ``tatonnement`` with the price gain switched off, so prices do
    not move by themselves; a permanent +-5% price offset is injected into one
    good's price at the start of the run, and the shortage response of *the same
    good* is read from the deterministic trajectory under a permanent 50% capacity
    shock to that good.  Matching the probed good to the shocked good is what makes
    the derivative the one the feedback loop actually uses: ``b_p,j = -ds_j/dlog p_j``.

    This is the notes' recipe (``docs/notes_delay_theorem.md`` §5.1) adapted to a
    price instrument, and it is what turns H1 into a theory-vs-measurement
    comparison ``price_gain*(m) = g*(m)/b_p`` instead of a story about amplitude.
    """
    cells: list[Cell] = []
    # 0.4 is the standard environment of the mechanism comparison (B2/B3/B4/B5), so
    # b_p must be measured there too; 0.2 and 2.0 bracket it and 0.8 is the
    # price-loop regime the Lead asked for.
    for eps in (0.2, STANDARD_ELASTICITY, HIGH_ELASTICITY, 2.0):
        for sector in _sector_names():
            for delta in (-0.05, 0.05):
                cell_id = _cell_id("B7", f"eps{eps}", sector, f"d{delta:+.2f}")
                sc = _scenario(
                    scenario_id="B7",
                    name=(
                        f"B7 gain probe: price offset {delta:+.0%} on {sector} "
                        f"with {sector} capacity shocked, epsilon={eps:g}"
                    ),
                    mechanism="tatonnement",
                    seeds=SEEDS["B7"],
                    lag=0,
                    noise=0.0,
                    sigma=0.0,
                    ar1=0.0,
                    price_elasticity=eps,
                    shocks=(_capacity_shock(sector, duration=0),),
                    mechanism_params=_mech_params("tatonnement", 0.0),
                    tags=("B7", "probe", "gain"),
                )
                cells.append(
                    Cell(
                        block="B7",
                        cell_id=cell_id,
                        mechanism="tatonnement",
                        scenario=sc,
                        seeds=SEEDS["B7"],
                        treatment={
                            "block": "B7",
                            "mechanism": "tatonnement",
                            "price_elasticity": float(eps),
                            "sector": sector,
                            "delta": float(delta),
                            "gain": 0.0,
                            "shock_size": SHOCK_SIZE,
                            "shock_duration": 0,
                            "shock_sector": sector,
                            "stock_ratio": STOCK_RATIO,
                            "substrate": "eurostat",
                        },
                        tech=_standard_tech_spec(),
                        probe={"kind": "price_offset", "sector": sector, "delta": float(delta)},
                        notes="deterministic probe; no measurement noise",
                    )
                )
    return cells


def _build_p2() -> list[Cell]:
    """P2 — target-response probe: sign and size of ``ds/dlog T`` (H1').

    ``material_balance`` with the revision gain off and the carried plan targets
    shifted down by 5% / 10% for one sector at a time.  If the plan-revision loop
    were the negative feedback of Proposition 1, the measured gain would be
    positive; the pilot says it is negative (Amendment 1).
    """
    cells: list[Cell] = []
    for sector in _sector_names():
        for delta in (-0.05, -0.10):
            cell_id = _cell_id("P2", sector, f"d{delta:+.2f}")
            sc = _scenario(
                scenario_id="P2",
                name=f"P2 sign probe: target offset {delta:+.0%} on {sector}",
                mechanism="material_balance",
                seeds=SEEDS["P2"],
                lag=0,
                noise=0.0,
                sigma=0.0,
                ar1=0.0,
                price_elasticity=0.0,
                shocks=(),
                mechanism_params=_mech_params("material_balance", 0.0, balance_rounds=3),
                tags=("P2", "probe", "gain"),
            )
            cells.append(
                Cell(
                    block="P2",
                    cell_id=cell_id,
                    mechanism="material_balance",
                    scenario=sc,
                    seeds=SEEDS["P2"],
                    treatment={
                        "block": "P2",
                        "mechanism": "material_balance",
                        "sector": sector,
                        "delta": float(delta),
                        "gain": 0.0,
                        "price_elasticity": 0.0,
                        "shock_size": 1.0,
                        "shock_sector": None,
                        "stock_ratio": STOCK_RATIO,
                        "substrate": "eurostat",
                    },
                    tech=_standard_tech_spec(),
                    probe={"kind": "target_offset", "sector": sector, "delta": float(delta)},
                    notes="deterministic probe; no shock, no noise",
                )
            )
    return cells


def _sector_names() -> tuple[str, ...]:
    """Sector codes of the 20-sector Eurostat substrate, without loading data.

    The codes are also present in ``data/processed/eurostat_2020_n20_meta.json``;
    the loader is tried first so that the grid follows the data layer if the
    aggregation changes, with a hard-coded fallback so that the grid can be built
    on a machine where the artifacts are absent (the runner will fail loudly on
    the substrate instead).
    """
    try:
        from planlab.data import read_substrate_meta

        meta = read_substrate_meta("eurostat_2020_n20")
        sectors = tuple(str(s) for s in meta["sectors"])
        if len(sectors) == 20:
            return sectors
    except Exception:  # pragma: no cover - artifact-free checkout
        pass
    return tuple(f"c{i:02d}" for i in range(1, 21))


_BLOCK_BUILDERS = {
    "B1": lambda: _build_b1(),
    "B1b": lambda: _build_b1b(),
    "B2": lambda: _build_b2(),
    "B3": lambda: _build_b3(),
    "B4": lambda: _build_b4(),
    "B5": lambda: _build_b5(),
    "B6": lambda: _build_b6(),
    "B8": lambda: _build_b8(),
    "B2L": lambda: _build_b2l(),
    "B7": lambda: _build_b7(),
    "P2": lambda: _build_p2(),
}

#: block order used by ``scripts/run_all.py`` and ``scripts/reproduce.sh``
BLOCK_ORDER: tuple[str, ...] = (
    "B1", "B1b", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B2L", "P2"
)

#: the pre-registered blocks, in protocol order
BLOCKS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6")

#: B7 is the Lead's requested addition to the pre-registered design: the
#: price-response probe that measures ``b_p`` and therefore makes the H1
#: boundary a theory-vs-measurement comparison.  B2L (priority ladder) and P2
#: (the ``b_T`` sign probe behind H1') are exploratory as well.

#: blocks that are additions to the pre-registered design (recorded as such)
EXPLORATORY_BLOCKS: tuple[str, ...] = ("B1b", "B8", "B2L", "P2")


def build_cells(block: str) -> list[Cell]:
    """All cells of one block, in a deterministic order.

    Args:
        block: one of :data:`BLOCK_ORDER`.

    Returns:
        The list of :class:`Cell`, always the same for the same ``block``.

    Raises:
        KeyError: if ``block`` is unknown.
    """
    try:
        return _BLOCK_BUILDERS[block]()
    except KeyError as exc:
        raise KeyError(f"unknown block {block!r}; known: {', '.join(BLOCK_ORDER)}") from exc


def build_grid(blocks: Sequence[str] | None = None) -> list[Cell]:
    """Cells for ``blocks`` (default: every block, in :data:`BLOCK_ORDER`)."""
    order = list(BLOCK_ORDER if blocks is None else blocks)
    cells: list[Cell] = []
    for block in order:
        cells.extend(build_cells(block))
    return cells


def block_run_count(block: str) -> int:
    """Planned number of runs (cells x seeds) for one block."""
    return sum(len(cell.seeds) for cell in build_cells(block))


def planned_run_count(blocks: Sequence[str] | None = None) -> dict[str, int]:
    """Planned run count per block plus ``total``."""
    order = list(BLOCK_ORDER if blocks is None else blocks)
    out = {block: block_run_count(block) for block in order}
    out["total"] = sum(out.values())
    return out


# ---------------------------------------------------------------------------
# manifest helper
# ---------------------------------------------------------------------------
def grid_fingerprint(cells: Sequence[Cell] | None = None) -> str:
    """SHA-256 prefix over the sorted cell hashes: identifies the frozen grid."""
    items = sorted(cell.hash() for cell in (build_grid() if cells is None else cells))
    return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()[:16]
