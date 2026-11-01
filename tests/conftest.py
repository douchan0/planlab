"""Shared pytest fixtures.

Everything here is deterministic: a test that fails must fail reproducibly.
"""

from __future__ import annotations

import numpy as np
import pytest

from planlab.core.engine import run_scenario
from planlab.core.io_model import toy_technology
from planlab.core.scenarios import Scenario, scenario_from_dict


@pytest.fixture(scope="session")
def toy_tech():
    """A small, productive, exactly balanced 6-sector technology (test fixture only)."""
    return toy_technology(6)


@pytest.fixture(scope="session")
def toy_tech_big():
    return toy_technology(12, seed=999)


def make_scenario(mechanism: str = "material_balance", **overrides) -> Scenario:
    """Build a scenario dict with sensible defaults, applying overrides."""
    base = {
        "id": overrides.pop("id", f"test_{mechanism}"),
        "mechanism": mechanism,
        "technology": {"toy": 6},
        "periods": 30,
        "seeds": [0],
        "burn_in": 5,
    }
    base.update(overrides)
    return scenario_from_dict(base)


@pytest.fixture
def calm_scenario():
    """No shock, no noise, no lag: the base year must be a fixed point."""
    return make_scenario("material_balance", id="calm", info={"lag": 0, "noise": 0.0})


@pytest.fixture
def shock_scenario():
    return make_scenario(
        "material_balance",
        id="shock40",
        periods=45,
        shocks=[{"type": "capacity", "sector": "s02", "start": 10, "duration": 6, "size": 0.6}],
    )


@pytest.fixture
def run_calm(toy_tech):
    def _run(mech: str = "material_balance", **overrides):
        sc = make_scenario(mech, **overrides)
        return run_scenario(sc, technology=toy_tech)

    return _run


@pytest.fixture
def run_shock(toy_tech):
    def _run(mech: str = "material_balance", size: float = 0.6, sector: str = "s02", **overrides):
        sc = make_scenario(
            mech,
            id=f"shock_{mech}",
            periods=45,
            shocks=[{"type": "capacity", "sector": sector, "start": 10, "duration": 6, "size": size}],
            **overrides,
        )
        return run_scenario(sc, technology=toy_tech)

    return _run


def balance_residual(record) -> float:
    """Canonical audit helper (re-exported so tests share one definition)."""
    from planlab.core.engine import balance_residual as _canonical

    return _canonical(record)
