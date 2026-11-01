"""Core tests: metrics, scenarios and shock processes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from planlab.core import metrics as M
from planlab.core.io_model import toy_technology
from planlab.core.scenarios import (
    DemandProcess,
    InfoRegime,
    ObjectiveWeights,
    Scenario,
    load_scenario,
    scenario_from_dict,
    scenario_hash,
)
from planlab.core.shocks import Shock, ShockPath, parse_shocks


# --------------------------------------------------------------------- metrics
def test_welfare_is_finite_at_zero_consumption():
    tech = toy_technology(4)
    w = M.welfare(tech, np.zeros(4))
    assert math.isfinite(w)


def test_welfare_increases_with_consumption():
    tech = toy_technology(4)
    low = M.welfare(tech, tech.base_final * 0.5)
    high = M.welfare(tech, tech.base_final * 1.5)
    assert high > low


def test_essentials_weighted_more():
    tech = toy_technology(4)
    w = tech.consumption_weights.copy()
    assert w[tech.essentials > 0.5].mean() > w[tech.essentials <= 0.5].mean()


def test_detrended_deviation_removes_a_linear_trend():
    t = np.arange(80)
    dev = M.detrended_deviation(2.0 + 0.1 * t, window=21)
    # a centred moving average cannot be unbiased at the edges, so check the interior
    assert np.max(np.abs(dev[15:65])) < 1e-9


def test_autocorrelation_of_sine_wave():
    t = np.arange(200)
    period = 10.0
    x = np.sin(2 * np.pi * t / period)
    ac = M.autocorrelation(x, max_lag=4)
    # lag 10 should be ~1 for a 10-period cycle; the first four are positive for cos
    assert abs(ac[0] - math.cos(2 * math.pi / period)) < 0.05


def test_dominant_period_detects_cycle():
    t = np.arange(120)
    dev = np.sin(2 * np.pi * t / 12.0)
    per = M._dominant_period(dev)
    assert abs(per - 12.0) < 1.0


def test_period_metrics_shortage_rate():
    tech = toy_technology(3)
    x = tech.base_output.copy()
    Z = tech.A * x.reshape(1, -1)
    demand = tech.base_final.copy()
    C = demand * 0.5
    unmet = demand - C
    pm = M.period_metrics(
        tech=tech, x=x, Z=Z, C=C, G=np.zeros(3), invest=np.zeros(3),
        inventory_end=np.zeros(3), demand=demand, unmet=unmet,
        price=np.ones(3), capacity=x, plan_targets=x, iterations=1,
    )
    assert abs(pm["shortage_rate"] - 0.5) < 1e-9
    assert pm["plan_fulfilment"] == pytest.approx(1.0)


def test_penalised_welfare_decreases_with_penalties():
    pm = {"welfare": 1.0, "shortage_rate": 0.2, "inventory_ratio": 0.1,
          "plan_deviation": 0.1, "consumption_dispersion": 0.1, "adjustment_cost": 0.1}
    heavy = M.penalised_welfare(pm, {"shortage": 5.0})
    light = M.penalised_welfare(pm, {"shortage": 0.5})
    assert heavy < light


# ------------------------------------------------------------------- scenarios
def test_scenario_roundtrip_and_hash_stability():
    data = {
        "id": "rt", "mechanism": "material_balance", "technology": {"toy": 6},
        "periods": 20, "seeds": [0, 1], "burn_in": 3,
    }
    sc = scenario_from_dict(data)
    again = scenario_from_dict(sc.to_dict())
    assert sc == again or sc.to_dict() == again.to_dict()
    assert scenario_hash(sc) == scenario_hash(again)


def test_scenario_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown scenario keys"):
        scenario_from_dict({
            "id": "x", "mechanism": "material_balance",
            "technology": {"toy": 6}, "nonsense": 1,
        })


def test_scenario_requires_core_keys():
    with pytest.raises(ValueError):
        scenario_from_dict({"mechanism": "material_balance", "technology": {"toy": 6}})


def test_info_regime_validates():
    with pytest.raises(ValueError):
        InfoRegime(lag=-1)
    with pytest.raises(ValueError):
        InfoRegime(noise=-0.1)


def test_demand_process_validates():
    with pytest.raises(ValueError):
        DemandProcess(ar1=1.0)
    with pytest.raises(ValueError):
        DemandProcess(sigma=-1.0)


def test_objective_weights_dict():
    ow = ObjectiveWeights(shortage=2.0)
    assert ow.as_dict()["shortage"] == 2.0
    assert set(ow.as_dict()) == {"shortage", "inventory", "plan_deviation",
                                 "inequality", "adjustment"}


def test_scenario_file_roundtrip(tmp_path):
    payload = {
        "id": "file_sc", "name": "file", "mechanism": "material_balance",
        "technology": {"toy": 6}, "periods": 12, "seeds": [3], "burn_in": 2,
        "shocks": [{"type": "capacity", "sector": "s01", "start": 4,
                    "duration": 2, "size": 0.8}],
    }
    path = tmp_path / "file_sc.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    sc = load_scenario(path)
    assert sc.id == "file_sc"
    assert len(sc.shocks) == 1
    assert sc.shocks[0].size == 0.8


def test_scenario_id_must_match_filename(tmp_path):
    path = tmp_path / "wrong_name.yaml"
    path.write_text(yaml.safe_dump({
        "id": "other", "mechanism": "material_balance", "technology": {"toy": 6},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match filename"):
        load_scenario(path)


def test_scenario_validation_of_burn_in_and_seeds():
    with pytest.raises(ValueError):
        Scenario(id="a", mechanism="m", technology={}, periods=10, burn_in=10)
    with pytest.raises(ValueError):
        Scenario(id="a", mechanism="m", technology={}, seeds=())


# ---------------------------------------------------------------------- shocks
def test_shock_factor_windows():
    s = Shock(type="capacity", start=5, duration=3, size=0.5, sector=None, ramp=0)
    assert s.factor(4) == 1.0
    assert s.factor(5) == 0.5
    assert s.factor(7) == 0.5
    assert s.factor(8) == 1.0


def test_shock_ramp_recovers_gradually():
    s = Shock(type="capacity", start=0, duration=2, size=0.2, sector=None, ramp=4)
    vals = [s.factor(t) for t in range(7)]
    assert vals[0] == pytest.approx(0.2) and vals[1] == pytest.approx(0.2)
    assert vals[2] < vals[3] < vals[4] < vals[5] <= 1.0
    assert vals[-1] == pytest.approx(1.0)


def test_shock_path_sector_targeting():
    path = ShockPath(parse_shocks([
        {"type": "capacity", "sector": "b", "start": 0, "duration": 1, "size": 0.4}
    ]), ["a", "b", "c"])
    f = path.capacity_factor(0)
    assert np.allclose(f, [1.0, 0.4, 1.0])
    assert path.peak_capacity_loss() == pytest.approx(0.6)


def test_shock_path_rejects_unknown_sector():
    with pytest.raises(ValueError, match="unknown sector"):
        ShockPath(parse_shocks([
            {"type": "capacity", "sector": "zz", "start": 0, "duration": 1, "size": 0.5}
        ]), ["a", "b"])


def test_parse_shocks_rejects_bad_type():
    with pytest.raises(ValueError):
        parse_shocks([{"type": "meteor", "start": 0, "duration": 1, "size": 1.0}])
