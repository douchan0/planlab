"""PlanLab command line interface.

    python -m planlab.cli run scenarios/baseline_mb_k0.yaml
    python -m planlab.cli list-scenarios
    python -m planlab.cli list-mechanisms
    python -m planlab.cli show-technology --source eurostat --year 2020 --n 20
    python -m planlab.cli verify

The CLI is deliberately thin: it exists so that a single scenario can be run and
inspected standalone, which is what "reproducible" means at the level of an
individual experimental cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["main", "build_parser"]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.4f}"
    return str(value)


def cmd_run(args: argparse.Namespace) -> int:
    from planlab.core.engine import run_scenario
    from planlab.core.scenarios import load_scenario

    scenario = load_scenario(args.scenario)
    if args.seed is not None:
        scenario = scenario.replace(seeds=[args.seed])
    result = run_scenario(scenario, seed=args.seed)
    metrics = result.run_metrics

    if args.json:
        payload = {
            "scenario": scenario.id,
            "mechanism": result.mechanism_name,
            "seed": result.seed,
            "run_metrics": {k: (None if v != v else float(v)) for k, v in metrics.items()},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"scenario : {scenario.id}")
    print(f"mechanism: {result.mechanism_name}   seed={result.seed}   periods={result.periods}")
    print("-" * 62)
    keys = [
        "mean_shortage_rate", "max_shortage_rate", "mean_shortage_essentials",
        "cum_welfare", "mean_welfare", "mean_plan_fulfilment", "mean_plan_deviation",
        "mean_inventory_ratio", "mean_capital_utilisation", "oscillation_amplitude",
        "mean_price_volatility", "mean_iterations", "converged_all",
        "shock_peak_shortage", "shock_amplification", "recovery_half_life",
    ]
    for k in keys:
        if k in metrics:
            print(f"  {k:28s} {_fmt(metrics[k])}")
    if args.trace:
        print("-" * 62)
        print(f"{'t':>3} {'shortage':>9} {'welfare':>10} {'x/base':>8} {'iter':>5}")
        base = result.technology.base_output
        for r in result.records:
            print(
                f"{r.t:3d} {_fmt(r.metrics['shortage_rate']):>9} "
                f"{_fmt(r.metrics['welfare']):>10} "
                f"{_fmt(float(np.mean(r.x / base))):>8} {r.iterations:5d}"
            )
    return 0


def cmd_list_scenarios(args: argparse.Namespace) -> int:
    from planlab.core.scenarios import list_scenarios, load_scenario

    paths = list_scenarios(args.directory)
    if not paths:
        print("no scenarios found")
        return 0
    for p in paths:
        try:
            sc = load_scenario(p)
            print(f"{sc.id:38s} {sc.mechanism:20s} periods={sc.periods:<4d} seeds={len(sc.seeds)}")
        except Exception as exc:  # pragma: no cover - reporting path
            print(f"{p.stem:38s} INVALID: {exc}")
    return 0


def cmd_list_mechanisms(_args: argparse.Namespace) -> int:
    from planlab.mechanisms import available_mechanisms, get_mechanism
    from planlab.core.io_model import toy_technology

    tech = toy_technology(3)
    for name in available_mechanisms():
        mech = get_mechanism(name, tech, {})
        doc = (type(mech).__doc__ or "").strip().splitlines()
        print(f"{name:20s} {doc[0] if doc else ''}")
    return 0


def cmd_show_technology(args: argparse.Namespace) -> int:
    from planlab.core.io_model import toy_technology
    from planlab.data import load_technology

    if args.toy:
        tech = toy_technology(args.toy)
    else:
        tech = load_technology(args.source, args.year, args.n)
    print(f"source      : {tech.meta.get('source', 'unknown')}")
    print(f"sectors     : {tech.n}  ({', '.join(tech.sectors[:8])}{' ...' if tech.n > 8 else ''})")
    print(f"total output: {float(tech.base_output.sum()):.1f}  (base units of the source table)")
    print(f"column sums : min={float(tech.A.sum(axis=0).min()):.4f} "
          f"max={float(tech.A.sum(axis=0).max()):.4f}  (must be < 1)")
    resid = tech.base_output - tech.A @ tech.base_output - tech.base_final
    scale = max(1.0, float(np.max(np.abs(tech.base_output))))
    print(f"I-O identity: max|residual| = {float(np.max(np.abs(resid))):.3e} "
          f"({float(np.max(np.abs(resid))) / scale:.2e} relative; CSV rounding)")
    print(f"essentials  : {[s for s, e in zip(tech.sectors, tech.essentials) if e > 0.5]}")
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Quick self-check of the mathematical invariants (not the full test suite)."""
    from planlab.core.engine import run_scenario
    from planlab.core.io_model import toy_technology
    from planlab.core.scenarios import scenario_from_dict
    from planlab.mechanisms import available_mechanisms

    tech = toy_technology(6)
    failures = 0
    for name in available_mechanisms():
        sc = scenario_from_dict(
            {
                "id": f"verify_{name}",
                "mechanism": name,
                "technology": {"toy": 6},
                "periods": 20,
                "seeds": [0],
                "burn_in": 5,
            }
        )
        try:
            res = run_scenario(sc, technology=tech)
        except Exception as exc:
            print(f"[FAIL] {name}: raised {type(exc).__name__}: {exc}")
            failures += 1
            continue
        ratio = float(np.mean(res.records[-1].x / tech.base_output))
        shortage = res.run_metrics["mean_shortage_rate"]
        ok = abs(ratio - 1.0) <= 0.05 and shortage <= 0.02
        print(f"[{'OK ' if ok else 'FAIL'}] {name:20s} x/base={ratio:.4f} shortage={shortage:.4f}")
        failures += 0 if ok else 1
    print(f"\n{len(available_mechanisms()) - failures}/{len(available_mechanisms())} mechanisms pass")
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="planlab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a single scenario file")
    r.add_argument("scenario", type=Path)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--json", action="store_true", help="machine-readable output")
    r.add_argument("--trace", action="store_true", help="print a per-period trace")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("list-scenarios", help="list scenario files")
    ls.add_argument("--directory", type=Path, default=None)
    ls.set_defaults(func=cmd_list_scenarios)

    lm = sub.add_parser("list-mechanisms", help="list registered mechanisms")
    lm.set_defaults(func=cmd_list_mechanisms)

    st = sub.add_parser("show-technology", help="inspect a technology substrate")
    st.add_argument("--source", default="eurostat")
    st.add_argument("--year", type=int, default=2020)
    st.add_argument("--n", type=int, default=20)
    st.add_argument("--toy", type=int, default=0, help="use the toy fixture with N sectors")
    st.set_defaults(func=cmd_show_technology)

    v = sub.add_parser("verify", help="quick invariant check across mechanisms")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
