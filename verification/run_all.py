"""Run every independent verification check and write verification/report.md.

Exit codes
----------
0   every check PASSED
1   at least one check FAILED (a finding that must not be silently shipped)
2   no failures, but at least one check was SKIPPED because its input
    (results/, results/tables/tests.csv, ...) did not exist yet: the audit is
    INCOMPLETE and this must not be mistaken for a pass.

Usage
-----
    cd planlab && source scripts/env.sh && .venv/bin/python verification/run_all.py
    ... verification/run_all.py --only theorem,engine      # subset
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
import traceback

from .common import RESULTS, ROOT, CheckResult, rel

CHECK_MODULES = (
    ("theorem", "check1_theorem", "Independent re-derivation of Proposition 1"),
    ("engine", "check2_engine", "Independent one-period implementation vs the engine"),
    ("accounting", "check3_accounting", "Accounting audit over the experiment records"),
    ("reproducibility", "check4_repro", "Reproducibility audit of the stored results"),
    ("statistics", "check5_stats", "Statistical audit (Wilcoxon/bootstrap/Holm/H4)"),
    ("claims", "check6_claims", "Mechanism claims for the headline interaction"),
)

#: Findings this verifier reported to the Lead, with the date they were reported.
#: Kept here (not in a mutable log) so that the record survives re-runs.
FINDINGS = [
    {
        "id": "F1",
        "date": "2026-11-01",
        "severity": "error (documentation)",
        "where": "docs/notes_delay_theorem.md (lines 50-51 and the table at 163-170)",
        "finding": "The numeric list and the table shifted the stability boundary by one lag "
                   "index against the note's own formula: it gave g*(1)=2, g*(2)=1, g*(3)=0.618, "
                   "g*(4)=0.445, whereas g*(m)=2 sin(pi/(2(2m+1))) gives g*(1)=1, g*(2)=0.618034, "
                   "g*(3)=0.445042 and 2 is g*(0).",
        "evidence": "check 1 (three independent routes: Schur-Cohn bisection, companion-matrix "
                    "spectral radius, time-domain growth exponent), m=0..30.",
        "resolution": "FIXED by the Lead the same day; the note now agrees with the paper and "
                      "records that the correction came from this check.",
    },
    {
        "id": "F2",
        "date": "2026-11-01",
        "severity": "error (documentation)",
        "where": "docs/notes_delay_theorem.md line 105",
        "finding": "\"Every period of delay more than halves the admissible gain\" is false: "
                   "2 -> 1 halves only at the first step; 1 -> 0.618 is a 38% cut and the ratios "
                   "approach 1 (g*(m) ~ pi/(2m+1)). The bounded object is m*g*(m) <= pi/2.",
        "evidence": "check 1 numbers for m=0..30; the paper already stated the correct version.",
        "resolution": "FIXED by the Lead the same day.",
    },
    {
        "id": "F3",
        "date": "2026-11-01",
        "severity": "gap in proof (not an error)",
        "where": "docs/notes_delay_theorem.md, proof of Proposition 1 (lines 84-90 of the original)",
        "finding": "The proof asserted that j=0 gives the smallest gain without ruling out j>=1, "
                   "for which theta_j > pi and sin(theta/2) decreases. The statement is true "
                   "(g_j = 2 sin(pi(1+4j)/(2(2m+1))) >= g_0 for 0<=j<=m, with equality also at "
                   "j=m, the conjugate pair), but the note did not show it.",
        "evidence": "check 1: the unit-circle identity and the root-free boundary agree with the "
                    "closed form for m=0..30; the paper's Section 4 already contains the correct "
                    "argument.",
        "resolution": "CLOSED by the Lead with the paper's x_m in [x_0, pi-x_0] argument.",
    },
    {
        "id": "F4",
        "date": "2026-11-01",
        "severity": "error (scientific explanation)",
        "where": "Lead's proposed explanation of the material_balance gain x rounds interaction",
        "finding": "The interaction is real (gain 0 / rounds 3 -> 1.0000; gain 1 / rounds 0 -> "
                   "1.0000; gain 1 / rounds 3 -> 0.9089 +- 0.0073), but the stated cause — "
                   "feasible_output being asymmetric in the *inventory* estimate — is refuted in "
                   "the regime that produces the loss: R(0.9 r) = R(r) = R(1.1 r) to 0.0e+00 in "
                   "every one of the mechanism's balancing calls (max goods-balance tightness "
                   "0.25; labour used/L = 0.99-1.00), removing the inventory estimate entirely "
                   "(r_est = 0) does not change the loss (0.9083 vs 0.9087), stock_ratio 0 and "
                   "0.5 do not change it (0.9061 / 0.9110), while labour supply x3 removes it "
                   "completely (1.0000). The asymmetry is real only at the knife-edge "
                   "goods-balance frontier (0 gain / -8.7% loss), where the mechanism never "
                   "operates. The correct mechanism is the one-sided projection onto the "
                   "LABOUR-constrained feasible set, and the result is conditional on "
                   "labor_supply_factor = 1.15.",
        "evidence": "check 6 (6a-6d); replicated independently by the Lead (factor 1.15 -> 0.8698, "
                    "1.40 -> 0.9922, 2.0 -> 1.0000).",
        "resolution": "OPEN at the time of writing: the paper's mechanism section was being "
                      "rewritten to the labour-projection version and labor_supply_factor was "
                      "added to B6 as a treatment. Re-run check 6 after the rewrite.",
    },
    {
        "id": "F5",
        "date": "2026-11-01",
        "severity": "bug (results artefacts, not the model)",
        "where": "results/<block>/audit_records.npz",
        "finding": "For the 6-sector cells the runner writes `A` FLATTENED into a fixed 20x20 "
                   "buffer (the 36 finite entries are the row-major 6x6 matrix, the other 364 "
                   "are NaN), while x/Z/... are NaN-padded but correctly placed and `sectors` is "
                   "empty. Any naive reader gets a NaN-filled A for toy cells.",
        "evidence": "check 3 loader: A.shape == (20,20) with 36 finite entries for a record whose "
                    "n_sectors == 6; reshaping the finite entries reproduces toy_technology(6).A "
                    "exactly (max |diff| = 0.0).",
        "resolution": "WORKED AROUND in check 3 (recover by reshaping); reported to the Lead and "
                      "to the experiments layer. Fix recommended before the final grid.",
    },
    {
        "id": "F6",
        "date": "2026-11-01",
        "severity": "bug (provenance)",
        "where": "results/<block>/cells.json",
        "finding": "The 6-sector B1 cells record scenario.technology = {source: eurostat, year: "
                   "2020, n_sectors: 20} while the authoritative cell technology is {toy: 6}. The "
                   "recorded scenario therefore does not reproduce the cell (a re-run from the "
                   "scenario dict alone produces 20-sector arrays), and the manifest's "
                   "scenario_hashes hash the cell dict rather than the scenario.",
        "evidence": "check 4: re-running the recorded scenario with cell.tech gives bit-identical "
                    "results, with scenario.technology it does not (0.436 relative difference in "
                    "x).",
        "resolution": "REPORTED to the Lead and the experiments layer; check 4 resolves the "
                      "technology from cell.tech and notes the disagreement.",
    },
]

#: per-check "what this does not prove" text
LIMITS = {
    "theorem": (
        "This verifies a linear difference equation. It does not prove that the simulated "
        "economy follows that linear loop: the reduction from the model to u_{t+1} = u_t - g "
        "u_{t-m} is a local linearisation whose gain g is an equilibrium object that must be "
        "measured numerically, and the model's nonlinearities (rationing, short-side programme, "
        "inventory) can change both the boundary and the period. Agreement between the "
        "simulation and the formula is a hypothesis test, not an identity."
    ),
    "engine": (
        "I implemented the *published* algorithm, so agreement shows that the code does what the "
        "specification says; it does not show that the specification is the right model. The "
        "per-period comparison also cannot detect a bias that is present identically in both "
        "my implementation and the engine's (e.g. a shared but wrong definition of a claim row)."
    ),
    "accounting": (
        "The balance identity is imposed by construction inside the engine (inventory is the "
        "residual), so a leak that moves value *between* categories while keeping the total will "
        "pass the identity. The audit therefore also recomputes the published metrics from the "
        "raw arrays and checks feasibility and the priority ladder, but it cannot prove that the "
        "categories themselves are economically the right ones. It also only covers the sampled "
        "records: a bug confined to un-sampled periods or un-sampled cells would survive."
    ),
    "reproducibility": (
        "Reproducibility is not validity: it shows that the stored numbers follow from the "
        "recorded inputs with the current code, not that the model is a good description of any "
        "economy. It also cannot detect a wrong result that is deterministically wrong, and the "
        "re-run uses the same underlying libraries, so a numerical bug inside numpy/scipy would "
        "reproduce identically."
    ),
    "statistics": (
        "I can recompute the test statistics and their multiplicity correction; I cannot verify "
        "that the pre-registered test was the right one for the question, that the run-level "
        "unit of analysis was respected in every table, or that the pre-registration was not "
        "silently narrowed after the results were seen. The permutation test assumes "
        "exchangeability of the shocked sectors, which is exact under the null only if the "
        "shock assignment is exchangeable — with a fixed 20-sector design that is the natural "
        "null but it does not model dependence through common shocks."
    ),
    "claims": (
        "This checks the claims I was asked to check, on one substrate and one calibration. It "
        "does not establish that the labour-projection mechanism is the only cause of the "
        "interaction (only that it is the binding margin in the tested configuration), nor that "
        "the interaction is economically meaningful outside the tested grid of gains, rounds and "
        "labour supply factors."
    ),
}


def _git() -> dict:
    out = {"commit": "", "dirty": None, "diff_stat": ""}
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        out["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip())
        out["diff_stat"] = subprocess.run(
            ["git", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()[-500:]
    except Exception:                                      # pragma: no cover
        pass
    return out


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _section(res: CheckResult) -> str:
    lines = [f"## Check {res.key} — {res.title}", ""]
    lines.append(f"**Status: {res.status}**")
    lines.append("")
    if res.summary:
        lines.append(res.summary)
        lines.append("")
    if res.numbers:
        lines.append("Measured:")
        lines.append("")
        for key, value in res.numbers.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                flat = ", ".join(f"{k}={_fmt(v)}" for k, v in list(value.items())[:14])
                lines.append(f"- `{key}`: {flat}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"- `{key}`:")
                for entry in value[:20]:
                    cells = ", ".join(f"{k}={_fmt(v)}" for k, v in entry.items())
                    lines.append(f"    - {cells}")
            else:
                lines.append(f"- `{key}`: {_fmt(value)}")
        lines.append("")
    if res.failures:
        lines.append("Failures:")
        lines.append("")
        for item in res.failures:
            lines.append(f"- {item}")
        lines.append("")
    if res.notes:
        lines.append("Notes:")
        lines.append("")
        for item in res.notes:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("What this does not prove. " + LIMITS.get(res.key, ""))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated check keys")
    parser.add_argument("--report", default=str(ROOT / "verification" / "report.md"))
    parser.add_argument("--json", default=str(ROOT / "verification" / "report.json"))
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    started = time.time()
    results: list[CheckResult] = []
    for key, module_name, title in CHECK_MODULES:
        if only and key not in only:
            continue
        print(f"== running {key} ...", flush=True)
        t0 = time.time()
        try:
            module = __import__(f"verification.{module_name}", fromlist=["run"])
            result = module.run()
        except Exception:                                  # pragma: no cover
            result = CheckResult(key=key, title=title, status="ERROR")
            result.fail(traceback.format_exc()[-4000:])
        result.numbers.setdefault("wall_s", round(time.time() - t0, 2))
        print(f"   {result.status:5s} in {time.time() - t0:.1f}s — {result.summary[:120]}",
              flush=True)
        results.append(result)

    elapsed = time.time() - started
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)
    n_err = sum(r.status == "ERROR" for r in results)
    n_pass = sum(r.status == "PASS" for r in results)

    git = _git()
    stamp = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    header = [
        "# PlanLab independent verification report",
        "",
        f"Generated: {stamp}  ",
        f"Repository: `{ROOT}`  ",
        f"git commit: `{git['commit'] or 'n/a'}` (dirty: {git['dirty']})  ",
        f"Python {platform.python_version()} on {platform.platform()}  ",
        f"Wall time: {elapsed:.1f} s  ",
        f"Checks: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_err} ERROR",
        "",
        "This suite is written by an independent verifier. It re-derives the theorem, "
        "re-implements a period from the written specification, audits the raw records behind "
        "the paper's numbers, re-runs sampled cells, and re-computes the statistics with its "
        "own code. It deliberately does not import the project's engine, metrics or aggregate "
        "modules to judge their output; where it does use project code it is only for *inputs* "
        "(the technology matrix) or for *records* (the stored arrays).",
        "",
    ]
    if n_skip or n_err:
        header.append(
            "> **The audit is INCOMPLETE.** Skipped or errored checks mean the "
            "corresponding evidence has NOT been verified, not that it passed."
        )
        header.append("")

    findings = [
        "## Findings reported to the Lead (the verifier's own record)",
        "",
        "An independent verifier is only worth having if its record shows what it caught. "
        "These are the defects found in the Lead's work, in the order they were reported.",
        "",
    ]
    for f in FINDINGS:
        findings.append(f"### {f['id']} — {f['severity']} ({f['date']})")
        findings.append("")
        findings.append(f"- Where: {f['where']}")
        findings.append(f"- Finding: {f['finding']}")
        findings.append(f"- Evidence: {f['evidence']}")
        findings.append(f"- Resolution: {f['resolution']}")
        findings.append("")

    body = "\n".join(_section(r) for r in results)
    report = "\n".join(header) + "\n" + "\n".join(findings) + "\n" + body
    Path = __import__("pathlib").Path
    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.json).write_text(
        json.dumps({"generated": stamp, "git": git, "elapsed_s": round(elapsed, 2),
                    "checks": [r.as_dict() for r in results]}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nwrote {rel(Path(args.report))} and {rel(Path(args.json))}")
    print(f"summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_err} ERROR")
    if n_fail or n_err:
        return 1
    if n_skip:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
