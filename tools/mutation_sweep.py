"""Mutation sweep: does the test suite catch typical bugs in EVERY module?

Usage:  python tools/mutation_sweep.py            (about a minute)

For each mutant (one small, realistic bug) the project is copied to a temporary folder, the bug is
applied to the copy, and the whole test suite runs there. A mutant is CAUGHT ("killed") if at least
one test fails. A mutant that survives means a bug of that kind would go unnoticed: a gap in the tests.
The real source files are never modified, and no bytecode cache is written (no stale .pyc problems).
Each mutation's text must occur exactly once in its file, otherwise the sweep reports an error
instead of silently testing nothing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY = ["src", "tests", "run.py", "pyproject.toml", "Scenarios"]


@dataclass(frozen=True)
class Mutant:
    file: str
    old: str
    new: str
    bug: str


MUTANTS: list[Mutant] = [
    # ---------------------------------------------------------------- phase 1: data loader
    Mutant("src/data_loader.py", 'if any((v or "").strip() for v in row.values())', "if True",
           "loader keeps comma-only rows"),
    Mutant("src/data_loader.py", 'row["capabilities"].split(";")', 'row["capabilities"].split(",")',
           "capabilities split on the wrong separator"),
    Mutant("src/data_loader.py", "if numbers != list(range(1, len(seq) + 1)):", "if False:",
           "gaps in observation numbers accepted"),
    Mutant("src/data_loader.py", "        if len(weathers) > 1:", "        if False:",
           "inconsistent weather per road accepted"),
    Mutant("src/data_loader.py", "            if rep.reported_status not in statuses:", "            if False:",
           "unknown reported status accepted"),
    # ---------------------------------------------------------------- phase 2: Bayes net
    Mutant("src/bayes.py", "cpt[s][report.reported_status]", "cpt[report.reported_status][s]",
           "report CPT read the wrong way round"),
    Mutant("src/bayes.py", "    return normalise(unnormalised)", "    return unnormalised",
           "posterior not normalised"),
    Mutant("src/bayes.py", 'weather = reports[0].weather if reports else None',
           'weather = reports[0].weather if reports else "Rain"',
           "unreported road: weather assumed instead of summed out"),
    # ---------------------------------------------------------------- phase 3: HMM
    Mutant("src/hmm.py", "belief[x] * model.transition[x][y]", "belief[x] * model.transition[y][x]",
           "transition table read the wrong way round"),
    Mutant("src/hmm.py", "return {x: p / z for x, p in weighted.items()}, z",
           "return dict(weighted), z", "filtered belief not normalised"),
    Mutant("src/hmm.py", "        if t > 0:", "        if True:",
           "transition applied before observation 1 (off-by-one)"),
    Mutant("src/hmm.py", "predict(scenario.traffic_model, filtered, steps=1)",
           "predict(scenario.traffic_model, filtered, steps=0)", "no prediction step to t = 0"),
    # ---------------------------------------------------------------- phase 4: cost model + A*
    Mutant("src/cost_model.py", 'STATUS_MULT: dict[str, float] = {"Open": 1.0, "Congested": 1.5}',
           'STATUS_MULT: dict[str, float] = {"Open": 1.0, "Congested": 1.0}',
           "Congested multiplier 1.0 instead of 1.5"),
    Mutant("src/cost_model.py", "cost = None if p_blocked >= tau else", "cost = None if p_blocked > tau else",
           "road with P(Blocked) exactly tau still usable"),
    Mutant("src/cost_model.py", "t_mult = expected_multiplier(traffic[edge]) if edge in traffic else unobserved_traffic",
           "t_mult = unobserved_traffic", "traffic belief ignored"),
    Mutant("src/search.py", "            g2 = g + edge_cost\n", "            g2 = g + 1.0\n",
           "search ignores road costs"),
    Mutant("src/search.py", "lambda n: _straight_line(scenario, n, goal) / speed",
           "lambda n: _straight_line(scenario, n, goal) / (speed / 3)", "inadmissible heuristic (3x too large)"),
    Mutant("src/search.py", "    if not heuristic_is_admissible(scenario):  # data breaks the premise: fall back to UCS",
           "    if False:", "no UCS fallback for roads shorter than the straight line"),
    # ---------------------------------------------------------------- phase 5: CSP allocation
    Mutant("src/allocation.py", """        inc = min(unassigned, key=lambda i: (len(ontime[i.id]), -WEIGHTS[i.priority],
                                             i.deadline, id_key(i.id)))""",
           "        inc = min(unassigned, key=lambda i: id_key(i.id))", "no MRV (id order)"),
    Mutant("src/allocation.py", "        start = max(self.now, st.free_at)", "        start = self.now",
           "busy responders not queued"),
    Mutant("src/allocation.py", "arrival + inc.time_required, arrival > inc.deadline",
           "arrival, arrival > inc.deadline", "handling time forgotten when queueing"),
    Mutant("src/allocation.py", "bound = late_w + sum(WEIGHTS[i.priority] for i in unassigned if not ontime[i.id])",
           "bound = late_w + sum(WEIGHTS[i.priority] for i in unassigned)", "over-strong lower bound"),
    Mutant("src/allocation.py", "        if prune and (bound, resp) >= best_obj:", "        if False:",
           "no branch-and-bound pruning"),
    Mutant("src/allocation.py", 'WEIGHTS: dict[str, int] = {"Critical": 8, "High": 4, "Medium": 2, "Low": 1}',
           'WEIGHTS: dict[str, int] = {"Critical": 1, "High": 1, "Medium": 1, "Low": 1}',
           "priority weights ignored"),
    Mutant("src/allocation.py", "    if best_obj == (math.inf, math.inf):", "    if False:",
           "no greedy fallback when the node limit is hit"),
    # ---------------------------------------------------------------- phase 6: simulator + statistics
    Mutant("src/simulator.py", "for t in sorted(groups)]", "for t in sorted(groups, reverse=True)]",
           "rounds processed latest first (clairvoyance)"),
    Mutant("src/statistics.py", "100.0 * len(on_time) / total if total else 100.0",
           "100.0 * len(on_time) / max(len(arrival), 1)", "deadlines met over handled instead of ALL"),
    Mutant("src/statistics.py", "sum(response) / len(response) if response else None",
           "sum(response) / len(incidents) if response else None", "average response over ALL incidents"),
    Mutant("run.py", '"path": list(a.path), "time": a.start}', '"path": list(a.path), "time": a.arrival}',
           "plan time = arrival instead of start"),
    # ---------------------------------------------------------------- phase 7: validator
    Mutant("src/validator.py", "            if free_at is not None and job.time < free_at - TOLERANCE:",
           "            if False:", "validator: no overlap check"),
    Mutant("src/validator.py", "            location = job.incident.location\n", "            pass\n",
           "validator forgets the responder moved"),
    Mutant("src/validator.py", "total += c.base * c.status_mult * c.traffic_mult", "total += c.base",
           "validator ignores the multipliers"),
    Mutant("src/validator.py", "            elif not costs[(a, b)].usable:", "            elif False:",
           "validator accepts blocked roads"),
    # ---------------------------------------------------------------- phase 8: experiments
    Mutant("src/experiments.py", '                if stranded or any(world_status[e] == "Blocked" for e in route):',
           '                if any(world_status[e] == "Blocked" for e in route):',
           "Monte Carlo: no stranding after a blocked route"),
    Mutant("src/experiments.py", "for mult in [1.0, stationary_multiplier(s)]:",
           "for mult in [1.0, stationary_multiplier(next(iter(scen.values())))]:",
           "experiment 3: one scenario's stationary multiplier for all"),
]


def _copy_project(dest: Path) -> None:
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dest / name)


def _run_tests(cwd: Path) -> tuple[bool, str]:
    """Returns (all tests passed, last line of the output)."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("RUN_MUTATION_SWEEP", None)  # never let the copied suite start another sweep
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                              cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timeout (counts as caught)"
    lines = proc.stdout.strip().splitlines()
    return proc.returncode == 0, lines[-1] if lines else proc.stderr.strip()[-200:]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mutation-sweep-") as tmp:
        work = Path(tmp)
        _copy_project(work)
        ok, summary = _run_tests(work)
        if not ok:
            print(f"The UNMUTATED suite fails, sweep aborted: {summary}")
            return 1
        print(f"Unmutated suite: {summary}\n")
        caught, problems = 0, []
        width = max(len(m.bug) for m in MUTANTS)
        for k, m in enumerate(MUTANTS, 1):
            target = work / m.file
            original = target.read_text()
            if original.count(m.old) != 1:
                problems.append(m.bug)
                print(f"{k:>2}. {m.bug:<{width}}  ERROR: mutation text found "
                      f"{original.count(m.old)} times in {m.file}")
                continue
            target.write_text(original.replace(m.old, m.new))
            passed, last = _run_tests(work)
            target.write_text(original)  # restore before the next mutant
            if passed:
                problems.append(m.bug)
            else:
                caught += 1
            print(f"{k:>2}. {m.bug:<{width}}  {'SURVIVED' if passed else 'caught'}   ({last})")
        print(f"\n{caught} of {len(MUTANTS)} mutants caught.")
        if problems:
            print("Not caught / not applied: " + "; ".join(problems))
        return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
