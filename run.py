"""Entry point: python run.py Scenarios/<name> [--out DIR] [--tau 0.5] [--policy csp|greedy]
                                         [--unobserved-traffic 1.0] [--node-limit 100000]

Writes DIR/plan.json (list of {responder_id, incident_id, path, time}) and DIR/statistics.txt.
Default DIR = output/<name>. Times are in seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.allocation import NODE_LIMIT, id_key
from src.data_loader import load_scenario
from src.search import heuristic_description, heuristic_is_admissible
from src.simulator import POLICIES, SimulationResult, simulate
from src.statistics import compute_statistics, format_statistics
from src.validator import check_statistics, validate


def plan_entries(result: SimulationResult) -> list[dict]:
    """EXACTLY the four keys from the assignment; time = when the responder starts the assignment."""
    ordered = sorted(result.assignments, key=lambda a: (a.start, id_key(a.responder_id)))
    return [
        {"responder_id": a.responder_id, "incident_id": a.incident_id,
         "path": list(a.path), "time": a.start}
        for a in ordered
    ]


def main(argv: list[str] | None = None) -> int:
    """Returns the exit code: 0 if the written plan is valid, 1 if the validator finds errors."""
    parser = argparse.ArgumentParser(description="Emergency response decision-support system")
    parser.add_argument("scenario", type=Path, help="scenario folder, e.g. Scenarios/small")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default output/<name>)")
    parser.add_argument("--tau", type=float, default=0.5, help="road unusable if P(Blocked) >= tau")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="csp")
    parser.add_argument("--unobserved-traffic", type=float, default=1.0,
                        help="traffic multiplier for roads without observations")
    parser.add_argument("--node-limit", type=int, default=NODE_LIMIT,
                        help="CSP search nodes per round before the best plan so far is used")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    out = args.out or Path("output") / args.scenario.name
    out.mkdir(parents=True, exist_ok=True)

    # The A* heuristic needs every road to be at least as long as the straight line: check the data.
    notes = [f"Route search: {heuristic_description(scenario)}"]
    if not heuristic_is_admissible(scenario):
        print(f"WARNING: {heuristic_description(scenario)}; using uniform-cost search")

    result = simulate(scenario, tau=args.tau, unobserved_traffic=args.unobserved_traffic,
                      policy=args.policy, node_limit=args.node_limit)
    if result.limit_rounds:
        rounds = ", ".join(f"t = {t:g}" for t in result.limit_rounds)
        warning = (f"CSP node limit ({args.node_limit}) hit in round(s) {rounds}: "
                   f"best plan found so far used, may not be optimal")
        print(f"WARNING: {warning}")
        notes.append(f"WARNING: {warning}")
    stats = compute_statistics(scenario.incidents, result.assignments, result.unhandled)
    header = f"policy {args.policy}, tau {args.tau}, unobserved traffic {args.unobserved_traffic}"

    (out / "plan.json").write_text(json.dumps(plan_entries(result), indent=2) + "\n")
    (out / "statistics.txt").write_text(format_statistics(stats, header, notes))

    avg = "n/a" if stats.avg_response_time is None else f"{stats.avg_response_time:.1f} s"
    print(f"{args.scenario.name} ({header}): {stats.handled}/{stats.total} handled, "
          f"{stats.responders_used} responders, avg response {avg}, "
          f"deadlines met {stats.deadlines_met_pct:.1f}% -> {out}")

    # Validate what was WRITTEN, read back from disk, with the independent validator.
    plan = json.loads((out / "plan.json").read_text())
    errors, warnings = validate(plan, scenario, tau=args.tau,
                                unobserved_traffic=args.unobserved_traffic)
    if not errors:  # the statistics check assumes a schema-valid plan
        errors += check_statistics(plan, scenario, (out / "statistics.txt").read_text(),
                                   tau=args.tau, unobserved_traffic=args.unobserved_traffic)
    for issue in errors:
        print(f"  ERROR {issue.code}: {issue.message}")
    print(f"plan valid: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
