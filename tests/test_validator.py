"""Phase 7 tests: the independent validator accepts the real plans and catches injected faults."""
import ast
import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from run import plan_entries
from src.cost_model import edge_costs
from src.data_loader import load_scenario
from src.simulator import simulate
from src.statistics import compute_statistics, format_statistics
from src.validator import check_statistics, validate

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "Scenarios"
SMALL = load_scenario(SCENARIOS / "small")


def real_plan(scenario, policy: str = "csp") -> list[dict]:
    """The plan exactly as run.py writes it (JSON round trip)."""
    return json.loads(json.dumps(plan_entries(simulate(scenario, policy=policy))))


def test_validator_is_independent_of_the_planner() -> None:
    tree = ast.parse((ROOT / "src" / "validator.py").read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"src.search", "src.allocation", "src.simulator"}, imported


@pytest.mark.parametrize("name", ["small", "medium", "large"])
@pytest.mark.parametrize("policy", ["csp", "greedy"])
def test_real_plans_are_valid(name: str, policy: str) -> None:
    s = load_scenario(SCENARIOS / name)
    result = simulate(s, policy=policy)
    plan = json.loads(json.dumps(plan_entries(result)))
    errors, warnings = validate(plan, s)
    assert errors == []
    # the warnings agree with the planner: late = its late assignments, unhandled = its unhandled
    late = {w.message.split(":")[0] for w in warnings if w.code == "late"}
    assert late == {a.incident_id for a in result.assignments if a.late}
    assert {w.message.split()[0] for w in warnings if w.code == "unhandled"} == set(result.unhandled)
    # the four statistics in statistics.txt match a recomputation from plan.json alone
    text = format_statistics(compute_statistics(s.incidents, result.assignments, result.unhandled))
    assert check_statistics(plan, s, text) == []


def test_statistics_mismatch_is_caught() -> None:
    plan = real_plan(SMALL)
    result = simulate(SMALL)
    text = format_statistics(compute_statistics(SMALL.incidents, result.assignments, result.unhandled))
    wrong = text.replace("Deadlines met: 28.6%", "Deadlines met: 100.0%")
    assert [e.code for e in check_statistics(plan, SMALL, wrong)] == ["statistics"]


# ------------------------------------------------------------------ fault injection
# Each fault changes the real small CSP plan (or builds a one-entry plan) so that EXACTLY ONE
# hard constraint is broken. The chosen entries are the last job of their responder where needed,
# so that the fault cannot cascade into the next job (e.g. a wrong path start).
#
# small CSP plan:  R1 I2 [N1,H1,N2] 0      R2 I4 [H1,N2] 0        R3 I3 [N2,H1,N4,N3] 0
#                  R4 I1 [N3,N4] 10        R4 I6 [N4,H1,N1] 91.1  R1 I5 [N2,N1] 122.2
#                  R3 I7 [N3,N4,H1] 151.2

COSTS = edge_costs(SMALL)


def entry(plan: list[dict], incident_id: str) -> dict:
    return next(e for e in plan if e["incident_id"] == incident_id)


def drop_middle_node(plan):
    # I3: N2,H1,N4,N3 -> N2,N4,N3 is blocked (0.527), so look for a drop that gives NO road at all
    for e in plan:
        p = e["path"]
        for k in range(1, len(p) - 1):
            if (p[k - 1], p[k + 1]) not in COSTS:
                e["path"] = p[:k] + p[k + 1:]
                return plan
    raise AssertionError("no path where dropping a middle node removes the road")


def through_blocked_road(plan):
    entry(plan, "I2")["path"] = ["N1", "N2"]  # N1->N2: P(Blocked) = 0.527 >= 0.5
    return plan


def wrong_first_node(plan):
    e = entry(plan, "I6")  # R4's last job, really starts at N4 (where I1 was)
    e["path"] = ["N2", "N1"]  # a usable road to the right destination, but R4 is not at N2
    return plan


def wrong_last_node(plan):
    entry(plan, "I7")["path"] = ["N3", "N4"]  # stops one node early (I7 is at H1)
    return plan


def incapable_responder(_plan):
    return [{"responder_id": "R1", "incident_id": "I4", "path": ["N1", "H1", "N2"], "time": 0}]
    # R1 (fire suppression, rescue) cannot do First aid; the rest of the entry is fine


def before_timestamp(_plan):
    return [{"responder_id": "R1", "incident_id": "I5", "path": ["N1"], "time": 0}]
    # R1 is at N1, can do Rescue, but I5 only becomes known at t = 50


def r4_before_available(plan):
    entry(plan, "I1")["time"] = 5.0  # R4 is busy until 10
    return plan


def same_incident_twice(plan):
    return plan + [copy.deepcopy(entry(plan, "I7"))]


def overlapping_jobs(plan):
    entry(plan, "I6")["time"] = 85.0  # R4 arrives at I1 at 81.1 and needs 10 s there
    return plan


def unknown_id(plan):
    return plan + [{"responder_id": "R1", "incident_id": "I99", "path": ["N1"], "time": 500}]


def missing_key(plan):
    del entry(plan, "I7")["time"]
    return plan


def wrong_type(plan):
    entry(plan, "I7")["time"] = "151"
    return plan


def repeated_node(plan):
    entry(plan, "I7")["path"] = ["N3", "N4", "N3", "N4", "H1"]
    return plan


FAULTS: list[tuple[str, Callable, str]] = [
    ("drop a middle path node", drop_middle_node, "no_road"),
    ("route through small N1->N2 (blocked at tau 0.5)", through_blocked_road, "blocked_road"),
    ("wrong first node", wrong_first_node, "path_start"),
    ("wrong last node", wrong_last_node, "path_end"),
    ("incapable responder", incapable_responder, "capability"),
    ("time before the incident timestamp", before_timestamp, "before_timestamp"),
    ("R4 starts before 10", r4_before_available, "before_available"),
    ("same incident twice", same_incident_twice, "duplicate"),
    ("two overlapping jobs of one responder", overlapping_jobs, "overlap"),
    ("unknown id", unknown_id, "unknown_id"),
    ("missing key", missing_key, "schema"),
    ("wrong type (time as string)", wrong_type, "schema"),
    ("repeated node in path", repeated_node, "repeated_node"),
]


@pytest.mark.parametrize("name, fault, expected", FAULTS, ids=[f[0] for f in FAULTS])
def test_fault_injection(name: str, fault: Callable, expected: str) -> None:
    plan = fault(real_plan(SMALL))
    errors, _ = validate(plan, SMALL)
    assert [e.code for e in errors] == [expected], [str(e) for e in errors]


def test_fault_table() -> None:
    """Prints the fault -> caught table for the report."""
    rows = []
    for name, fault, expected in FAULTS:
        errors, _ = validate(fault(real_plan(SMALL)), SMALL)
        codes = [e.code for e in errors]
        rows.append((name, expected, "yes" if codes == [expected] else f"NO {codes}"))
    width = max(len(r[0]) for r in rows)
    print("\n" + f"{'Injected fault':<{width}}  {'Expected error':<17} Caught")
    for name, expected, caught in rows:
        print(f"{name:<{width}}  {expected:<17} {caught}")
    assert all(r[2] == "yes" for r in rows)
