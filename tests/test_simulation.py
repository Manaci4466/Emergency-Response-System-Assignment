"""Phase 6 tests: online simulation, statistics and run.py end to end."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.allocation import Assignment
from src.data_loader import Incident, load_scenario
from src.simulator import simulate
from src.statistics import compute_statistics, format_statistics

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "Scenarios"
NAMES = ["small", "medium", "large"]


def test_no_clairvoyance() -> None:
    """Changing a FUTURE incident (small I7, t = 60) must not change any earlier assignment."""
    s = load_scenario(SCENARIOS / "small")
    before = simulate(s)

    i7 = next(i for i in s.incidents if i.id == "I7")
    assert i7.timestamp == 60
    changed = dataclasses.replace(i7, location="N3", priority="Critical", deadline=70.0,
                                  required_capability="First aid")
    s.incidents = [changed if i.id == "I7" else i for i in s.incidents]
    after = simulate(s)

    timestamp = {i.id: i.timestamp for i in s.incidents}
    early = lambda res: [a for a in res.assignments if timestamp[a.incident_id] < 60]  # noqa: E731
    assert early(before) and early(before) == early(after)
    # sanity: the change really mattered for I7 itself, otherwise this test proves nothing
    i7_before = next(a for a in before.assignments if a.incident_id == "I7")
    i7_after = next(a for a in after.assignments if a.incident_id == "I7")
    assert i7_before != i7_after


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("policy", ["csp", "greedy"])
def test_every_incident_handled_or_unhandled_once(name: str, policy: str) -> None:
    s = load_scenario(SCENARIOS / name)
    res = simulate(s, policy=policy)
    assigned = [a.incident_id for a in res.assignments]
    assert len(assigned) == len(set(assigned))  # nobody assigned twice
    assert sorted(assigned + res.unhandled) == sorted(i.id for i in s.incidents)
    assert all(a.start >= next(i.timestamp for i in s.incidents if i.id == a.incident_id)
               for a in res.assignments)  # nothing starts before the incident is known


def test_statistics_hand_made() -> None:
    incidents = [
        Incident("I1", "A", "Critical", 50.0, "x", 10.0, 5.0),  # on time: arrival 30
        Incident("I2", "B", "High", 40.0, "x", 0.0, 5.0),  # late: arrival 60
        Incident("I3", "C", "Low", 100.0, "x", 20.0, 5.0),  # unhandled
    ]
    assignments = [
        Assignment("R1", "I1", ("A",), 10.0, 30.0, 35.0, False),
        Assignment("R1", "I2", ("A", "B"), 35.0, 60.0, 65.0, True),
    ]
    st = compute_statistics(incidents, assignments, ["I3"])
    assert st.handled == 2 and st.responders_used == 1
    assert st.avg_response_time == pytest.approx(((30 - 10) + (60 - 0)) / 2)  # handled only: 40
    assert st.on_time == 1
    assert st.deadlines_met_pct == pytest.approx(100 / 3)  # 1 / ALL 3, unhandled counts as missed
    assert st.late == [("I2", pytest.approx(20.0))]
    assert st.unhandled == ["I3"]
    assert st.per_priority["Low"].handled == 0
    text = format_statistics(st)
    assert text.splitlines()[:6] == [
        "System statistics:", "",
        " * 2 incidents handled",
        " * 1 responders used",
        " * Average response time: 40.0 s",
        " * Deadlines met: 33.3%",
    ]
    assert "EXPECTED times" in text


@pytest.mark.parametrize("name", NAMES)
def test_run_py_end_to_end(name: str, tmp_path: Path) -> None:
    out = tmp_path / name
    proc = subprocess.run(
        [sys.executable, "run.py", str(SCENARIOS / name), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert name in proc.stdout
    plan = json.loads((out / "plan.json").read_text())
    assert isinstance(plan, list) and plan
    for entry in plan:
        assert set(entry) == {"responder_id", "incident_id", "path", "time"}
        assert isinstance(entry["responder_id"], str) and isinstance(entry["incident_id"], str)
        assert isinstance(entry["path"], list) and all(isinstance(n, str) for n in entry["path"])
        assert isinstance(entry["time"], (int, float))
    assert [e["time"] for e in plan] == sorted(e["time"] for e in plan)
    stats = (out / "statistics.txt").read_text()
    assert stats.startswith("System statistics:")
