"""Phase 1 tests: the loader reads every scenario correctly."""
from pathlib import Path

import pytest

from src.data_loader import load_reports, load_responders, load_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "Scenarios"


# Expected counts (nodes, roads, responders, incidents) from CLAUDE.md
@pytest.mark.parametrize(
    "name, counts",
    [("small", (5, 14, 4, 7)), ("medium", (12, 30, 10, 12)), ("large", (26, 60, 10, 20))],
)
def test_counts(name: str, counts: tuple[int, int, int, int]) -> None:
    s = load_scenario(SCENARIOS / name)
    assert (len(s.nodes), len(s.roads), len(s.responders), len(s.incidents)) == counts


def test_small_details() -> None:
    s = load_scenario(SCENARIOS / "small")

    # capabilities are split on ';'
    r1 = next(r for r in s.responders if r.id == "R1")
    assert r1.capabilities == {"Fire suppression", "Rescue"}
    r4 = next(r for r in s.responders if r.id == "R4")
    assert not r4.available and r4.time_until_available == 10.0

    # incidents: absolute deadline, numbers parsed as float
    i5 = next(i for i in s.incidents if i.id == "I5")
    assert (i5.location, i5.priority, i5.deadline, i5.timestamp, i5.time_required) == (
        "N1", "High", 200.0, 50.0, 20.0,
    )

    # roads are directed: both directions exist as separate objects
    assert ("H1", "N1") in s.roads and ("N1", "H1") in s.roads
    assert s.roads[("H1", "N1")].distance == 458.0
    assert s.roads[("H1", "N1")].speed == 15.0

    # reports: 3 per reported road, grouped per DIRECTED road (blank lines skipped)
    assert [r.reported_status for r in s.reports[("N1", "N2")]] == ["Blocked", "Blocked", "Congested"]
    assert [r.reported_status for r in s.reports[("N2", "N1")]] == ["Congested"] * 3
    assert all(r.weather == "Rain" for r in s.reports[("N1", "N2")])

    # traffic observations: separate ordered sequence per directed road
    assert s.traffic_obs[("H1", "N1")][:4] == ["Low", "Low", "Low", "Medium"]
    assert s.traffic_obs[("N1", "H1")][2] == "Medium"
    assert len(s.traffic_obs[("H1", "N1")]) == 10

    # CPT orientation: OUTER key = true status, inner = reported status.
    # A diagonal entry like Police[Blocked][Blocked] reads the same whichever key is outer,
    # and the Citizen table is symmetric, so checks on those can never detect a flipped table.
    # Police is asymmetric: P(report Congested | true Open) = 0.06, but flipped it would be 0.08.
    police = s.road_model.report_given_status["Police"]
    assert police["Open"]["Congested"] == 0.06
    assert police["Congested"]["Open"] == 0.08
    # Same idea for the HMM tables (outer key = current / true state):
    assert s.traffic_model.transition["Low"]["Medium"] == 0.18  # flipped: 0.15
    assert s.traffic_model.emission["Low"]["Medium"] == 0.14  # flipped: 0.10
    assert s.traffic_model.initial["Low"] == 0.60

    # adjacency list contains exactly the roads leaving each node
    assert all(road.source == n for n, roads in s.out_edges.items() for road in roads)
    assert sum(len(v) for v in s.out_edges.values()) == len(s.roads)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    f = tmp_path / "responders.csv"
    f.write_text(
        "responder_id,type,start_location,capabilities,availability,time_until_available\n"
        "\n"
        "R1,FireTruck,N1,Fire suppression; Rescue ,Available,0\n"
        "\n"
        ",,,,,\n"
    )
    responders = load_responders(f)
    assert len(responders) == 1
    assert responders[0].capabilities == {"Fire suppression", "Rescue"}


def test_reports_must_share_weather(tmp_path: Path) -> None:
    f = tmp_path / "road_reports.csv"
    header = "from,to,weather,source,reported_status\n"
    f.write_text(header + "N1,N2,Rain,Citizen,Blocked\nN1,N2,Storm,Police,Blocked\n")
    with pytest.raises(ValueError, match="weather"):
        load_reports(f)
    # Different weather on the OPPOSITE direction is fine: roads are directed.
    f.write_text(header + "N1,N2,Rain,Citizen,Blocked\nN2,N1,Storm,Police,Blocked\n")
    assert len(load_reports(f)) == 2
