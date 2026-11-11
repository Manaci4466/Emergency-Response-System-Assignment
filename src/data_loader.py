"""Phase 1: load one scenario folder into typed dataclasses.

No AI happens here; this is the problem formulation that later phases use:
  - graph (nodes + DIRECTED roads)            -> search (A*), practicum 1
  - incidents (variables) / responders (domains) -> CSP allocation, practicum 1
  - road_probabilities.json (CPTs)            -> Bayes net, Notes 6
  - traffic_probabilities.json (HMM tables)   -> forward algorithm, Notes 8

Units: metres, metres/second, seconds. Deadlines are absolute times.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

# A conditional probability table: CPT[given][value] = P(value | given).
# For the report models the OUTER key is the TRUE status, the inner key the REPORTED status.
CPT = dict[str, dict[str, float]]
Edge = tuple[str, str]  # (from, to): roads are directed, A->B and B->A are different roads

# Report source in road_reports.csv -> key of its CPT in road_probabilities.json
REPORT_SOURCES: dict[str, str] = {
    "Citizen": "citizen_report",
    "Police": "police_report",
    "Sensor": "sensor_report",
}
PRIORITIES: tuple[str, ...] = ("Critical", "High", "Medium", "Low")


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    x: float  # metres
    y: float  # metres
    type: str  # "normal", "hospital", "police" or "fire_station" (informational only)


@dataclass(frozen=True)
class Road:
    source: str  # "from" is a Python keyword, so: source -> target
    target: str
    distance: float  # metres
    road_type: str  # "main" or "secondary"
    speed: float  # metres / second


@dataclass(frozen=True)
class Responder:
    id: str
    type: str
    start_location: str
    capabilities: frozenset[str]
    available: bool  # availability column == "Available"
    time_until_available: float  # seconds from t = 0


@dataclass(frozen=True)
class Incident:
    id: str
    location: str
    priority: str  # Critical / High / Medium / Low
    deadline: float  # ABSOLUTE time (s) by which a responder should arrive
    required_capability: str
    timestamp: float  # time (s) the incident becomes known
    time_required: float  # handling duration on site (s)


@dataclass(frozen=True)
class RoadReport:
    weather: str  # weather observed on this road (evidence for the Bayes net)
    source: str  # Citizen / Police / Sensor
    reported_status: str  # Open / Congested / Blocked


@dataclass(frozen=True)
class RoadModel:
    """Bayes net tables: Weather -> Status -> each Report."""
    weather_prior: dict[str, float]  # P(W)
    status_given_weather: CPT  # P(S | W), outer key = weather
    report_given_status: dict[str, CPT]  # source -> P(R | S), outer key = TRUE status


@dataclass(frozen=True)
class TrafficModel:
    """HMM tables. initial = P(X1), the state that emits observation 1."""
    initial: dict[str, float]  # P(X1)
    transition: CPT  # P(X_{t+1} | X_t), outer key = X_t
    emission: CPT  # P(E_t | X_t), outer key = true traffic X_t


@dataclass
class Scenario:
    nodes: dict[str, Node]
    roads: dict[Edge, Road]
    responders: list[Responder]
    incidents: list[Incident]
    reports: dict[Edge, list[RoadReport]]  # roads WITHOUT reports are simply absent
    traffic_obs: dict[Edge, list[str]]  # ordered by observation_nr; absent = no data
    road_model: RoadModel
    traffic_model: TrafficModel
    out_edges: dict[str, list[Road]] = field(default_factory=dict)  # adjacency list for search

    def __post_init__(self) -> None:
        for node_id in self.nodes:
            self.out_edges[node_id] = []
        for road in self.roads.values():
            self.out_edges[road.source].append(road)


# ---------------------------------------------------------------- helpers

def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dicts, skipping blank lines (the scenario files use them as separators)."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # DictReader already skips completely empty lines; also drop rows that are only commas/spaces.
    return [
        {k.strip(): (v or "").strip() for k, v in row.items()}
        for row in rows
        if any((v or "").strip() for v in row.values())
    ]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check_distribution(dist: dict[str, float], what: str) -> None:
    """Every row of a CPT must sum to 1 (small tolerance for rounding in the files)."""
    total = sum(dist.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{what} sums to {total}, expected 1")


def _check_cpt(cpt: CPT, what: str) -> None:
    for given, row in cpt.items():
        _check_distribution(row, f"{what}[{given}]")


# ---------------------------------------------------------------- loaders

def load_graph(path: Path) -> tuple[dict[str, Node], dict[Edge, Road]]:
    data = _read_json(path)
    nodes = {
        n["id"]: Node(n["id"], n["name"], float(n["x"]), float(n["y"]), n["type"])
        for n in data["nodes"]
    }
    roads: dict[Edge, Road] = {}
    for r in data["roads"]:
        road = Road(r["from"], r["to"], float(r["distance"]), r["road_type"], float(r["speed"]))
        if road.source not in nodes or road.target not in nodes:
            raise ValueError(f"road {road.source}->{road.target} uses an unknown node")
        if road.speed <= 0:
            raise ValueError(f"road {road.source}->{road.target} has speed {road.speed}")
        edge = (road.source, road.target)
        if edge in roads:
            raise ValueError(f"duplicate road {edge}")
        roads[edge] = road
    return nodes, roads


def load_responders(path: Path) -> list[Responder]:
    return [
        Responder(
            id=row["responder_id"],
            type=row["type"],
            start_location=row["start_location"],
            capabilities=frozenset(c.strip() for c in row["capabilities"].split(";") if c.strip()),
            available=row["availability"] == "Available",
            time_until_available=float(row["time_until_available"]),
        )
        for row in _read_csv(path)
    ]


def load_incidents(path: Path) -> list[Incident]:
    incidents = [
        Incident(
            id=row["incident_id"],
            location=row["location"],
            priority=row["priority"],
            deadline=float(row["deadline"]),
            required_capability=row["required_capability"],
            timestamp=float(row["timestamp"]),
            time_required=float(row["time_required"]),
        )
        for row in _read_csv(path)
    ]
    for inc in incidents:
        if inc.priority not in PRIORITIES:
            raise ValueError(f"{inc.id}: unknown priority {inc.priority!r}")
    return incidents


def load_reports(path: Path) -> dict[Edge, list[RoadReport]]:
    reports: dict[Edge, list[RoadReport]] = {}
    for row in _read_csv(path):
        if row["source"] not in REPORT_SOURCES:
            raise ValueError(f"unknown report source {row['source']!r}")
        edge = (row["from"], row["to"])
        reports.setdefault(edge, []).append(
            RoadReport(row["weather"], row["source"], row["reported_status"])
        )
    # Weather is ONE evidence variable per road in the Bayes net (W -> S -> reports),
    # so all reports for the same directed road must agree on it.
    for edge, road_reports in reports.items():
        weathers = {r.weather for r in road_reports}
        if len(weathers) > 1:
            raise ValueError(f"reports for road {edge} disagree on weather: {sorted(weathers)}")
    return reports


def load_traffic_observations(path: Path) -> dict[Edge, list[str]]:
    """One observation sequence PER DIRECTED ROAD, sorted by observation_nr."""
    numbered: dict[Edge, list[tuple[int, str]]] = {}
    for row in _read_csv(path):
        edge = (row["from"], row["to"])
        numbered.setdefault(edge, []).append((int(row["observation_nr"]), row["traffic_observation"]))
    result: dict[Edge, list[str]] = {}
    for edge, seq in numbered.items():
        seq.sort()
        numbers = [nr for nr, _ in seq]
        if numbers != list(range(1, len(seq) + 1)):
            raise ValueError(f"traffic observations for {edge} are not numbered 1..n: {numbers}")
        result[edge] = [obs for _, obs in seq]
    return result


def load_road_model(path: Path) -> RoadModel:
    data = _read_json(path)
    model = RoadModel(
        weather_prior=data["weather_prior"],
        status_given_weather=data["road_status_given_weather"],
        report_given_status={src: data[key] for src, key in REPORT_SOURCES.items()},
    )
    _check_distribution(model.weather_prior, "weather_prior")
    _check_cpt(model.status_given_weather, "road_status_given_weather")
    for src, cpt in model.report_given_status.items():
        _check_cpt(cpt, f"{src} report")
    return model


def load_traffic_model(path: Path) -> TrafficModel:
    data = _read_json(path)
    model = TrafficModel(data["initial"], data["transition"], data["emission"])
    _check_distribution(model.initial, "initial")
    _check_cpt(model.transition, "transition")
    _check_cpt(model.emission, "emission")
    return model


def load_scenario(folder: str | Path) -> Scenario:
    folder = Path(folder)
    nodes, roads = load_graph(folder / "city_graph.json")
    scenario = Scenario(
        nodes=nodes,
        roads=roads,
        responders=load_responders(folder / "responders.csv"),
        incidents=load_incidents(folder / "incidents.csv"),
        reports=load_reports(folder / "road_reports.csv"),
        traffic_obs=load_traffic_observations(folder / "observed_traffic.csv"),
        road_model=load_road_model(folder / "road_probabilities.json"),
        traffic_model=load_traffic_model(folder / "traffic_probabilities.json"),
    )
    # Cross-file consistency: everything must refer to existing nodes / roads.
    for r in scenario.responders:
        if r.start_location not in nodes:
            raise ValueError(f"responder {r.id} starts at unknown node {r.start_location}")
    for i in scenario.incidents:
        if i.location not in nodes:
            raise ValueError(f"incident {i.id} at unknown node {i.location}")
    for edge in list(scenario.reports) + list(scenario.traffic_obs):
        if edge not in roads:
            raise ValueError(f"data refers to non-existent road {edge}")
    return scenario
