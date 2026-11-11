"""@brief Phase 1: load one scenario folder into typed, validated dataclasses.

No AI happens here; this is the problem formulation that the later phases use:
  - graph (nodes + DIRECTED roads)               -> search (A*), Notes 1, practicum 1
  - incidents (variables) / responders (domains) -> CSP allocation, Notes 2, practicum 1
  - road_probabilities.json (CPTs)               -> Bayes net, Notes 6
  - traffic_probabilities.json (HMM tables)      -> forward algorithm, Notes 8

Units: metres, metres/second, seconds. Deadlines are absolute times.
Every check runs once at load time, so a data error fails here with a clear message instead of
as a KeyError deep inside a later phase.
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
    """@brief A node (crossing or building) of the city graph.

    @par Fields
    - id: unique node id, e.g. "N1" or "H1".
    - name: human-readable name.
    - x, y: coordinates in metres (used by the A* heuristic).
    - type: "normal", "hospital", "police" or "fire_station" (informational only).
    """
    id: str
    name: str
    x: float  # metres
    y: float  # metres
    type: str  # "normal", "hospital", "police" or "fire_station" (informational only)


@dataclass(frozen=True)
class Road:
    """@brief A DIRECTED road from source to target.

    A->B and B->A are different roads with their own reports and observations (design decision 6).

    @par Fields
    - source, target: node ids ("from" is a Python keyword, hence source/target).
    - distance: length in metres.
    - road_type: "main" or "secondary".
    - speed: free-flow speed in metres per second.
    """
    source: str  # "from" is a Python keyword, so: source -> target
    target: str
    distance: float  # metres
    road_type: str  # "main" or "secondary"
    speed: float  # metres / second


@dataclass(frozen=True)
class Responder:
    """@brief An emergency responder (ambulance, fire truck, police unit).

    @par Fields
    - id: responder id, e.g. "R1".
    - type: vehicle type.
    - start_location: node where the responder starts.
    - capabilities: set of capabilities (the CSV column is split on ';').
    - available: True if the availability column says "Available".
    - time_until_available: seconds from t = 0 until the responder can start a job.
    """
    id: str
    type: str
    start_location: str
    capabilities: frozenset[str]
    available: bool  # availability column == "Available"
    time_until_available: float  # seconds from t = 0


@dataclass(frozen=True)
class Incident:
    """@brief An incident that needs one responder with a given capability.

    @par Fields
    - id: incident id, e.g. "I1".
    - location: node of the incident.
    - priority: Critical, High, Medium or Low.
    - deadline: ABSOLUTE time (s) by which a responder should arrive.
    - required_capability: capability the responder must have.
    - timestamp: time (s) at which the incident becomes known.
    - time_required: handling duration on site (s).
    """
    id: str
    location: str
    priority: str  # Critical / High / Medium / Low
    deadline: float  # ABSOLUTE time (s) by which a responder should arrive
    required_capability: str
    timestamp: float  # time (s) the incident becomes known
    time_required: float  # handling duration on site (s)


@dataclass(frozen=True)
class RoadReport:
    """@brief One report about the status of one directed road.

    @par Fields
    - weather: weather observed on this road (evidence for the Bayes net).
    - source: Citizen, Police or Sensor.
    - reported_status: Open, Congested or Blocked.
    """
    weather: str  # weather observed on this road (evidence for the Bayes net)
    source: str  # Citizen / Police / Sensor
    reported_status: str  # Open / Congested / Blocked


@dataclass(frozen=True)
class RoadModel:
    """@brief Bayes-net tables for one road: Weather -> Status -> each Report (Notes 6).

    @par Fields
    - weather_prior: P(W).
    - status_given_weather: P(S | W), outer key = weather.
    - report_given_status: per source, P(R | S) with the OUTER key = TRUE status.
    """
    weather_prior: dict[str, float]  # P(W)
    status_given_weather: CPT  # P(S | W), outer key = weather
    report_given_status: dict[str, CPT]  # source -> P(R | S), outer key = TRUE status


@dataclass(frozen=True)
class TrafficModel:
    """@brief HMM tables for the traffic level of a road (Notes 8).

    @par Fields
    - initial: P(X1), the state that emits observation 1 (no transition before it).
    - transition: P(X_t+1 | X_t), outer key = X_t.
    - emission: P(E_t | X_t), outer key = the true traffic level X_t.
    """
    initial: dict[str, float]  # P(X1)
    transition: CPT  # P(X_{t+1} | X_t), outer key = X_t
    emission: CPT  # P(E_t | X_t), outer key = true traffic X_t


@dataclass
class Scenario:
    """@brief A complete, validated scenario: graph, responders, incidents, evidence and models.

    @par Fields
    - nodes: node id -> Node.
    - roads: (from, to) -> Road (directed).
    - responders, incidents: lists in file order.
    - reports: (from, to) -> reports; roads without reports are absent.
    - traffic_obs: (from, to) -> observed levels ordered by observation_nr; absent = no data.
    - road_model, traffic_model: the probability tables.
    - out_edges: node id -> outgoing roads (adjacency list for search), built automatically.
    """
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
        """@brief Build the adjacency list out_edges from the roads."""
        for node_id in self.nodes:
            self.out_edges[node_id] = []
        for road in self.roads.values():
            self.out_edges[road.source].append(road)


# ---------------------------------------------------------------- helpers

def _read_csv(path: Path) -> list[dict[str, str]]:
    """@brief Read a CSV file into a list of dicts, skipping blank lines.

    The scenario files use blank lines as separators. csv.DictReader already skips completely empty
    lines; rows that contain only commas or spaces are dropped as well. Keys and values are stripped.
    @param path CSV file.
    @return One dict per data row.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # DictReader already skips completely empty lines; also drop rows that are only commas/spaces.
    return [
        {k.strip(): (v or "").strip() for k, v in row.items()}
        for row in rows
        if any((v or "").strip() for v in row.values())
    ]


def _read_json(path: Path) -> dict:
    """@brief Read a JSON file.
    @param path JSON file.
    @return The parsed JSON object.
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check_distribution(dist: dict[str, float], what: str) -> None:
    """@brief Check that a probability distribution sums to 1.
    @param dist Value -> probability.
    @param what Name used in the error message.
    @exception ValueError If the sum differs from 1 by more than 1e-6.
    """
    total = sum(dist.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{what} sums to {total}, expected 1")


def _check_cpt(cpt: CPT, what: str) -> None:
    """@brief Check that every row of a conditional probability table sums to 1.
    @param cpt Table given -> value -> probability.
    @param what Name used in the error message.
    @exception ValueError If a row does not sum to 1.
    """
    for given, row in cpt.items():
        _check_distribution(row, f"{what}[{given}]")


# ---------------------------------------------------------------- loaders

def load_graph(path: Path) -> tuple[dict[str, Node], dict[Edge, Road]]:
    """@brief Load city_graph.json.
    @param path Path to city_graph.json.
    @return (nodes by id, roads by (from, to)).
    @exception ValueError For an unknown node, a non-positive speed or a duplicate road.
    """
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
    """@brief Load responders.csv; capabilities are split on ';'.
    @param path Path to responders.csv.
    @return The responders in file order.
    """
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
    """@brief Load incidents.csv.
    @param path Path to incidents.csv.
    @return The incidents in file order.
    @exception ValueError For an unknown priority.
    """
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
    """@brief Load road_reports.csv, grouped per DIRECTED road.

    Weather is ONE evidence variable per road in the Bayes net (W -> S -> reports), so all reports for
    the same directed road must agree on it.
    @param path Path to road_reports.csv.
    @return (from, to) -> list of reports.
    @exception ValueError For an unknown source or inconsistent weather on one road.
    """
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
    """@brief Load observed_traffic.csv: one observation sequence PER DIRECTED ROAD.

    The sequence is sorted by observation_nr, which must be exactly 1..n (a gap would make the HMM
    treat two distant observations as consecutive time steps).
    @param path Path to observed_traffic.csv.
    @return (from, to) -> observed levels in time order.
    @exception ValueError If the numbers of a road are not 1..n.
    """
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
    """@brief Load road_probabilities.json into a RoadModel and check every row sums to 1.

    The report tables are mapped from the CSV source names (Citizen, Police, Sensor) to their JSON keys.
    @param path Path to road_probabilities.json.
    @return The Bayes-net tables.
    """
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
    """@brief Load traffic_probabilities.json into a TrafficModel and check every row sums to 1.
    @param path Path to traffic_probabilities.json.
    @return The HMM tables.
    """
    data = _read_json(path)
    model = TrafficModel(data["initial"], data["transition"], data["emission"])
    _check_distribution(model.initial, "initial")
    _check_cpt(model.transition, "transition")
    _check_cpt(model.emission, "emission")
    return model


def load_scenario(folder: str | Path) -> Scenario:
    """@brief Load and validate a complete scenario folder.

    Besides the per-file checks, it checks cross-file consistency (every responder, incident, report
    and observation refers to an existing node or road) and fails fast on unknown values such as a
    misspelled status "Blockd".
    @param folder Scenario folder, e.g. Scenarios/small.
    @return The validated Scenario.
    @exception ValueError For any inconsistency in the data.
    """
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
    # Fail fast on typos ("Blockd"): every value must exist in the matching probability table,
    # otherwise it would only crash later as a KeyError deep inside bayes.py / hmm.py.
    weathers = set(scenario.road_model.weather_prior)
    statuses = set(next(iter(scenario.road_model.status_given_weather.values())))
    traffic_levels = set(scenario.traffic_model.initial)
    for edge, road_reports in scenario.reports.items():
        for rep in road_reports:
            if rep.weather not in weathers:
                raise ValueError(f"report on {edge}: unknown weather {rep.weather!r}")
            if rep.reported_status not in statuses:
                raise ValueError(f"report on {edge}: unknown status {rep.reported_status!r}")
    for edge, seq in scenario.traffic_obs.items():
        for obs in seq:
            if obs not in traffic_levels:
                raise ValueError(f"traffic observation on {edge}: unknown level {obs!r}")
    return scenario
