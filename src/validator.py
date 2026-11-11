"""Phase 7: independent plan validator.

INDEPENDENT of the planner: it reads plan.json and the raw scenario files and uses only
data_loader and cost_model (the model's DEFINITION of travel time and P(Blocked)). It never imports
search, allocation or simulator; a test checks that. Travel times are recomputed by summing the edge
costs along the path given in the plan, so a wrong route or a wrong cost in the planner shows up here.

Errors (the plan breaks a hard constraint):
  schema          entry is not {responder_id: str, incident_id: str, path: non-empty [str], time: number >= 0}
  unknown_id      responder or incident id not in the scenario
  duplicate       incident assigned more than once (the extra entry is not checked further)
  capability      responder lacks the required capability
  path_start      first node != where the responder is at that moment
                  (start_location for its first job, the previous incident's location afterwards)
  path_end        last node != incident location
  no_road         consecutive nodes without a DIRECTED road between them
  blocked_road    road with P(Blocked) >= tau
  repeated_node   the path visits a node twice
  before_timestamp  job starts before the incident is known (clairvoyance)
  before_available  job starts before the responder's time_until_available
  overlap         per responder (sorted by time) a job starts before the previous
                  arrival + time_required (recomputed, tolerance 1e-6)
Warnings: late (recomputed arrival > deadline) and unhandled (incident missing from the plan).

LIMITATION: validity is relative to the BELIEVED road states (P(Blocked) < tau) and to EXPECTED
travel times under the model. A valid plan can still use a road that is in reality blocked, and real
travel times can differ; the validator cannot check the true road states, which nobody observes.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from src.cost_model import EdgeCost, edge_costs
from src.data_loader import Incident, Scenario

TOLERANCE = 1e-6
KEYS = {"responder_id", "incident_id", "path", "time"}


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


@dataclass(frozen=True)
class _Job:
    """A schema-valid, non-duplicate plan entry, with its recomputed arrival (None if unknown)."""
    index: int
    responder_id: str
    incident: Incident
    path: list[str]
    time: float
    arrival: float | None


def _schema_ok(entry: Any) -> bool:
    if not isinstance(entry, dict) or set(entry) != KEYS:
        return False
    t = entry["time"]
    return (
        isinstance(entry["responder_id"], str)
        and isinstance(entry["incident_id"], str)
        and isinstance(entry["path"], list) and len(entry["path"]) > 0
        and all(isinstance(n, str) for n in entry["path"])
        and isinstance(t, (int, float)) and not isinstance(t, bool)  # bool is a subclass of int
        and math.isfinite(t) and t >= 0
    )


def travel_time(path: list[str], costs: dict[tuple[str, str], EdgeCost]) -> float | None:
    """Expected travel time = Σ edge costs along the path (None if a road does not exist).

    Uses base · status_mult · traffic_mult also for blocked roads, so the arrival can still be
    computed; the blocked road itself is reported as a separate error.
    """
    total = 0.0
    for a, b in zip(path, path[1:]):
        c = costs.get((a, b))
        if c is None:
            return None
        total += c.base * c.status_mult * c.traffic_mult
    return total


def validate(
    plan: Any, scenario: Scenario, tau: float = 0.5, unobserved_traffic: float = 1.0
) -> tuple[list[Issue], list[Issue]]:
    errors: list[Issue] = []
    warnings: list[Issue] = []
    if not isinstance(plan, list):
        return [Issue("schema", "plan is not a list")], warnings

    costs = edge_costs(scenario, tau=tau, unobserved_traffic=unobserved_traffic)
    responders = {r.id: r for r in scenario.responders}
    incidents = {i.id: i for i in scenario.incidents}
    seen: set[str] = set()
    jobs: list[_Job] = []

    # Pass 1: per entry (schema, ids, duplicates, capability, path shape, start time)
    for k, entry in enumerate(plan):
        where = f"entry {k}"
        if not _schema_ok(entry):
            errors.append(Issue("schema", f"{where}: wrong keys or types: {entry!r}"))
            continue
        rid, iid, path, time = entry["responder_id"], entry["incident_id"], entry["path"], entry["time"]
        where = f"entry {k} ({rid} -> {iid})"
        if rid not in responders or iid not in incidents:
            errors.append(Issue("unknown_id", f"{where}: unknown responder or incident id"))
            continue
        if iid in seen:
            errors.append(Issue("duplicate", f"{where}: {iid} is assigned more than once"))
            continue
        seen.add(iid)
        resp, inc = responders[rid], incidents[iid]

        if inc.required_capability not in resp.capabilities:
            errors.append(Issue("capability", f"{where}: {rid} lacks {inc.required_capability!r}"))
        if path[-1] != inc.location:
            errors.append(Issue("path_end", f"{where}: path ends at {path[-1]}, incident is at {inc.location}"))
        if len(set(path)) != len(path):
            errors.append(Issue("repeated_node", f"{where}: path visits a node twice: {path}"))
        for a, b in zip(path, path[1:]):
            if (a, b) not in costs:
                errors.append(Issue("no_road", f"{where}: no road {a}->{b}"))
            elif not costs[(a, b)].usable:
                errors.append(Issue("blocked_road", f"{where}: road {a}->{b} has "
                                    f"P(Blocked) = {costs[(a, b)].p_blocked:.3f} >= tau {tau}"))
        if time < inc.timestamp:
            errors.append(Issue("before_timestamp", f"{where}: starts at {time} before the "
                                f"incident is known at {inc.timestamp}"))
        if time < resp.time_until_available:
            errors.append(Issue("before_available", f"{where}: starts at {time}, {rid} is "
                                f"busy until {resp.time_until_available}"))
        jobs.append(_Job(k, rid, inc, path, time, _arrival(time, travel_time(path, costs))))

    # Pass 2: per responder, in time order: path start location and overlap
    for rid, resp in responders.items():
        # free_at is about the PREVIOUS job only; the initial busy time is checked in pass 1
        location, free_at = resp.start_location, None
        for job in sorted((j for j in jobs if j.responder_id == rid), key=lambda j: (j.time, j.index)):
            where = f"entry {job.index} ({rid} -> {job.incident.id})"
            if job.path[0] != location:
                errors.append(Issue("path_start", f"{where}: path starts at {job.path[0]}, "
                                    f"{rid} is at {location}"))
            if free_at is not None and job.time < free_at - TOLERANCE:
                errors.append(Issue("overlap", f"{where}: starts at {job.time:.2f}, previous "
                                    f"job ends at {free_at:.2f}"))
            # After the job the responder is at the incident location (decision 8).
            location = job.incident.location
            free_at = None if job.arrival is None else job.arrival + job.incident.time_required

    # Warnings: late arrivals and unhandled incidents
    for job in jobs:
        if job.arrival is not None and job.arrival > job.incident.deadline:
            warnings.append(Issue("late", f"{job.incident.id}: arrives {job.arrival:.2f}, "
                                  f"deadline {job.incident.deadline} (+{job.arrival - job.incident.deadline:.2f} s)"))
    for iid in incidents:
        if iid not in seen:
            warnings.append(Issue("unhandled", f"{iid} is not in the plan"))
    return errors, warnings


def _arrival(time: float, travel: float | None) -> float | None:
    return None if travel is None else time + travel


def plan_statistics(
    plan: list[dict], scenario: Scenario, tau: float = 0.5, unobserved_traffic: float = 1.0
) -> dict[str, float | None]:
    """The four required statistics recomputed from plan.json alone (assumes a schema-valid plan)."""
    costs = edge_costs(scenario, tau=tau, unobserved_traffic=unobserved_traffic)
    incidents = {i.id: i for i in scenario.incidents}
    arrivals: dict[str, float] = {}
    for e in plan:
        travel = travel_time(e["path"], costs)
        if travel is not None and e["incident_id"] in incidents:
            arrivals.setdefault(e["incident_id"], e["time"] + travel)
    response = [arrivals[i] - incidents[i].timestamp for i in arrivals]
    on_time = sum(1 for i in arrivals if arrivals[i] <= incidents[i].deadline)
    return {
        "handled": len(arrivals),
        "responders_used": len({e["responder_id"] for e in plan}),
        "avg_response_time": sum(response) / len(response) if response else None,
        "deadlines_met_pct": 100.0 * on_time / len(incidents) if incidents else 100.0,
    }


_STAT_PATTERNS = {
    "handled": r"\*\s*(\d+) incidents handled",
    "responders_used": r"\*\s*(\d+) responders used",
    "avg_response_time": r"Average response time\s*:\s*([\d.]+|n/a)",
    "deadlines_met_pct": r"Deadlines met\s*:\s*([\d.]+)%",
}


def check_statistics(
    plan: list[dict], scenario: Scenario, statistics_text: str,
    tau: float = 0.5, unobserved_traffic: float = 1.0,
) -> list[Issue]:
    """Compare the four lines of statistics.txt with values recomputed from plan.json."""
    expected = plan_statistics(plan, scenario, tau, unobserved_traffic)
    errors: list[Issue] = []
    for key, pattern in _STAT_PATTERNS.items():
        m = re.search(pattern, statistics_text)
        if m is None:
            errors.append(Issue("statistics", f"{key}: line not found in statistics.txt"))
            continue
        value, want = m.group(1), expected[key]
        if value == "n/a" or want is None:
            ok = value == "n/a" and want is None
        elif key in ("handled", "responders_used"):
            ok = int(value) == want
        else:  # printed with one decimal
            ok = abs(float(value) - want) <= 0.05 + 1e-9
        if not ok:
            errors.append(Issue("statistics", f"{key}: statistics.txt says {value}, plan gives {want}"))
    return errors
