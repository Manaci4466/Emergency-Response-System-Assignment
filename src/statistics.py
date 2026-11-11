"""Phase 6b: performance statistics and the statistics.txt text.

Definitions (all times in seconds):
  handled            = incidents with an assigned responder
  responders used    = distinct responders with at least one assignment
  response time      = arrival − incident timestamp, averaged over HANDLED incidents only
  deadlines met (%)  = on-time arrivals (arrival <= deadline) / ALL incidents; unhandled = missed
These are EXPECTED times under the model (Bayes road status + HMM traffic), not observed ground truth.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.allocation import Assignment, id_key
from src.data_loader import PRIORITIES, Incident


@dataclass(frozen=True)
class PriorityStats:
    total: int
    handled: int
    on_time: int


@dataclass(frozen=True)
class Statistics:
    total: int
    handled: int
    responders_used: int
    avg_response_time: float | None  # None if nothing was handled
    on_time: int
    deadlines_met_pct: float
    per_priority: dict[str, PriorityStats]
    late: list[tuple[str, float]]  # (incident id, seconds late)
    unhandled: list[str]


def compute_statistics(
    incidents: Sequence[Incident], assignments: Sequence[Assignment], unhandled: Sequence[str]
) -> Statistics:
    by_id = {i.id: i for i in incidents}
    arrival = {a.incident_id: a.arrival for a in assignments}
    if len(arrival) != len(assignments):
        raise ValueError("an incident was assigned more than once")
    response = [arrival[i] - by_id[i].timestamp for i in arrival]
    on_time = {i for i in arrival if arrival[i] <= by_id[i].deadline}
    late = sorted(((i, arrival[i] - by_id[i].deadline) for i in arrival if i not in on_time),
                  key=lambda x: id_key(x[0]))
    per_priority = {
        p: PriorityStats(
            total=sum(1 for i in incidents if i.priority == p),
            handled=sum(1 for i in arrival if by_id[i].priority == p),
            on_time=sum(1 for i in on_time if by_id[i].priority == p),
        )
        for p in PRIORITIES
        if any(i.priority == p for i in incidents)
    }
    total = len(incidents)
    return Statistics(
        total=total,
        handled=len(arrival),
        responders_used=len({a.responder_id for a in assignments}),
        avg_response_time=sum(response) / len(response) if response else None,
        on_time=len(on_time),
        deadlines_met_pct=100.0 * len(on_time) / total if total else 100.0,
        per_priority=per_priority,
        late=late,
        unhandled=sorted(unhandled, key=id_key),
    )


def format_statistics(stats: Statistics, header: str = "") -> str:
    avg = "n/a" if stats.avg_response_time is None else f"{stats.avg_response_time:.1f} s"
    lines = [
        "System statistics:",
        "",
        f" * {stats.handled} incidents handled",
        f" * {stats.responders_used} responders used",
        f" * Average response time: {avg}",
        f" * Deadlines met: {stats.deadlines_met_pct:.1f}%",
        "",
        "Details" + (f" ({header})" if header else "") + ":",
        f" * Incidents in scenario: {stats.total} (handled {stats.handled}, "
        f"unhandled {len(stats.unhandled)})",
        f" * On time: {stats.on_time} / {stats.total}",
        " * Per priority (total / handled / on time):",
    ]
    for p, ps in stats.per_priority.items():
        pct = 100.0 * ps.on_time / ps.total
        lines.append(f"     {p:<8} {ps.total} / {ps.handled} / {ps.on_time}  ({pct:.1f}% on time)")
    late = ", ".join(f"{i} (+{d:.1f} s)" for i, d in stats.late) or "none"
    lines += [
        f" * Late incidents: {late}",
        f" * Unhandled incidents: {', '.join(stats.unhandled) or 'none'}",
        "",
        "Definitions:",
        " * Units: seconds (s). Times are absolute simulation times from t = 0.",
        " * Incidents handled: incidents with an assigned responder.",
        " * Responders used: distinct responders with at least one assignment.",
        " * Average response time: mean of (arrival - incident timestamp) over handled incidents only.",
        " * Deadlines met: on-time arrivals (arrival <= deadline) / ALL incidents;"
        " unhandled incidents count as missed.",
        " * All times are EXPECTED times under the model (Bayesian road status + HMM traffic),"
        " not observed ground truth.",
    ]
    return "\n".join(lines) + "\n"
