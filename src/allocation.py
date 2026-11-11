"""Phase 5: responder allocation as a CSP, solved per round (practicum 1: backtracking, MRV,
forward checking) with branch and bound on a lexicographic objective.

One round = the incidents that become known at time `now` (no clairvoyance).
  Variables   : this round's incidents.
  Domains     : responders with the required capability that can reach the incident
                (unary constraints, filtered first; empty domain -> unhandled).
  Hard        : capability, reachability, a responder does one job at a time (queueing:
                the next job starts when the previous one finishes, at its location).
  Soft        : deadlines. Late help is better than no help, so lateness is a PENALTY, never a
                reason to fail a branch.
  Objective   : lexicographic (1) Σ priority weight of LATE incidents, (2) Σ (arrival − timestamp).

Limitation: the jobs of one responder are done in the order the search assigns them, and the
search order is fixed by MRV. The CSP is therefore not exhaustive over every job order.

Because the assignment order also fixes the job order of a shared responder, MRV here affects the
QUALITY of the solution, not only the speed (in a classical CSP variable order only changes speed).
Without MRV (plain id order) medium t=0 gets worse: R10 does I2 before I3, and the Critical I3
arrives late at 41.6 s (late weight 8 instead of 4).
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.cost_model import EdgeCost
from src.data_loader import Edge, Incident, Scenario
from src.search import astar

WEIGHTS: dict[str, int] = {"Critical": 8, "High": 4, "Medium": 2, "Low": 1}
NODE_LIMIT = 100_000

Objective = tuple[float, float]  # (late weight, total response time)


@dataclass(frozen=True)
class ResponderState:
    location: str
    free_at: float  # absolute time (s) at which the responder can start a new job


@dataclass(frozen=True)
class Assignment:
    responder_id: str
    incident_id: str
    path: tuple[str, ...]
    start: float  # = the plan's "time": when the responder leaves
    arrival: float
    finish: float  # arrival + time_required; the responder is then free at the incident
    late: bool


@dataclass(frozen=True)
class AllocationStats:
    nodes: int  # assignments tried (search nodes)
    hit_limit: bool  # True -> node limit reached, best solution found so far returned
    late_weight: float
    response_time: float


AllocationResult = tuple[list[Assignment], list[str], dict[str, ResponderState], AllocationStats]


def id_key(identifier: str) -> tuple[str, int]:
    """Natural order for ids: I2 < I10 (plain string order would put I10 first)."""
    m = re.fullmatch(r"(\D*)(\d+)", identifier)
    return (m.group(1), int(m.group(2))) if m else (identifier, 0)


def initial_states(scenario: Scenario) -> dict[str, ResponderState]:
    return {r.id: ResponderState(r.start_location, r.time_until_available) for r in scenario.responders}


class _Planner:
    """Shared helpers for the CSP and the greedy baseline (routes are cached per round)."""

    def __init__(self, scenario: Scenario, costs: dict[Edge, EdgeCost], now: float) -> None:
        self.scenario = scenario
        self.costs = costs
        self.now = now
        self._routes: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    def route(self, a: str, b: str) -> tuple[list[str], float] | None:
        if (a, b) not in self._routes:
            res = astar(self.scenario, self.costs, a, b)
            self._routes[(a, b)] = None if res is None else (res[0], res[1])
        return self._routes[(a, b)]

    def option(self, inc: Incident, rid: str, st: ResponderState) -> Assignment | None:
        """Assign responder rid (in state st) to inc. None if it cannot reach the incident."""
        r = self.route(st.location, inc.location)
        if r is None:
            return None
        path, travel = r
        start = max(self.now, st.free_at)  # wait until the responder is free (queueing)
        arrival = start + travel
        return Assignment(rid, inc.id, tuple(path), start, arrival,
                          arrival + inc.time_required, arrival > inc.deadline)

    def domains(
        self, incidents: Sequence[Incident], states: dict[str, ResponderState]
    ) -> dict[str, list[str]]:
        """Unary constraints: capability + reachable from the responder's current location."""
        return {
            inc.id: [
                r.id for r in self.scenario.responders
                if inc.required_capability in r.capabilities
                and self.route(states[r.id].location, inc.location) is not None
            ]
            for inc in incidents
        }


def _apply(states: dict[str, ResponderState], a: Assignment, location: str) -> dict[str, ResponderState]:
    """After a job the responder stays at the incident location until `finish` (decision 8)."""
    return {**states, a.responder_id: ResponderState(location, a.finish)}


def allocate(
    scenario: Scenario,
    costs: dict[Edge, EdgeCost],
    now: float,
    incidents: Sequence[Incident],
    states: dict[str, ResponderState],
    node_limit: int = NODE_LIMIT,
    prune: bool = True,
) -> AllocationResult:
    """Solve ONE round with backtracking + MRV + forward checking + branch and bound.

    prune=False switches branch and bound off (same answer, more nodes); only used by tests.
    """
    planner = _Planner(scenario, costs, now)
    domains = planner.domains(incidents, states)
    unhandled = sorted((i.id for i in incidents if not domains[i.id]), key=id_key)
    variables = [i for i in incidents if domains[i.id]]

    best_obj: Objective = (math.inf, math.inf)
    best: tuple[list[Assignment], list[str]] = ([], [])
    nodes = 0
    hit_limit = False

    def on_time(inc: Incident, sts: dict[str, ResponderState]) -> list[Assignment]:
        opts = (planner.option(inc, rid, sts[rid]) for rid in domains[inc.id])
        return [a for a in opts if a is not None and not a.late]

    def search(
        unassigned: list[Incident], sts: dict[str, ResponderState],
        chosen: list[Assignment], skipped: list[str], late_w: float, resp: float,
    ) -> None:
        nonlocal best_obj, best, nodes, hit_limit
        if hit_limit:
            return
        if not unassigned:  # complete assignment: keep it if strictly better
            if (late_w, resp) < best_obj:
                best_obj, best = (late_w, resp), (list(chosen), list(skipped))
            return

        # Forward checking: recompute every remaining incident's ON-TIME set in this branch.
        # An empty set does NOT fail the branch (deadlines are soft); it only means that incident
        # is guaranteed late, so its weight is added to a lower bound. The bound is valid because
        # assigning more jobs can only make a responder free later and somewhere else, and by the
        # triangle inequality of shortest paths that never makes it arrive earlier.
        ontime = {i.id: on_time(i, sts) for i in unassigned}
        bound = late_w + sum(WEIGHTS[i.priority] for i in unassigned if not ontime[i.id])
        if prune and (bound, resp) >= best_obj:  # branch and bound: cannot beat the best
            return

        # MRV: fewest on-time responders first (ties: higher weight, earlier deadline, id).
        inc = min(unassigned, key=lambda i: (len(ontime[i.id]), -WEIGHTS[i.priority],
                                             i.deadline, id_key(i.id)))
        rest = [i for i in unassigned if i is not inc]
        options = [a for rid in domains[inc.id]
                   if (a := planner.option(inc, rid, sts[rid])) is not None]
        options.sort(key=lambda a: (a.arrival, id_key(a.responder_id)))  # earliest arrival first

        if not options:  # nobody can reach it from where they are in THIS branch: counts as missed
            search(rest, sts, chosen, skipped + [inc.id], late_w + WEIGHTS[inc.priority], resp)
            return
        for a in options:
            nodes += 1
            if nodes > node_limit:
                hit_limit = True
                return
            search(rest, _apply(sts, a, inc.location), chosen + [a], skipped,
                   late_w + (WEIGHTS[inc.priority] if a.late else 0),
                   resp + (a.arrival - inc.timestamp))

    search(variables, states, [], [], 0.0, 0.0)
    assignments, skipped = best
    # (with node_limit >= number of incidents the first DFS leaf is always reached, so best is set)
    new_states = dict(states)
    by_id = {i.id: i for i in incidents}
    for a in sorted(assignments, key=lambda a: a.start):
        new_states[a.responder_id] = ResponderState(by_id[a.incident_id].location, a.finish)
    stats = AllocationStats(nodes, hit_limit, best_obj[0], best_obj[1])
    return assignments, sorted(unhandled + skipped, key=id_key), new_states, stats


def greedy(
    scenario: Scenario,
    costs: dict[Edge, EdgeCost],
    now: float,
    incidents: Sequence[Incident],
    states: dict[str, ResponderState],
) -> AllocationResult:
    """Baseline: incidents in id order, each gets the capable responder that arrives earliest."""
    planner = _Planner(scenario, costs, now)
    sts = dict(states)
    assignments: list[Assignment] = []
    unhandled: list[str] = []
    late_w = resp = 0.0
    for inc in sorted(incidents, key=lambda i: id_key(i.id)):
        options = [a for r in scenario.responders if inc.required_capability in r.capabilities
                   if (a := planner.option(inc, r.id, sts[r.id])) is not None]
        if not options:
            unhandled.append(inc.id)
            continue
        a = min(options, key=lambda a: (a.arrival, id_key(a.responder_id)))
        assignments.append(a)
        sts = _apply(sts, a, inc.location)
        late_w += WEIGHTS[inc.priority] if a.late else 0
        resp += a.arrival - inc.timestamp
    # nodes = one decision per incident (greedy never backtracks)
    return assignments, unhandled, sts, AllocationStats(len(incidents), False, late_w, resp)

