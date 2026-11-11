"""@brief Phase 5: responder allocation as a CSP, solved per round (Notes 2: backtracking,
MRV, forward checking) with branch and bound on a lexicographic objective.

One round = the incidents that become known at time `now` (no clairvoyance).
  - Variables : this round's incidents.
  - Domains   : responders with the required capability that can reach the incident
                (unary constraints, filtered first; empty domain -> unhandled).
  - Hard      : capability, reachability, one job at a time (queueing: the next job starts when the
                previous one finishes, at its location).
  - Soft      : deadlines. Late help is better than no help, so lateness is a PENALTY, never a
                reason to fail a branch.
  - Objective : lexicographic (1) Σ priority weight of LATE incidents, (2) Σ (arrival - timestamp).

Because the assignment order also fixes the job order of a shared responder, MRV here affects the
QUALITY of the solution, not only the speed (in a classical CSP variable order only changes speed).
Without MRV (plain id order) medium t=0 gets worse: R10 does I2 before I3, and the Critical I3
arrives late at 41.6 s (late weight 8 instead of 4).

@note Limitation: the jobs of one responder are done in the order the search assigns them, and the
search order is fixed by MRV, so the CSP is not exhaustive over every job order.
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
    """@brief Where a responder is and when it can start its next job.

    @par Fields
    - location: node id (start location, or the location of its last incident).
    - free_at: absolute time (s) at which the responder can start a new job.
    """
    location: str
    free_at: float  # absolute time (s) at which the responder can start a new job


@dataclass(frozen=True)
class Assignment:
    """@brief One responder assigned to one incident.

    @par Fields
    - responder_id, incident_id: the pair.
    - path: route from the responder's location to the incident.
    - start: time the responder leaves (the plan's "time").
    - arrival: start + expected travel time.
    - finish: arrival + time_required; the responder is then free at the incident location.
    - late: arrival > deadline.
    """
    responder_id: str
    incident_id: str
    path: tuple[str, ...]
    start: float  # = the plan's "time": when the responder leaves
    arrival: float
    finish: float  # arrival + time_required; the responder is then free at the incident
    late: bool


@dataclass(frozen=True)
class AllocationStats:
    """@brief Search statistics of one round.

    @par Fields
    - nodes: assignments tried (search nodes).
    - hit_limit: True if the node limit was reached (best solution found so far returned).
    - late_weight: Σ priority weights of late (or forced-missed) incidents of the chosen plan.
    - response_time: Σ (arrival - timestamp) of the chosen plan.
    """
    nodes: int  # assignments tried (search nodes)
    hit_limit: bool  # True -> node limit reached, best solution found so far returned
    late_weight: float
    response_time: float


AllocationResult = tuple[list[Assignment], list[str], dict[str, ResponderState], AllocationStats]


def id_key(identifier: str) -> tuple[str, int]:
    """@brief Natural sort key for ids, so I2 sorts before I10.
    @param identifier An id such as "I10" or "R2".
    @return (prefix, number), or (identifier, 0) if it has no number.
    """
    m = re.fullmatch(r"(\D*)(\d+)", identifier)
    return (m.group(1), int(m.group(2))) if m else (identifier, 0)


def initial_states(scenario: Scenario) -> dict[str, ResponderState]:
    """@brief Responder states at t = 0: start location and time_until_available.
    @param scenario The scenario.
    @return Responder id -> ResponderState.
    """
    return {r.id: ResponderState(r.start_location, r.time_until_available) for r in scenario.responders}


class _Planner:
    """@brief Helpers shared by the CSP and the greedy baseline for one round (routes are cached)."""

    def __init__(self, scenario: Scenario, costs: dict[Edge, EdgeCost], now: float) -> None:
        """@brief Create a planner for the round at time `now`.
        @param scenario The scenario.
        @param costs Edge costs.
        @param now Time of the round (s).
        """
        self.scenario = scenario
        self.costs = costs
        self.now = now
        self._routes: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    def route(self, a: str, b: str) -> tuple[list[str], float] | None:
        """@brief Cached A* route from a to b.
        @param a Start node id.
        @param b Goal node id.
        @return (path, travel time in s), or None if unreachable.
        """
        if (a, b) not in self._routes:
            res = astar(self.scenario, self.costs, a, b)
            self._routes[(a, b)] = None if res is None else (res[0], res[1])
        return self._routes[(a, b)]

    def option(self, inc: Incident, rid: str, st: ResponderState) -> Assignment | None:
        """@brief The assignment that results from sending responder rid (in state st) to an incident.

        start = max(now, free_at) (queueing), arrival = start + A* time, finish = arrival + time_required.
        @param inc The incident.
        @param rid The responder id.
        @param st The responder's current state in this branch.
        @return The Assignment, or None if the responder cannot reach the incident.
        """
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
        """@brief Unary constraints: capability and reachability from the responder's current location.
        @param incidents The round's incidents.
        @param states Current responder states.
        @return Incident id -> ids of the responders in its domain.
        """
        return {
            inc.id: [
                r.id for r in self.scenario.responders
                if inc.required_capability in r.capabilities
                and self.route(states[r.id].location, inc.location) is not None
            ]
            for inc in incidents
        }


def _apply(states: dict[str, ResponderState], a: Assignment, location: str) -> dict[str, ResponderState]:
    """@brief New states after an assignment: the responder waits at the incident until `finish` (decision 8).
    @param states Current responder states.
    @param a The assignment.
    @param location The incident location.
    @return A new state dict (the input is not modified).
    """
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
    """@brief Solve ONE round with backtracking + MRV + forward checking + branch and bound.

    Forward checking (Notes 2), adapted to soft deadlines: instead of pruning domains, it recomputes
    every remaining incident's ON-TIME set after each assignment. An empty set does not fail the branch: its weight is added to a lower bound, and the
    branch is pruned when (bound, response time so far) ≥ the best plan found. The bound is valid
    because extra jobs only make a responder free later and elsewhere, which by the triangle
    inequality of shortest paths never makes it arrive earlier.
    @param scenario The scenario.
    @param costs Edge costs.
    @param now Time of the round (s).
    @param incidents The incidents that become known at `now`.
    @param states Responder states at the start of the round.
    @param node_limit Maximum search nodes; if hit, the best plan found so far is used (or greedy if
           no complete plan was found yet).
    @param prune False switches branch and bound off (same answer, more nodes); used by tests only.
    @return (assignments, unhandled incident ids, new responder states, AllocationStats).
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
        """@brief Assignments of `inc` that arrive on time, given the branch's responder states."""
        opts = (planner.option(inc, rid, sts[rid]) for rid in domains[inc.id])
        return [a for a in opts if a is not None and not a.late]

    def search(
        unassigned: list[Incident], sts: dict[str, ResponderState],
        chosen: list[Assignment], skipped: list[str], late_w: float, resp: float,
    ) -> None:
        """@brief Recursive backtracking over the unassigned incidents of one branch."""
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
    if best_obj == (math.inf, math.inf):
        # The node limit was hit before even ONE complete plan was found (only possible when
        # node_limit < number of incidents): fall back to greedy so no incident silently vanishes.
        g_assign, g_unhandled, g_states, g_stats = greedy(scenario, costs, now, incidents, states)
        return g_assign, g_unhandled, g_states, AllocationStats(
            nodes, True, g_stats.late_weight, g_stats.response_time)
    assignments, skipped = best
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
    """@brief Baseline: incidents in id order, each gets the capable responder that arrives earliest.
    @param scenario The scenario.
    @param costs Edge costs.
    @param now Time of the round (s).
    @param incidents The round's incidents.
    @param states Responder states at the start of the round.
    @return Same format as allocate (nodes = one decision per incident).
    """
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

