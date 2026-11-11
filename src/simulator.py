"""@brief Phase 6a: online simulation, one allocation round per incident timestamp.

No clairvoyance: a round at time t only sees the incidents with timestamp == t and the current
responder states. Assignments are committed immediately (decision 7); a later, more urgent incident
cannot take a responder back, it can only queue behind it.
Edge costs are computed ONCE from the t = 0 beliefs (decision 3): road reports are available at t = 0
and the traffic observations are historical, so no new evidence arrives during the simulation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.allocation import NODE_LIMIT, Assignment, allocate, greedy, id_key, initial_states
from src.cost_model import edge_costs
from src.data_loader import Incident, Scenario

POLICIES = {"csp": allocate, "greedy": greedy}


@dataclass(frozen=True)
class RoundStats:
    """@brief Statistics of one simulation round.

    @par Fields
    - time: the round's time (s).
    - incident_ids: the incidents of the round.
    - nodes, hit_limit, late_weight, response_time: copied from AllocationStats.
    """
    time: float
    incident_ids: tuple[str, ...]
    nodes: int
    hit_limit: bool
    late_weight: float
    response_time: float


@dataclass(frozen=True)
class SimulationResult:
    """@brief Result of a complete simulation.

    @par Fields
    - policy, tau, unobserved_traffic: the settings used.
    - assignments: all assignments of all rounds.
    - unhandled: ids of incidents nobody could reach.
    - rounds: per-round statistics.
    """
    policy: str
    tau: float
    unobserved_traffic: float
    assignments: list[Assignment]
    unhandled: list[str]
    rounds: list[RoundStats]

    @property
    def limit_rounds(self) -> list[float]:
        """@brief Times of the rounds in which the CSP hit its node limit.
        @return List of round times (empty in the provided scenarios).
        """
        return [r.time for r in self.rounds if r.hit_limit]


def rounds_by_time(incidents: list[Incident]) -> list[tuple[float, list[Incident]]]:
    """@brief Group incidents by timestamp, in ascending time order (ids in natural order).
    @param incidents All incidents of the scenario.
    @return List of (time, incidents of that time).
    """
    groups: dict[float, list[Incident]] = defaultdict(list)
    for inc in incidents:
        groups[inc.timestamp].append(inc)
    return [(t, sorted(groups[t], key=lambda i: id_key(i.id))) for t in sorted(groups)]


def simulate(
    scenario: Scenario, tau: float = 0.5, unobserved_traffic: float = 1.0, policy: str = "csp",
    unreported: str = "prior", node_limit: int = NODE_LIMIT,
) -> SimulationResult:
    """@brief Run the whole scenario round by round.
    @param scenario The scenario.
    @param tau Blocked threshold.
    @param unobserved_traffic Multiplier for roads without traffic observations.
    @param policy "csp" or "greedy".
    @param unreported "prior" or "open" (see cost_model.edge_costs).
    @param node_limit CSP node limit per round.
    @return The SimulationResult.
    @exception ValueError For an unknown policy.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}, choose from {sorted(POLICIES)}")
    solve = POLICIES[policy]
    costs = edge_costs(scenario, tau=tau, unobserved_traffic=unobserved_traffic,
                       unreported=unreported)  # once, t = 0
    states = initial_states(scenario)
    assignments: list[Assignment] = []
    unhandled: list[str] = []
    rounds: list[RoundStats] = []
    for t, incidents in rounds_by_time(scenario.incidents):
        # Only THIS round's incidents and the CURRENT states are passed: no look-ahead.
        if policy == "csp":
            new, missed, states, st = allocate(scenario, costs, t, incidents, states, node_limit)
        else:
            new, missed, states, st = solve(scenario, costs, t, incidents, states)
        assignments.extend(new)
        unhandled.extend(missed)
        rounds.append(RoundStats(t, tuple(i.id for i in incidents), st.nodes, st.hit_limit,
                                 st.late_weight, st.response_time))
    return SimulationResult(policy, tau, unobserved_traffic, assignments, unhandled, rounds)
