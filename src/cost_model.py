"""Phase 4a: expected travel time per directed road.

    cost = base · status_mult · traffic_mult       (seconds)
    base         = distance / speed
    status_mult  = (P(O)·1 + P(C)·1.5) / (P(O) + P(C))   expected slowdown GIVEN the road is passable
    traffic_mult = E[traffic multiplier] over the HMM belief (or `unobserved_traffic` without data)
A road is unusable (cost None) if P(Blocked) >= tau (design decision 1).

Every multiplier is >= 1, so cost >= distance / speed. The A* heuristic relies on that.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.bayes import road_status_beliefs
from src.data_loader import Edge, Scenario
from src.hmm import expected_multiplier, traffic_beliefs

STATUS_MULT: dict[str, float] = {"Open": 1.0, "Congested": 1.5}


@dataclass(frozen=True)
class EdgeCost:
    """All parts are kept so they can be printed and checked separately."""
    base: float  # distance / speed (s)
    p_blocked: float  # P(Blocked | evidence) from the Bayes net
    status_mult: float
    traffic_mult: float
    cost: float | None  # expected travel time (s), None = unusable

    @property
    def usable(self) -> bool:
        return self.cost is not None


def status_multiplier(belief: dict[str, float]) -> float:
    """Expected status multiplier, conditioned on the road NOT being blocked."""
    passable = belief["Open"] + belief["Congested"]
    if passable == 0:
        return float("inf")  # certainly blocked; only reachable if tau > 1
    return (belief["Open"] * STATUS_MULT["Open"] + belief["Congested"] * STATUS_MULT["Congested"]) / passable


CERTAIN_OPEN: dict[str, float] = {"Open": 1.0, "Congested": 0.0, "Blocked": 0.0}


def edge_costs(
    scenario: Scenario, tau: float = 0.5, unobserved_traffic: float = 1.0,
    unreported: str = "prior",
) -> dict[Edge, EdgeCost]:
    """unreported="prior": roads without reports get the prior marginal (decision 2, P(B) = 0.135).
    unreported="open": they are assumed open (P(Blocked) 0, status multiplier 1.0); experiment only.
    """
    if unreported not in ("prior", "open"):
        raise ValueError(f"unreported must be 'prior' or 'open', not {unreported!r}")
    status = road_status_beliefs(scenario)  # Phase 2: every road
    if unreported == "open":
        status = {e: (b if e in scenario.reports else CERTAIN_OPEN) for e, b in status.items()}
    traffic = traffic_beliefs(scenario)  # Phase 3: only roads with observations
    costs: dict[Edge, EdgeCost] = {}
    for edge, road in scenario.roads.items():
        base = road.distance / road.speed
        p_blocked = status[edge]["Blocked"]
        s_mult = status_multiplier(status[edge])
        t_mult = expected_multiplier(traffic[edge]) if edge in traffic else unobserved_traffic
        cost = None if p_blocked >= tau else base * s_mult * t_mult
        costs[edge] = EdgeCost(base, p_blocked, s_mult, t_mult, cost)
    return costs
