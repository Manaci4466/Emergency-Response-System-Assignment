"""@brief Phase 4b: A* route search (Notes 1), graph search with a closed set.

    f(n) = g(n) + h(n)
    g(n) = expected travel time from the start to n along the path found (s)
    h(n) = straight-line distance(n, goal) / max_speed(scenario)   (fastest road in THIS scenario)

h is ADMISSIBLE: every road is at least as long as the straight line between its ends, no road
is faster than max_speed, and every cost multiplier is ≥ 1, so h never overestimates.
h is CONSISTENT: for a road (u, v), cost(u,v) ≥ dist(u,v)/max_speed ≥ h(u) - h(v)
(triangle inequality). With a closed set this matters: a consistent h guarantees that the first
time a node is popped it has its optimal g, so never re-opening closed nodes is safe.
Both properties are checked by tests, not just claimed.

The first premise is a property of the DATA, so it is checked for every scenario: if some road is
shorter than the straight line between its ends, h could overestimate, and we fall back to h = 0
(uniform-cost search), which is always optimal.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Callable

from src.cost_model import EdgeCost
from src.data_loader import Edge, Scenario

Heuristic = Callable[[str], float]
SearchResult = tuple[list[str], float, int]  # (path, cost in seconds, number of expanded nodes)


def _straight_line(scenario: Scenario, a: str, b: str) -> float:
    """@brief Euclidean distance between two nodes, in metres.
    @param scenario The scenario (node coordinates).
    @param a First node id.
    @param b Second node id.
    @return The straight-line distance.
    """
    na, nb = scenario.nodes[a], scenario.nodes[b]
    return math.hypot(na.x - nb.x, na.y - nb.y)


def max_speed(scenario: Scenario) -> float:
    """@brief The fastest road speed in this scenario (30 m/s in the provided ones).
    @param scenario The scenario.
    @return The maximum road speed in m/s.
    """
    return max((r.speed for r in scenario.roads.values()), default=1.0)


def straight_line_violations(scenario: Scenario) -> list[Edge]:
    """@brief Roads that are SHORTER than the straight line between their ends.

    None exist in the provided scenarios; if any exist, the straight-line heuristic could
    overestimate and A* falls back to uniform-cost search.
    @param scenario The scenario.
    @return The offending roads, sorted.
    """
    return sorted(e for e, r in scenario.roads.items()
                  if r.distance < _straight_line(scenario, *e) - 1e-9)


def heuristic_is_admissible(scenario: Scenario) -> bool:
    """@brief Whether the data satisfies the premise of the straight-line heuristic.
    @param scenario The scenario.
    @return True if no road is shorter than the straight line between its ends.
    """
    return not straight_line_violations(scenario)


def heuristic_description(scenario: Scenario) -> str:
    """@brief One-line description of the route search used (written to statistics.txt).
    @param scenario The scenario.
    @return E.g. "A*, h = straight-line distance / 30 m/s (admissible, consistent)".
    """
    bad = straight_line_violations(scenario)
    if bad:
        roads = ", ".join(f"{a}->{b}" for a, b in bad)
        return f"h = 0 (uniform-cost search): road(s) {roads} shorter than the straight line"
    return f"A*, h = straight-line distance / {max_speed(scenario):g} m/s (admissible, consistent)"


def straight_line_time(scenario: Scenario, a: str, b: str) -> float:
    """@brief Straight-line distance / max_speed: a lower bound on the travel time from a to b.
    @param scenario The scenario.
    @param a Start node id.
    @param b Goal node id.
    @return The lower bound in seconds.
    """
    return _straight_line(scenario, a, b) / max_speed(scenario)


def _best_first(
    scenario: Scenario, costs: dict[Edge, EdgeCost], start: str, goal: str, h: Heuristic
) -> SearchResult | None:
    """@brief Best-first graph search with a closed set; A* and UCS differ only in h.

    The frontier is a heap of (f, node, g, path), so equal f values are broken alphabetically by node
    id. The goal test happens when a node is POPPED (not when generated), which makes the result
    optimal for a consistent h.
    @param scenario The scenario (adjacency list).
    @param costs Edge costs; unusable roads (cost None) are skipped.
    @param start Start node id.
    @param goal Goal node id.
    @param h Heuristic function of a node id.
    @return (path, cost in seconds, number of expanded nodes), or None if the goal is unreachable.
    """
    if start == goal:
        return [start], 0.0, 0
    # Heap entries (f, node, g, path): equal f is broken by the node id, i.e. alphabetically.
    frontier: list[tuple[float, str, float, list[str]]] = [(h(start), start, 0.0, [start])]
    closed: set[str] = set()
    expanded = 0
    while frontier:
        _f, node, g, path = heapq.heappop(frontier)
        if node in closed:
            continue  # an older, worse copy of a node we already expanded
        if node == goal:  # goal test when POPPED, not when generated -> optimal
            return path, g, expanded
        closed.add(node)
        expanded += 1
        for road in scenario.out_edges[node]:
            edge_cost = costs[(road.source, road.target)].cost
            if edge_cost is None or road.target in closed:  # unusable road / already done
                continue
            g2 = g + edge_cost
            heapq.heappush(frontier, (g2 + h(road.target), road.target, g2, path + [road.target]))
    return None  # goal unreachable


def astar(
    scenario: Scenario, costs: dict[Edge, EdgeCost], start: str, goal: str
) -> SearchResult | None:
    """@brief Fastest route with A* (falls back to UCS if the heuristic premise fails on the data).
    @param scenario The scenario.
    @param costs Edge costs from cost_model.edge_costs.
    @param start Start node id.
    @param goal Goal node id.
    @return (path, cost in seconds, expanded nodes), or None if unreachable.
    """
    if not heuristic_is_admissible(scenario):  # data breaks the premise: fall back to UCS
        return ucs(scenario, costs, start, goal)
    speed = max_speed(scenario)
    return _best_first(scenario, costs, start, goal,
                       lambda n: _straight_line(scenario, n, goal) / speed)


def ucs(
    scenario: Scenario, costs: dict[Edge, EdgeCost], start: str, goal: str
) -> SearchResult | None:
    """@brief Uniform-cost search = A* with h = 0 (the reference to check A* against).
    @param scenario The scenario.
    @param costs Edge costs.
    @param start Start node id.
    @param goal Goal node id.
    @return (path, cost in seconds, expanded nodes), or None if unreachable.
    """
    return _best_first(scenario, costs, start, goal, lambda n: 0.0)
