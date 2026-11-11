"""Phase 4 tests: cost model, heuristic properties and A* vs UCS."""
import math
from pathlib import Path

import pytest

from src.cost_model import edge_costs
from src.data_loader import Node, Road, Scenario, load_scenario
from src.search import MAX_SPEED, astar, straight_line_time, ucs

SCENARIOS = Path(__file__).resolve().parent.parent / "Scenarios"
NAMES = ["small", "medium", "large"]


@pytest.mark.parametrize("name", NAMES)
def test_roads_not_shorter_than_straight_line(name: str) -> None:
    s = load_scenario(SCENARIOS / name)
    ratios = []
    for road in s.roads.values():
        a, b = s.nodes[road.source], s.nodes[road.target]
        straight = math.hypot(a.x - b.x, a.y - b.y)
        assert road.distance >= straight, (road.source, road.target)
        ratios.append(road.distance / straight)
        assert road.speed <= MAX_SPEED  # no road is faster than the heuristic assumes
    print(f"\n{name}: smallest distance / straight-line ratio = {min(ratios):.4f}")


@pytest.mark.parametrize("name", NAMES)
def test_heuristic_is_consistent(name: str) -> None:
    s = load_scenario(SCENARIOS / name)
    costs = edge_costs(s)
    for goal in s.nodes:
        h = {n: straight_line_time(s, n, goal) for n in s.nodes}
        assert h[goal] == 0
        for (u, v), c in costs.items():
            if c.usable:
                assert c.cost >= h[u] - h[v] - 1e-9, (u, v, goal)


def floyd_warshall(s: Scenario, costs: dict) -> dict[tuple[str, str], float]:
    """Independent all-pairs reference (shares NO code with search.py).

    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j]) for every intermediate node k.
    """
    inf = math.inf
    dist = {(i, j): (0.0 if i == j else inf) for i in s.nodes for j in s.nodes}
    for (u, v), c in costs.items():
        if c.usable:
            dist[u, v] = min(dist[u, v], c.cost)
    for k in s.nodes:
        for i in s.nodes:
            for j in s.nodes:
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist


@pytest.mark.parametrize("name", NAMES)
def test_astar_equals_ucs_on_every_pair(name: str) -> None:
    # Differential testing. A* and UCS share _best_first, so a bug inside it can make both wrong in
    # the same way and "A* == UCS" still passes. Hence two INDEPENDENT checks on every pair:
    #   (a) recompute the returned path's cost from its roads (all must exist and be usable);
    #   (b) compare with Floyd-Warshall, an all-pairs algorithm written here in the test.
    s = load_scenario(SCENARIOS / name)
    costs = edge_costs(s)
    reference = floyd_warshall(s, costs)
    total_astar = total_ucs = pairs = 0
    for start in s.nodes:
        for goal in s.nodes:
            a, u = astar(s, costs, start, goal), ucs(s, costs, start, goal)
            assert (a is None) == (u is None), (start, goal)
            if a is None:
                assert reference[start, goal] == math.inf, (start, goal)  # really unreachable
                continue
            pairs += 1
            path, cost, expanded = a
            assert path[0] == start and path[-1] == goal
            # (a) the path is real and its roads add up to the returned cost
            road_costs = [costs[(x, y)].cost for x, y in zip(path, path[1:])]
            assert all(c is not None for c in road_costs), (start, goal, path)
            assert cost == pytest.approx(sum(road_costs)), (start, goal)
            # (b) independent reference: the cost is really optimal
            assert cost == pytest.approx(reference[start, goal]), (start, goal)
            assert u[1] == pytest.approx(reference[start, goal]), (start, goal)
            assert expanded <= u[2], (start, goal)  # A* never expands more than UCS
            total_astar += expanded
            total_ucs += u[2]
    print(f"\n{name}: {pairs} reachable pairs, expanded A* {total_astar} vs UCS {total_ucs}")


def test_small_blocked_roads_and_tau() -> None:
    s = load_scenario(SCENARIOS / "small")
    costs = edge_costs(s, tau=0.5)
    for edge in [("H1", "N3"), ("N3", "H1")]:
        assert costs[edge].p_blocked == pytest.approx(0.998, abs=1e-3)
        assert not costs[edge].usable
    assert costs[("N1", "N2")].p_blocked == pytest.approx(0.527, abs=1e-3)
    assert not costs[("N1", "N2")].usable  # 0.527 >= 0.5
    assert edge_costs(s, tau=0.6)[("N1", "N2")].usable  # 0.527 < 0.6
    # the reverse direction is a different road: all its reports say Congested
    assert costs[("N2", "N1")].usable


def test_cost_parts_multiply() -> None:
    s = load_scenario(SCENARIOS / "small")
    for c in edge_costs(s, tau=1.1).values():  # tau > 1: nothing is cut off
        assert c.status_mult >= 1 and c.traffic_mult >= 1
        assert c.cost == pytest.approx(c.base * c.status_mult * c.traffic_mult)


def tiny_scenario() -> Scenario:
    """A --secondary 1000 m, 15 m/s--> B   (shortest in metres, 66.7 s)
       A --main 708 m--> C --main 708 m--> B   (1416 m but 30 m/s, 47.2 s)
       D is isolated (unreachable)."""
    small = load_scenario(SCENARIOS / "small")
    nodes = {
        "A": Node("A", "A", 0, 0, "normal"),
        "B": Node("B", "B", 1000, 0, "normal"),
        "C": Node("C", "C", 500, 500, "normal"),  # |AC| = |CB| = 707.1 m
        "D": Node("D", "D", 0, 1000, "normal"),
    }
    roads = {
        ("A", "B"): Road("A", "B", 1000, "secondary", 15),
        ("A", "C"): Road("A", "C", 708, "main", 30),
        ("C", "B"): Road("C", "B", 708, "main", 30),
    }
    return Scenario(nodes, roads, [], [], {}, {}, small.road_model, small.traffic_model)


def test_fastest_not_shortest() -> None:
    s = tiny_scenario()
    costs = edge_costs(s)
    path, cost, _ = astar(s, costs, "A", "B")
    assert path == ["A", "C", "B"]  # longer in metres, but faster
    # no reports and no traffic: every road gets the same prior status multiplier (~1.104)
    mult = costs[("A", "C")].status_mult
    assert cost == pytest.approx((708 / 30 + 708 / 30) * mult)
    assert costs[("A", "B")].cost > cost


def test_unreachable_and_trivial() -> None:
    s = tiny_scenario()
    costs = edge_costs(s)
    assert astar(s, costs, "A", "D") is None
    assert ucs(s, costs, "B", "A") is None  # roads are directed: no way back
    assert astar(s, costs, "A", "A") == (["A"], 0.0, 0)
