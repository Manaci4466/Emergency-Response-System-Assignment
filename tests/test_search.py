"""Phase 4 tests: cost model, heuristic properties and A* vs UCS."""
import math
from pathlib import Path

import pytest

from src.cost_model import edge_costs
from src.data_loader import Node, Road, Scenario, load_scenario
from src.search import (astar, heuristic_description, heuristic_is_admissible, max_speed,
                        straight_line_time, ucs)

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
        assert road.speed <= max_speed(s)  # no road is faster than the heuristic assumes
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


# ---------------------------------------------------------- robustness for unseen scenarios

def _copy_with_road(name: str, edge: tuple[str, str], **changes) -> Scenario:
    import dataclasses
    s = load_scenario(SCENARIOS / name)
    roads = dict(s.roads)
    roads[edge] = dataclasses.replace(roads[edge], **changes)
    return dataclasses.replace(s, roads=roads, out_edges={})


def _all_pairs_match_reference(s: Scenario) -> None:
    costs = edge_costs(s)
    reference = floyd_warshall(s, costs)
    for start in s.nodes:
        for goal in s.nodes:
            res = astar(s, costs, start, goal)
            if res is None:
                assert reference[start, goal] == math.inf
            else:
                assert res[1] == pytest.approx(reference[start, goal]), (start, goal)


def test_heuristic_speed_comes_from_the_scenario() -> None:
    s = _copy_with_road("small", ("N1", "H1"), speed=40.0)  # one road faster than 30 m/s
    assert max_speed(s) == 40.0
    assert "40 m/s" in heuristic_description(s)
    costs = edge_costs(s)
    for goal in s.nodes:  # still consistent with the new speed
        for (u, v), c in costs.items():
            if c.usable:
                assert c.cost >= straight_line_time(s, u, goal) - straight_line_time(s, v, goal) - 1e-9
    _all_pairs_match_reference(s)


def test_road_shorter_than_straight_line_falls_back_to_ucs() -> None:
    s = _copy_with_road("large", ("N16", "N17"), distance=1.0)  # shorter than the straight line
    assert not heuristic_is_admissible(s)
    assert heuristic_description(s).startswith("h = 0")
    costs = edge_costs(s)
    for start in s.nodes:  # fallback = exactly UCS (same path, cost and expansions)
        assert astar(s, costs, start, "N1") == ucs(s, costs, start, "N1")
    _all_pairs_match_reference(s)  # and still optimal


def test_fallback_is_needed() -> None:
    """Without the fallback, A* really returns a suboptimal route on a copy with a too-short road.

    medium with N9->N7 shortened to 1 m: the straight-line h overestimates near N7, so plain A*
    (forced through _best_first with that h) returns N2->N7 at 10.8 s, while the optimum is 7.7 s.
    astar() detects the bad data, falls back to UCS and finds the optimum.
    """
    from src.search import _best_first, _straight_line
    s = _copy_with_road("medium", ("N9", "N7"), distance=1.0)
    costs = edge_costs(s)
    optimum = floyd_warshall(s, costs)["N2", "N7"]
    speed = max_speed(s)
    plain = _best_first(s, costs, "N2", "N7", lambda n: _straight_line(s, n, "N7") / speed)
    assert plain[1] == pytest.approx(10.78, abs=0.01) and optimum == pytest.approx(7.7, abs=0.05)
    assert plain[1] > optimum + 1  # plain A* is wrong here
    assert astar(s, costs, "N2", "N7")[1] == pytest.approx(optimum)  # the fallback is right
