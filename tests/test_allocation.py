"""Phase 5 tests: CSP allocation for one round (t = 0)."""
import math
from itertools import permutations, product
from pathlib import Path

import pytest

from src.allocation import WEIGHTS, allocate, greedy, initial_states
from src.cost_model import edge_costs
from src.data_loader import Incident, load_scenario
from src.search import astar

SCENARIOS = Path(__file__).resolve().parent.parent / "Scenarios"


def round_zero(name: str):
    s = load_scenario(SCENARIOS / name)
    costs = edge_costs(s)
    incidents = [i for i in s.incidents if i.timestamp == 0]
    return s, costs, incidents, initial_states(s)


def test_small_t0_csp() -> None:
    s, costs, incidents, states = round_zero("small")
    assignments, unhandled, new_states, stats = allocate(s, costs, 0.0, incidents, states)
    got = {a.incident_id: a for a in assignments}
    assert unhandled == []
    assert {i: a.responder_id for i, a in got.items()} == {"I1": "R4", "I2": "R1", "I3": "R3", "I4": "R2"}
    assert got["I4"].arrival == pytest.approx(56.41, abs=0.01) and not got["I4"].late  # deadline 60
    assert got["I1"].start == 10.0  # R4 is busy until 10
    assert got["I1"].arrival == pytest.approx(81.09, abs=0.01) and got["I1"].late  # deadline 80
    assert got["I2"].path == ("N1", "H1", "N2")  # detour: N1->N2 is blocked at tau 0.5
    assert got["I2"].arrival == pytest.approx(92.24, abs=0.01) and got["I2"].late  # deadline 90
    assert got["I3"].arrival == pytest.approx(131.23, abs=0.01) and not got["I3"].late  # deadline 150
    assert stats.late_weight == 12  # I1 High 4 + I2 Critical 8
    assert not stats.hit_limit
    # after its job a responder waits at the incident location until it finishes
    assert new_states["R2"].location == "N2"
    assert new_states["R2"].free_at == pytest.approx(56.41 + 10, abs=0.01)
    print(f"\nsmall t=0: CSP nodes {stats.nodes}, late weight {stats.late_weight}, "
          f"response time {stats.response_time:.2f}")


def test_small_t0_greedy_is_worse() -> None:
    s, costs, incidents, states = round_zero("small")
    g_assign, _, _, g_stats = greedy(s, costs, 0.0, incidents, states)
    got = {a.incident_id: a for a in g_assign}
    assert got["I1"].responder_id == "R2"  # id order: I1 is handled first and grabs R2
    # I4 then goes to R2 QUEUED behind I1 (89.62 beats R4's 147.40), but it is still late
    assert got["I4"].responder_id == "R2"
    assert got["I4"].start == pytest.approx(23.31, abs=0.01)  # R2 finishes I1 at 13.31 + 10
    assert got["I4"].arrival == pytest.approx(89.62, abs=0.01) and got["I4"].late
    assert g_stats.late_weight == 16  # I2 8 + I4 8
    _, _, _, csp_stats = allocate(s, costs, 0.0, incidents, states)
    assert csp_stats.late_weight < g_stats.late_weight


def test_medium_t0_queueing_and_split() -> None:
    s, costs, incidents, states = round_zero("medium")
    assignments, unhandled, _, _ = allocate(s, costs, 0.0, incidents, states)
    got = {a.incident_id: a for a in assignments}
    assert got["I3"].responder_id == "R10" and got["I3"].arrival == 0.0  # R10 is already at N2
    assert got["I2"].responder_id == "R10"  # queued behind I3
    assert got["I2"].start == pytest.approx(10.0)  # I3 takes 10 s
    assert got["I2"].arrival == pytest.approx(20.78, abs=0.01) and got["I2"].late  # deadline 20
    # The map is split at tau 0.5: police unit R9 (N1) cannot reach the N2 side.
    assert astar(s, costs, "N1", "N2") is None and astar(s, costs, "N1", "N7") is None
    assert all(a.responder_id != "R9" for a in assignments)
    assert unhandled == []


def test_unknown_capability_is_unhandled() -> None:
    s, costs, incidents, states = round_zero("small")
    teleport = Incident("I99", "N1", "Critical", 100.0, "Teleportation", 0.0, 10.0)
    assignments, unhandled, _, _ = allocate(s, costs, 0.0, incidents + [teleport], states)
    assert unhandled == ["I99"]
    assert {a.incident_id for a in assignments} == {"I1", "I2", "I3", "I4"}  # the rest still assigned


def brute_force(s, costs, now, incidents, states):
    """Independent reference: EVERY order of the incidents x EVERY capable responder choice.

    Assignments are applied in that order with the same queueing rules. Shares only astar with
    the CSP. Returns (best objective, number of complete plans evaluated).
    """
    capable = [[r.id for r in s.responders if i.required_capability in r.capabilities]
               for i in incidents]
    routes: dict = {}  # cache only: same astar results, just not recomputed per plan
    best, evaluated = (math.inf, math.inf), 0
    for order in permutations(range(len(incidents))):
        for choice in product(*(capable[k] for k in order)):
            sts, late_w, resp, ok = dict(states), 0, 0.0, True
            for k, rid in zip(order, choice):
                inc, st = incidents[k], sts[rid]
                key = (st.location, inc.location)
                if key not in routes:
                    routes[key] = astar(s, costs, *key)
                route = routes[key]
                if route is None:
                    ok = False
                    break
                arrival = max(now, st.free_at) + route[1]
                late_w += WEIGHTS[inc.priority] if arrival > inc.deadline else 0
                resp += arrival - inc.timestamp
                sts[rid] = type(st)(inc.location, arrival + inc.time_required)
            if ok:
                evaluated += 1
                best = min(best, (late_w, resp))
    return best, evaluated


@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_csp_matches_brute_force_on_t0(name: str) -> None:
    s, costs, incidents, states = round_zero(name)
    _, _, _, stats = allocate(s, costs, 0.0, incidents, states)
    (bf_late, bf_resp), evaluated = brute_force(s, costs, 0.0, incidents, states)
    print(f"\n{name} t=0: CSP nodes {stats.nodes} vs brute force {evaluated} complete plans "
          f"({len(incidents)} incidents)")
    assert stats.late_weight == bf_late
    assert stats.response_time == pytest.approx(bf_resp)
    # The CSP explores fewer nodes than brute force evaluates plans. NOTE: this is weak evidence
    # for pruning: even WITHOUT pruning the CSP needs only 11 / 7 / 202 nodes, because it tries
    # one incident order per branch while brute force tries all orders. See the next test.
    assert stats.nodes < evaluated


@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_pruning_saves_nodes_but_never_changes_the_answer(name: str) -> None:
    s, costs, incidents, states = round_zero(name)
    _, _, _, pruned = allocate(s, costs, 0.0, incidents, states)
    _, _, _, full = allocate(s, costs, 0.0, incidents, states, prune=False)
    print(f"\n{name} t=0: nodes with pruning {pruned.nodes}, without {full.nodes}")
    assert (pruned.late_weight, pruned.response_time) == pytest.approx(
        (full.late_weight, full.response_time))
    assert pruned.nodes <= full.nodes
    if name != "medium":  # pruning cuts branches on small (7 vs 11) and large (28 vs 202);
        assert pruned.nodes < full.nodes  # on medium the first plan is optimal and nothing is cut


def assignment_trap():
    """Hand-made trap where the FIRST solution found is not optimal (only main roads, 30 m/s).

        R1 at P1:  P1 -> X  30 m (1 s)    P1 -> Y  60 m (2 s)
        R2 at P2:  P2 -> X  60 m (2 s)    P2 -> Y 300 m (10 s)
    I1 at X and I2 at Y, both Medium with loose deadlines (everything on time), so only the
    response time matters. MRV tie -> I1 first; earliest arrival -> R1; then I2 must take R2:
    1 + 10 = 11 s. The optimum swaps them: 2 + 2 = 4 s. X and Y have no outgoing roads, so a
    responder cannot do both jobs.
    """
    from src.data_loader import Node, Responder, Road, Scenario
    small = load_scenario(SCENARIOS / "small")
    nodes = {n: Node(n, n, x, 0, "normal") for n, x in
             [("Y", -60), ("P1", 0), ("X", 30), ("P2", 90)]}
    roads = {(a, b): Road(a, b, d, "main", 30) for a, b, d in
             [("P1", "X", 30), ("P1", "Y", 60), ("P2", "X", 60), ("P2", "Y", 300)]}
    responders = [Responder("R1", "Unit", "P1", frozenset({"Help"}), True, 0.0),
                  Responder("R2", "Unit", "P2", frozenset({"Help"}), True, 0.0)]
    incidents = [Incident("I1", "X", "Medium", 1000.0, "Help", 0.0, 1.0),
                 Incident("I2", "Y", "Medium", 1000.0, "Help", 0.0, 1.0)]
    s = Scenario(nodes, roads, responders, incidents, {}, {}, small.road_model, small.traffic_model)
    return s, edge_costs(s), incidents, initial_states(s)


def test_branch_and_bound_improves_on_first_solution() -> None:
    s, costs, incidents, states = assignment_trap()
    mult = costs[("P1", "X")].status_mult  # same prior multiplier on every road (no reports)
    assignments, _, _, stats = allocate(s, costs, 0.0, incidents, states)
    assert {a.incident_id: a.responder_id for a in assignments} == {"I1": "R2", "I2": "R1"}
    assert stats.late_weight == 0
    assert stats.response_time == pytest.approx(4 * mult)
    assert stats.response_time == pytest.approx(brute_force(s, costs, 0.0, incidents, states)[0][1])
    # greedy falls into the trap
    _, _, _, g_stats = greedy(s, costs, 0.0, incidents, states)
    assert g_stats.response_time == pytest.approx(11 * mult)
