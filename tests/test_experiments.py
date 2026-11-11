"""Phase 8 tests: experiment parameters and the Monte Carlo replay."""
import dataclasses
from pathlib import Path

import pytest

from src.bayes import road_status_beliefs
from src.cost_model import edge_costs
from src.data_loader import load_scenario
from src.experiments import metrics, monte_carlo, stationary_multiplier, usable_roads
from src.hmm import stationary_distribution
from src.simulator import simulate

SCENARIOS = Path(__file__).resolve().parent.parent / "Scenarios"
NAMES = ["small", "medium", "large"]


@pytest.mark.parametrize("name", NAMES)
def test_tau_monotonicity(name: str) -> None:
    """A higher tau never makes fewer roads usable (usable sets are nested)."""
    s = load_scenario(SCENARIOS / name)
    taus = [k / 20 for k in range(21)]  # 0.0, 0.05, ..., 1.0
    sets = [usable_roads(s, t) for t in taus]
    for lower, higher in zip(sets, sets[1:]):
        assert lower <= higher


def test_unreported_open_only_changes_unreported_roads() -> None:
    s = load_scenario(SCENARIOS / "small")
    prior, open_ = edge_costs(s), edge_costs(s, unreported="open")
    for e in s.roads:
        if e in s.reports:
            assert open_[e] == prior[e]
        else:
            assert prior[e].p_blocked == pytest.approx(0.135)
            assert (open_[e].p_blocked, open_[e].status_mult) == (0.0, 1.0)
    with pytest.raises(ValueError):
        edge_costs(s, unreported="closed")


def test_stationary_distribution() -> None:
    pi = stationary_distribution(load_scenario(SCENARIOS / "small").traffic_model)
    assert pi == pytest.approx({"Low": 45 / 127, "Medium": 49 / 127, "High": 33 / 127}, abs=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_same_result(name: str) -> None:
    s = load_scenario(SCENARIOS / name)
    plan = simulate(s).assignments
    a = monte_carlo(s, plan, n=500, seed=7)
    assert a == monte_carlo(s, plan, n=500, seed=7)
    assert 0.0 <= a.p_any_route_blocked <= 1.0


def certain_world(name: str):
    """Scenario copy where every road is certainly Open and every traffic level certainly Low.

    P(S | w) = Open for every weather -> every posterior (and the prior marginal) is Open, whatever
    the reports say. HMM: P(X1 = Low) = 1 and Low -> Low with probability 1 -> belief Low forever.
    """
    s = load_scenario(SCENARIOS / name)
    open_row = {"Open": 1.0, "Congested": 0.0, "Blocked": 0.0}
    road_model = dataclasses.replace(
        s.road_model, status_given_weather={w: dict(open_row) for w in s.road_model.weather_prior})
    tm = s.traffic_model
    traffic_model = dataclasses.replace(
        tm, initial={"Low": 1.0, "Medium": 0.0, "High": 0.0},
        transition={**tm.transition, "Low": {"Low": 1.0, "Medium": 0.0, "High": 0.0}})
    return dataclasses.replace(s, road_model=road_model, traffic_model=traffic_model, out_edges={})


@pytest.mark.parametrize("name", NAMES)
def test_certain_world_reproduces_the_deterministic_plan(name: str) -> None:
    s = certain_world(name)
    res = simulate(s)
    planned = metrics(s, res)
    mc = monte_carlo(s, res.assignments, n=200, seed=1)
    assert mc.p_any_route_blocked == 0.0
    # every world is identical to the plan: no spread at all
    assert mc.mean_deadlines_met == mc.p5_deadlines_met == mc.p95_deadlines_met == planned.deadlines_met_pct
    late = {a.incident_id for a in res.assignments if a.late} | set(res.unhandled)
    assert mc.missed == {i.id: (1.0 if i.id in late else 0.0) for i in s.incidents}


def test_small_i2_route_blocked_in_about_a_quarter_of_worlds() -> None:
    s = load_scenario(SCENARIOS / "small")
    mc = monte_carlo(s, simulate(s).assignments, n=5000, seed=2026)
    # R1's route N1->H1->N2 uses two unreported roads, P(Blocked) = 0.135 each (independent)
    expected = 1 - (1 - 0.135) ** 2  # 0.252
    assert mc.route_blocked["I2"] == pytest.approx(expected, abs=0.02)  # ~3.3 standard errors


def test_blocked_route_strands_the_responder() -> None:
    """Force the first road of R1's route to I2 to be Blocked: I2 is missed in EVERY world, and so
    is R1's next job I5, whose fixed route starts at I2's location, where R1 never arrives."""
    s = certain_world("small")
    res = simulate(s)
    r1 = sorted((a for a in res.assignments if a.responder_id == "R1"), key=lambda a: a.start)
    assert [a.incident_id for a in r1] == ["I2", "I5"]
    blocked_road = (r1[0].path[0], r1[0].path[1])  # in this certain world: N1 -> N2 directly
    status = road_status_beliefs(s)
    status[blocked_road] = {"Open": 0.0, "Congested": 0.0, "Blocked": 1.0}
    mc = monte_carlo(s, res.assignments, status=status, n=50, seed=3)
    assert mc.p_any_route_blocked == 1.0
    assert mc.route_blocked["I2"] == 1.0
    assert mc.missed["I2"] == 1.0 and mc.missed["I5"] == 1.0  # stranded
    others = {a.incident_id for a in res.assignments if a.responder_id != "R1" and not a.late
              and blocked_road not in zip(a.path, a.path[1:])}
    assert others and all(mc.missed[i] == 0.0 for i in others)  # other responders unaffected


def test_stationary_multiplier_is_per_scenario() -> None:
    """medium has its own traffic tables: its stationary multiplier is 1.25, not small's 1.2264."""
    small, medium, large = (load_scenario(SCENARIOS / n) for n in NAMES)
    assert stationary_multiplier(small) == pytest.approx(155.75 / 127)  # π = (45, 49, 33)/127
    assert stationary_multiplier(large) == pytest.approx(155.75 / 127)
    assert stationary_multiplier(medium) == pytest.approx(1.25)


def test_experiment_3_uses_each_scenarios_own_multiplier() -> None:
    from src.experiments import unobserved_traffic_rows
    scen = {n: load_scenario(SCENARIOS / n) for n in NAMES}
    mults = {n: m for n, m, _ in unobserved_traffic_rows(scen) if m != 1.0}
    assert mults["medium"] == pytest.approx(1.25)
    assert mults["small"] == pytest.approx(155.75 / 127)
