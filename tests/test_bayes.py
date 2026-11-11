"""Phase 2 tests: Bayesian road status posterior."""
from pathlib import Path

import pytest

from src.bayes import road_status_beliefs, status_posterior
from src.data_loader import RoadReport, load_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "Scenarios"


@pytest.fixture(scope="module")
def small():
    return load_scenario(SCENARIOS / "small")


def test_small_n1_n2_hand_check(small) -> None:
    # Rain; Citizen=Blocked, Police=Blocked, Sensor=Congested.
    # Unnormalised  P(S|Rain) · P(cit=B|S) · P(pol=B|S) · P(sen=C|S):
    #   Open      : 0.60 · 0.10 · 0.02 · 0.10 = 0.00012
    #   Congested : 0.30 · 0.20 · 0.08 · 0.80 = 0.00384
    #   Blocked   : 0.10 · 0.70 · 0.90 · 0.07 = 0.00441
    #   sum = 0.00837 -> O 0.0143, C 0.4588, B 0.5269
    post = status_posterior(small.road_model, "Rain", small.reports[("N1", "N2")])
    assert post["Open"] == pytest.approx(0.014, abs=1e-3)
    assert post["Congested"] == pytest.approx(0.459, abs=1e-3)
    assert post["Blocked"] == pytest.approx(0.527, abs=1e-3)
    # and road_status_beliefs gives the same answer for that road
    assert road_status_beliefs(small)[("N1", "N2")] == post


def test_no_reports_with_weather_is_the_cpt_row(small) -> None:
    post = status_posterior(small.road_model, "Clear", [])
    assert post == pytest.approx(small.road_model.status_given_weather["Clear"])


def test_no_reports_no_weather_is_the_prior_marginal(small) -> None:
    # P(S) = Σ_w P(w) P(S|w), e.g. Open = 0.5·0.95 + 0.3·0.60 + 0.2·0.15 = 0.685
    post = status_posterior(small.road_model, None, [])
    assert post == pytest.approx({"Open": 0.685, "Congested": 0.18, "Blocked": 0.135})


def test_one_source_missing(small) -> None:
    # Only Citizen + Police (no Sensor): O .6·.1·.02, C .3·.2·.08, B .1·.7·.9
    reports = [RoadReport("Rain", "Citizen", "Blocked"), RoadReport("Rain", "Police", "Blocked")]
    post = status_posterior(small.road_model, "Rain", reports)
    o, c, b = 0.6 * 0.1 * 0.02, 0.3 * 0.2 * 0.08, 0.1 * 0.7 * 0.9
    assert post == pytest.approx({"Open": o / (o + c + b), "Congested": c / (o + c + b),
                                  "Blocked": b / (o + c + b)})
    assert sum(post.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_every_road_gets_a_distribution(name: str) -> None:
    s = load_scenario(SCENARIOS / name)
    beliefs = road_status_beliefs(s)
    assert set(beliefs) == set(s.roads)  # EVERY road, reported or not
    for edge, post in beliefs.items():
        assert set(post) == {"Open", "Congested", "Blocked"}
        assert sum(post.values()) == pytest.approx(1.0), edge
        if edge not in s.reports:  # unreported -> prior marginal
            assert post == pytest.approx({"Open": 0.685, "Congested": 0.18, "Blocked": 0.135})
