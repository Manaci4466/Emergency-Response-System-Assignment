"""Phase 3 tests: forward algorithm and traffic beliefs."""
from fractions import Fraction as F
from pathlib import Path

import pytest

from src.data_loader import TrafficModel, load_scenario
from src.hmm import expected_multiplier, forward, predict, traffic_beliefs

SCENARIOS = Path(__file__).resolve().parent.parent / "Scenarios"


def umbrella() -> TrafficModel:
    """Classic umbrella HMM: states R(ain)/D(ry), evidence +/- (umbrella or not), exact fractions."""
    return TrafficModel(
        initial={"R": F(1, 2), "D": F(1, 2)},
        transition={"R": {"R": F(3, 4), "D": F(1, 4)}, "D": {"R": F(1, 4), "D": F(3, 4)}},
        emission={"R": {"+": F(3, 4), "-": F(1, 4)}, "D": {"+": F(1, 4), "-": F(3, 4)}},
    )


def test_umbrella_exact() -> None:
    # By hand:  t1 '+': R 3/8, D 1/8 -> z=1/2,  B=(3/4, 1/4)
    #           predict -> R 5/8; '+': R 15/32, D 3/32 -> z=9/16, B=(5/6, 1/6)
    #           predict -> R 2/3; '-': R 1/6,  D 1/4  -> z=5/12, B=(2/5, 3/5)
    #           likelihood = 1/2 · 9/16 · 5/12 = 15/128
    belief, likelihood = forward(umbrella(), ["+", "+", "-"])
    assert belief["R"] == F(2, 5)  # exact, no tolerance
    assert likelihood == F(15, 128)


def test_initial_is_p_x1_no_transition_before_first_observation() -> None:
    # Convention: model.initial = P(X1), the state that EMITS observation 1, so there is no
    # time-elapse step before it. With ONE observation "Low":
    #   B(x) ∝ P(Low | x) · P(X1 = x) = (0.85·0.6, 0.10·0.3, 0.02·0.1) = (0.51, 0.03, 0.002)
    #   normalised by 0.542 -> (0.941, 0.0554, 0.0037)
    # The off-by-one bug (predict before observation 1) would give (0.925, 0.069, 0.005).
    # Why the other tests cannot catch that bug:
    #   - umbrella: its prior (1/2, 1/2) is the stationary distribution of its symmetric T,
    #     so one extra transition changes nothing;
    #   - scenario data: after ~10 observations the start is forgotten (the HMM converges),
    #     the difference on N4->H1 is ~1e-7, far below the 1e-3 tolerance.
    # Only a single observation keeps the starting belief visible.
    model = load_scenario(SCENARIOS / "small").traffic_model
    belief, likelihood = forward(model, ["Low"])
    assert belief == pytest.approx({"Low": 0.941, "Medium": 0.0554, "High": 0.0037}, abs=1e-4)
    assert likelihood == pytest.approx(0.542)  # = P(E1 = Low) = Σ_x P(Low|x) P(X1=x)


@pytest.mark.parametrize(
    "name, edge, p_high",
    [("small", ("N4", "H1"), 0.548), ("large", ("N16", "N17"), 0.704)],
)
def test_predicted_high(name: str, edge: tuple[str, str], p_high: float) -> None:
    beliefs = traffic_beliefs(load_scenario(SCENARIOS / name))
    assert beliefs[edge]["High"] == pytest.approx(p_high, abs=1e-3)


def test_prediction_converges_to_stationary_distribution() -> None:
    model = load_scenario(SCENARIOS / "small").traffic_model
    from_low = predict(model, {"Low": 1.0, "Medium": 0.0, "High": 0.0}, steps=50)
    from_high = predict(model, {"Low": 0.0, "Medium": 0.0, "High": 1.0}, steps=50)
    print("\nstationary distribution:", {k: round(v, 4) for k, v in from_low.items()})
    # The start no longer matters. Convergence is geometric: the 2nd eigenvalue of T is 0.742,
    # so the error after 50 steps is about 0.742^50 ≈ 3e-7. 1e-6 is safe; 1e-9 would not be.
    assert from_low == pytest.approx(from_high, abs=1e-6)
    stationary = {"Low": F(45, 127), "Medium": F(49, 127), "High": F(33, 127)}
    assert from_low == pytest.approx({k: float(v) for k, v in stationary.items()}, abs=1e-6)

    # Exact check with fractions: π = π · T holds with no rounding at all.
    exact = TrafficModel(
        initial=stationary,
        transition={x: {y: F(str(p)) for y, p in row.items()} for x, row in model.transition.items()},
        emission={},
    )
    assert predict(exact, stationary) == stationary


@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_every_belief_sums_to_one(name: str) -> None:
    s = load_scenario(SCENARIOS / name)
    beliefs = traffic_beliefs(s)
    assert set(beliefs) == set(s.traffic_obs)  # only roads WITH observations
    for edge, belief in beliefs.items():
        assert sum(belief.values()) == pytest.approx(1.0), edge
        assert 1.0 <= expected_multiplier(belief) <= 1.5


def test_expected_multiplier() -> None:
    assert expected_multiplier({"Low": 0.5, "Medium": 0.3, "High": 0.2}) == pytest.approx(
        0.5 * 1.0 + 0.3 * 1.25 + 0.2 * 1.5
    )
