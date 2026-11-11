"""Phase 3: HMM traffic model (Notes 8, forward algorithm / filtering).

Per directed road:  X_1 -> X_2 -> ... -> X_n     hidden: true traffic level (Low/Medium/High)
                     |      |             |
                    E_1    E_2           E_n     evidence: observed traffic level

Forward algorithm, for every observation:
  1. time elapse : B'(x_t) = Σ_{x_{t-1}} P(x_t | x_{t-1}) · B(x_{t-1})   (skipped for t = 1)
  2. observe     : B(x_t) ∝ P(e_t | x_t) · B'(x_t), then normalise
Convention: model.initial = P(X_1), the state that emits observation 1 (no transition before it).
Normalising every step keeps the numbers from underflowing; the normalising constants are
z_t = P(e_t | e_1..e_{t-1}), so their product is the likelihood P(e_1..e_n) (chain rule).
We use forward (filtering), not Viterbi: we need the distribution of the CURRENT state,
not the single most likely history.

Works with float and with fractions.Fraction (only +, *, / are used).
"""
from __future__ import annotations

from collections.abc import Sequence

from src.data_loader import Edge, Scenario, TrafficModel

Belief = dict[str, float]  # also used with Fraction values in the tests

DEFAULT_MULTIPLIERS: dict[str, float] = {"Low": 1.0, "Medium": 1.25, "High": 1.5}


def predict(model: TrafficModel, belief: Belief, steps: int = 1) -> Belief:
    """Time elapse only: B'(y) = Σ_x B(x) · P(y | x), repeated `steps` times."""
    for _ in range(steps):
        belief = {
            y: sum(belief[x] * model.transition[x][y] for x in belief)
            for y in model.initial
        }
    return belief


def observe(model: TrafficModel, belief: Belief, evidence: str) -> tuple[Belief, float]:
    """Weight by the emission P(e | x) and normalise. Returns (new belief, normaliser z)."""
    weighted = {x: belief[x] * model.emission[x][evidence] for x in belief}
    z = sum(weighted.values())  # z = P(e_t | e_1..e_{t-1})
    if z == 0:
        raise ValueError(f"observation {evidence!r} has probability 0 under the current belief")
    return {x: p / z for x, p in weighted.items()}, z


def forward(model: TrafficModel, observations: Sequence[str]) -> tuple[Belief, float]:
    """Filter: returns (P(X_n | e_1..e_n), likelihood P(e_1..e_n))."""
    belief: Belief = dict(model.initial)  # P(X_1): no transition before observation 1
    likelihood = 1  # int 1 keeps Fraction exact; becomes float with float tables
    for t, evidence in enumerate(observations):
        if t > 0:
            belief = predict(model, belief)  # 1. time elapse  X_{t-1} -> X_t
        belief, z = observe(model, belief, evidence)  # 2. observe + normalise
        likelihood *= z
    return belief, likelihood


def traffic_beliefs(scenario: Scenario) -> dict[Edge, Belief]:
    """Traffic 'now' (t = 0): filtered belief after the last observation + ONE prediction step.

    Only roads WITH observations are returned; the cost model uses multiplier 1.0 for the rest.
    """
    result: dict[Edge, Belief] = {}
    for edge, observations in scenario.traffic_obs.items():
        filtered, _ = forward(scenario.traffic_model, observations)
        result[edge] = predict(scenario.traffic_model, filtered, steps=1)
    return result


def expected_multiplier(
    belief: Belief, multipliers: dict[str, float] = DEFAULT_MULTIPLIERS
) -> float:
    """E[mult] = Σ_x P(x) · mult(x)  (expected value over the traffic belief)."""
    return sum(p * multipliers[x] for x, p in belief.items())
