"""@brief Phase 3: HMM traffic model (Notes 8, forward algorithm / filtering).

Per directed road:  X_1 -> X_2 -> ... -> X_n (hidden true traffic level Low/Medium/High), each X_t
emitting E_t (the observed level).

Forward algorithm, for every observation:
  1. time elapse : B'(x_t) = Σ_{x_t-1} P(x_t | x_t-1) · B(x_t-1)   (skipped for t = 1)
  2. observe     : B(x_t) ∝ P(e_t | x_t) · B'(x_t), then normalise

Convention: model.initial = P(X_1), the state that emits observation 1 (no transition before it).
Normalising every step keeps the numbers from underflowing; the normalising constants are
z_t = P(e_t | e_1..e_t-1), so their product is the likelihood P(e_1..e_n) (chain rule).
We use forward (filtering), not Viterbi: we need the distribution of the CURRENT state, not the
single most likely history. Works with float and with fractions.Fraction (only +, *, / are used).
"""
from __future__ import annotations

from collections.abc import Sequence

from src.data_loader import Edge, Scenario, TrafficModel

Belief = dict[str, float]  # also used with Fraction values in the tests

DEFAULT_MULTIPLIERS: dict[str, float] = {"Low": 1.0, "Medium": 1.25, "High": 1.5}


def predict(model: TrafficModel, belief: Belief, steps: int = 1) -> Belief:
    """@brief Time elapse only: B'(y) = Σ_x B(x) · P(y | x), repeated `steps` times.
    @param model The HMM tables.
    @param belief Current belief over the traffic level.
    @param steps Number of time steps to predict.
    @return The predicted belief.
    """
    for _ in range(steps):
        belief = {
            y: sum(belief[x] * model.transition[x][y] for x in belief)
            for y in model.initial
        }
    return belief


def observe(model: TrafficModel, belief: Belief, evidence: str) -> tuple[Belief, float]:
    """@brief Observation step: weight the belief by P(e | x) and normalise.
    @param model The HMM tables.
    @param belief Belief after the time-elapse step.
    @param evidence The observed traffic level.
    @return (new belief, normaliser z = P(e_t | e_1..e_t-1)).
    @exception ValueError If the observation has probability 0 under the belief.
    """
    weighted = {x: belief[x] * model.emission[x][evidence] for x in belief}
    z = sum(weighted.values())  # z = P(e_t | e_1..e_{t-1})
    if z == 0:
        raise ValueError(f"observation {evidence!r} has probability 0 under the current belief")
    return {x: p / z for x, p in weighted.items()}, z


def forward(model: TrafficModel, observations: Sequence[str]) -> tuple[Belief, float]:
    """@brief Forward algorithm (filtering) over one observation sequence.
    @param model The HMM tables.
    @param observations Observed levels in time order.
    @return (P(X_n | e_1..e_n), likelihood P(e_1..e_n)).
    @note Checked exactly with fractions on the practicum umbrella example: P(Rain_3 | +,+,-) = 2/5,
    likelihood 15/128.
    """
    belief: Belief = dict(model.initial)  # P(X_1): no transition before observation 1
    likelihood = 1  # int 1 keeps Fraction exact; becomes float with float tables
    for t, evidence in enumerate(observations):
        if t > 0:
            belief = predict(model, belief)  # 1. time elapse  X_{t-1} -> X_t
        belief, z = observe(model, belief, evidence)  # 2. observe + normalise
        likelihood *= z
    return belief, likelihood


def traffic_beliefs(scenario: Scenario) -> dict[Edge, Belief]:
    """@brief Traffic "now" (t = 0) for every road with observations (design decision 3).

    Filtered belief after the last observation plus ONE prediction step. Roads without observations
    are not returned; the cost model gives them the unobserved-traffic multiplier (default 1.0).
    @param scenario The loaded scenario.
    @return (from, to) -> belief over Low / Medium / High.
    """
    result: dict[Edge, Belief] = {}
    for edge, observations in scenario.traffic_obs.items():
        filtered, _ = forward(scenario.traffic_model, observations)
        result[edge] = predict(scenario.traffic_model, filtered, steps=1)
    return result


def expected_multiplier(
    belief: Belief, multipliers: dict[str, float] = DEFAULT_MULTIPLIERS
) -> float:
    """@brief Expected traffic multiplier E[mult] = Σ_x P(x) · mult(x) (design decision 4).
    @param belief Belief over the traffic level.
    @param multipliers Multiplier per level (default Low 1.0, Medium 1.25, High 1.5).
    @return The expected multiplier (between 1.0 and 1.5 with the defaults).
    """
    return sum(p * multipliers[x] for x, p in belief.items())


def stationary_distribution(model: TrafficModel, tol: float = 1e-12, max_steps: int = 100_000) -> Belief:
    """@brief Stationary distribution π with π = π · T.

    Found by predicting from the initial belief until the belief stops changing. Predicting far
    ahead converges to π, so far-ahead beliefs lose their road-specific information.
    @param model The HMM tables.
    @param tol Convergence tolerance (largest change per step).
    @param max_steps Upper bound on the number of steps.
    @return π (small and large: (45, 49, 33)/127; medium has its own tables).
    @exception RuntimeError If it does not converge within max_steps.
    """
    belief: Belief = dict(model.initial)
    for _ in range(max_steps):
        nxt = predict(model, belief)
        if max(abs(nxt[x] - belief[x]) for x in belief) < tol:
            return nxt
        belief = nxt
    raise RuntimeError("prediction did not converge")
