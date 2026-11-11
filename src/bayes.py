"""Phase 2: Bayesian road status (Notes 6, inference by enumeration).

Network per directed road:   Weather -> Status -> {Citizen, Police, Sensor}
  query    : Status S (Open / Congested / Blocked)
  evidence : the reports, and the weather if it was reported
  hidden   : the weather when it is NOT reported (summed out with the prior)

Because every report depends only on S, the reports are conditionally independent given S,
so their likelihoods can simply be multiplied:
    P(S | w, r1..rn)  ∝  P(S | w) · Π_i P(r_i | S)
P(w) is the same constant for every value of S, so normalisation removes it.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.data_loader import Edge, RoadModel, RoadReport, Scenario


def normalise(dist: dict[str, float]) -> dict[str, float]:
    """Divide by the total so the values sum to 1 (this is the '∝' step)."""
    total = sum(dist.values())
    if total <= 0:
        raise ValueError(f"cannot normalise, all probabilities are zero: {dist}")
    return {k: v / total for k, v in dist.items()}


def status_prior(model: RoadModel, weather: str | None) -> dict[str, float]:
    """P(S | w) if the weather is observed, else P(S) = Σ_w P(w) · P(S | w) (sum out weather)."""
    if weather is not None:
        return dict(model.status_given_weather[weather])
    statuses = next(iter(model.status_given_weather.values())).keys()
    return {
        s: sum(p_w * model.status_given_weather[w][s] for w, p_w in model.weather_prior.items())
        for s in statuses
    }


def status_posterior(
    model: RoadModel, weather: str | None, reports: Sequence[RoadReport]
) -> dict[str, float]:
    """P(S | weather, reports) ∝ P(S | weather) · Π P(report | S), normalised.

    weather=None -> weather is hidden and summed out with the prior.
    Any subset of sources may be present (a missing source just contributes no factor).
    """
    unnormalised = status_prior(model, weather)
    for report in reports:
        cpt = model.report_given_status[report.source]  # outer key = TRUE status
        for s in unnormalised:
            unnormalised[s] *= cpt[s][report.reported_status]
    return normalise(unnormalised)


def road_status_beliefs(scenario: Scenario) -> dict[Edge, dict[str, float]]:
    """Posterior status for EVERY directed road.

    Reported road: weather comes from its reports (the loader guarantees they agree).
    Unreported road: no weather, no reports -> prior marginal P(S).
    """
    beliefs: dict[Edge, dict[str, float]] = {}
    for edge in scenario.roads:
        reports = scenario.reports.get(edge, [])
        weather = reports[0].weather if reports else None
        beliefs[edge] = status_posterior(scenario.road_model, weather, reports)
    return beliefs
