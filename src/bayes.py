"""@brief Phase 2: Bayesian road status (Notes 6, inference by enumeration).

Network per directed road:   Weather -> Status -> {Citizen, Police, Sensor}
  - query    : Status S (Open / Congested / Blocked)
  - evidence : the reports, and the weather if it was reported
  - hidden   : the weather when it is NOT reported (summed out with the prior)

S is each report's ONLY parent and there are no edges between the reports, so the reports are
conditionally independent given S and their likelihoods can simply be multiplied:
    P(S | w, r1..rn)  ∝  P(S | w) · Π_i P(r_i | S)
P(w) is the same constant for every value of S, so normalisation removes it.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.data_loader import Edge, RoadModel, RoadReport, Scenario


def normalise(dist: dict[str, float]) -> dict[str, float]:
    """@brief Divide by the total so the values sum to 1 (the '∝' step).
    @param dist Unnormalised non-negative values.
    @return The normalised distribution.
    @exception ValueError If all values are zero.
    """
    total = sum(dist.values())
    if total <= 0:
        raise ValueError(f"cannot normalise, all probabilities are zero: {dist}")
    return {k: v / total for k, v in dist.items()}


def status_prior(model: RoadModel, weather: str | None) -> dict[str, float]:
    """@brief Prior over the road status, before the reports are used.

    If the weather is observed this is the table row P(S | w); otherwise the weather is summed out:
    P(S) = Σ_w P(w) · P(S | w), which gives Open 0.685, Congested 0.18, Blocked 0.135 (decision 2).
    @param model The Bayes-net tables.
    @param weather Observed weather, or None if unknown.
    @return Status -> probability.
    """
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
    """@brief Posterior P(S | weather, reports) ∝ P(S | weather) · Π P(report | S), normalised.

    Any subset of sources may be present: a missing source simply contributes no factor.
    @param model The Bayes-net tables (report CPTs: outer key = TRUE status).
    @param weather Observed weather, or None to sum it out with the prior.
    @param reports The reports for this road.
    @return Status -> posterior probability.
    @note Example: small N1->N2 (Rain; Blocked, Blocked, Congested) gives O 0.014, C 0.459, B 0.527.
    """
    unnormalised = status_prior(model, weather)
    for report in reports:
        cpt = model.report_given_status[report.source]  # outer key = TRUE status
        for s in unnormalised:
            unnormalised[s] *= cpt[s][report.reported_status]
    return normalise(unnormalised)


def road_status_beliefs(scenario: Scenario) -> dict[Edge, dict[str, float]]:
    """@brief Posterior status for EVERY directed road of the scenario.

    A reported road takes its weather from its reports (the loader guarantees they agree); an
    unreported road has no weather and no reports and gets the prior marginal P(S).
    @param scenario The loaded scenario.
    @return (from, to) -> status distribution.
    """
    beliefs: dict[Edge, dict[str, float]] = {}
    for edge in scenario.roads:
        reports = scenario.reports.get(edge, [])
        weather = reports[0].weather if reports else None
        beliefs[edge] = status_posterior(scenario.road_model, weather, reports)
    return beliefs
