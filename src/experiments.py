"""@brief Phase 8: experiments. Run with `python -m src.experiments`; writes results/experiments.md.

1-4: sensitivity of the design decisions (tau, unreported roads, unobserved traffic, policy).

5-6: Monte Carlo robustness (Notes 6, sampling). The plan is made with EXPECTED costs; here we
sample N "true" worlds and replay the FIXED plan in each:
  - each road's true status ~ its Bayes posterior (unreported roads: the prior marginal),
  - each observed road's true traffic level ~ its HMM belief at t = 0 (unobserved roads: Low),
  - a job starts at max(its planned time, the responder's finish time in this world),
  - a route that contains a road that is Blocked in this world fails: the incident is missed, and
    the responder is stranded, so its later jobs (whose fixed routes start at the incident it never
    reached) are missed too.

No re-planning: this measures the plan, not the planner.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.allocation import Assignment, id_key
from src.bayes import road_status_beliefs
from src.cost_model import STATUS_MULT, edge_costs
from src.data_loader import Edge, Scenario, load_scenario
from src.hmm import DEFAULT_MULTIPLIERS, expected_multiplier, stationary_distribution, traffic_beliefs
from src.simulator import SimulationResult, simulate
from src.statistics import compute_statistics

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["small", "medium", "large"]
TAUS = [0.1, 0.5, 0.6, 0.9]
N_WORLDS = 5000
SEED = 2026

Dist = dict[str, float]


# ------------------------------------------------------------------ deterministic metrics

@dataclass(frozen=True)
class Metrics:
    """@brief The deterministic metrics compared across experiments.

    @par Fields
    - deadlines_met_pct: on-time / all incidents (%).
    - weighted_missed: Σ priority weight of late or unhandled incidents.
    - critical_on_time: "on time/total" for Critical incidents.
    - avg_response: average response time over handled incidents (s).
    - unhandled: number of unhandled incidents.
    """
    deadlines_met_pct: float
    weighted_missed: int  # Σ priority weight of late OR unhandled incidents
    critical_on_time: str  # "on time / total"
    avg_response: float | None
    unhandled: int


def metrics(scenario: Scenario, result: SimulationResult) -> Metrics:
    """@brief Metrics of a simulation result (reuses statistics.compute_statistics).
    @param scenario The scenario.
    @param result The simulation result.
    @return The Metrics.
    """
    st = compute_statistics(scenario.incidents, result.assignments, result.unhandled)
    return Metrics(st.deadlines_met_pct, st.weighted_missed, st.critical_on_time,
                   st.avg_response_time, len(st.unhandled))


def _fmt(m: Metrics) -> str:
    """@brief Format Metrics as the cells of a markdown table row.
    @param m The metrics.
    @return "met% | weighted | critical | avg | unhandled".
    """
    avg = "n/a" if m.avg_response is None else f"{m.avg_response:.1f}"
    return (f"{m.deadlines_met_pct:.1f}% | {m.weighted_missed} | {m.critical_on_time} | "
            f"{avg} | {m.unhandled}")


METRIC_HEADER = "Deadlines met | Weighted missed | Critical on time | Avg response (s) | Unhandled"


def usable_roads(scenario: Scenario, tau: float) -> set[Edge]:
    """@brief Roads that are usable at a given tau.
    @param scenario The scenario.
    @param tau Blocked threshold.
    @return Set of usable (from, to) roads.
    """
    return {e for e, c in edge_costs(scenario, tau=tau).items() if c.usable}


def tau_changes(scenario: Scenario, tau: float, reference: float = 0.5) -> str:
    """@brief Which roads open or close compared with a reference tau, grouped by P(Blocked).
    @param scenario The scenario.
    @param tau The tau to compare.
    @param reference The reference tau (default 0.5).
    @return A one-line explanation, e.g. "opens N1→N2, N2→N4 (P = 0.527)".
    """
    if tau == reference:
        return "reference"
    costs = edge_costs(scenario, tau=tau)
    ref = usable_roads(scenario, reference)
    changed: dict[float, list[str]] = {}
    for e, c in costs.items():
        if c.usable != (e in ref):
            changed.setdefault(round(c.p_blocked, 3), []).append(f"{e[0]}→{e[1]}")
    if not changed:
        return "no road changes"
    verb = "opens" if tau > reference else "closes"
    parts = []
    for p, roads in sorted(changed.items()):
        listed = ", ".join(sorted(roads)) if len(roads) <= 4 else f"{len(roads)} roads"
        parts.append(f"{listed} (P = {p})")
    return f"{verb} " + "; ".join(parts)


# ------------------------------------------------------------------ Monte Carlo

@dataclass(frozen=True)
class MonteCarloResult:
    """@brief Result of replaying one plan in N sampled worlds.

    @par Fields
    - worlds: number of sampled worlds.
    - p_any_route_blocked: fraction of worlds in which at least one planned route is blocked.
    - mean_deadlines_met, p5_deadlines_met, p95_deadlines_met: deadlines met (%) over the worlds.
    - most_missed: the 3 incidents missed most often, with their fraction.
    - missed: every incident -> fraction of worlds in which it is missed (late or failed).
    - route_blocked: incident -> fraction of worlds in which its planned route is blocked.
    """
    worlds: int
    p_any_route_blocked: float
    mean_deadlines_met: float
    p5_deadlines_met: float
    p95_deadlines_met: float
    most_missed: list[tuple[str, float]]  # (incident id, fraction of worlds missed), top 3
    missed: dict[str, float]  # every incident -> fraction of worlds in which it is missed
    route_blocked: dict[str, float]  # incident id -> fraction of worlds its planned route is blocked


def _sample(dist: Dist, rng: random.Random) -> str:
    """@brief Draw one value from a discrete distribution (inverse CDF, keys in a fixed order).
    @param dist Value -> probability.
    @param rng The random generator (seeded, so results are reproducible).
    @return The sampled value.
    """
    u, acc = rng.random(), 0.0
    for value, p in dist.items():
        acc += p
        if u < acc:
            return value
    return next(reversed(dist))  # u fell in the rounding gap at the very top


def _percentile(sorted_vals: list[float], q: float) -> float:
    """@brief Nearest-rank percentile of a sorted list.
    @param sorted_vals Values in ascending order.
    @param q Quantile between 0 and 1.
    @return The percentile value.
    """
    return sorted_vals[round(q * (len(sorted_vals) - 1))]


def monte_carlo(
    scenario: Scenario,
    assignments: list[Assignment],
    status: dict[Edge, Dist] | None = None,
    traffic: dict[Edge, Dist] | None = None,
    n: int = N_WORLDS,
    seed: int = SEED,
) -> MonteCarloResult:
    """@brief Replay a fixed plan in n sampled worlds (Notes 6, sampling).
    @param scenario The scenario.
    @param assignments The plan to replay (its routes are fixed).
    @param status Road status beliefs to sample from (default: the Bayes posteriors).
    @param traffic Traffic beliefs to sample from (default: the HMM beliefs at t = 0).
    @param n Number of worlds.
    @param seed Random seed (same seed -> identical result).
    @return The MonteCarloResult.
    """
    status = road_status_beliefs(scenario) if status is None else status
    traffic = traffic_beliefs(scenario) if traffic is None else traffic
    incidents = {i.id: i for i in scenario.incidents}
    edges_of = {a.incident_id: list(zip(a.path, a.path[1:])) for a in assignments}
    needed = sorted({e for es in edges_of.values() for e in es})
    jobs_by_responder: dict[str, list[Assignment]] = {}
    for a in sorted(assignments, key=lambda a: (a.start, id_key(a.incident_id))):
        jobs_by_responder.setdefault(a.responder_id, []).append(a)

    rng = random.Random(seed)
    met: list[float] = []
    any_blocked = 0
    missed = Counter()
    blocked_route = Counter()
    for _ in range(n):
        # 1. sample one world (only the roads the plan uses matter)
        world_status = {e: _sample(status[e], rng) for e in needed}
        world_level = {e: _sample(traffic[e], rng) if e in traffic else "Low" for e in needed}
        if any(s == "Blocked" for s in world_status.values()):
            any_blocked += 1
        # 2. replay the fixed plan
        on_time = 0
        for jobs in jobs_by_responder.values():
            free_at, stranded = 0.0, False
            for a in jobs:
                inc = incidents[a.incident_id]
                route = edges_of[a.incident_id]
                if any(world_status[e] == "Blocked" for e in route):
                    blocked_route[a.incident_id] += 1
                if stranded or any(world_status[e] == "Blocked" for e in route):
                    stranded = True
                    missed[a.incident_id] += 1
                    continue
                start = max(a.start, free_at)
                travel = sum(scenario.roads[e].distance / scenario.roads[e].speed
                             * STATUS_MULT[world_status[e]] * DEFAULT_MULTIPLIERS[world_level[e]]
                             for e in route)
                arrival = start + travel
                if arrival <= inc.deadline:
                    on_time += 1
                else:
                    missed[a.incident_id] += 1
                free_at = arrival + inc.time_required
        for iid in incidents:  # unhandled in the plan: missed in every world
            if iid not in edges_of:
                missed[iid] += 1
        met.append(100.0 * on_time / len(incidents))

    met.sort()
    top = sorted(missed.items(), key=lambda kv: (-kv[1], id_key(kv[0])))[:3]
    return MonteCarloResult(
        worlds=n,
        p_any_route_blocked=any_blocked / n,
        mean_deadlines_met=sum(met) / n,
        p5_deadlines_met=_percentile(met, 0.05),
        p95_deadlines_met=_percentile(met, 0.95),
        most_missed=[(i, c / n) for i, c in top],
        missed={i: missed[i] / n for i in incidents},
        route_blocked={i: blocked_route[i] / n for i in edges_of},
    )


def stationary_multiplier(scenario: Scenario) -> float:
    """@brief Expected traffic multiplier under THIS scenario's stationary distribution.
    @param scenario The scenario (medium has its own traffic tables: 1.25; small and large: 1.2264).
    @return The expected multiplier.
    """
    return expected_multiplier(stationary_distribution(scenario.traffic_model))


def unobserved_traffic_rows(scen: dict[str, Scenario]) -> list[tuple[str, float, Metrics]]:
    """@brief Experiment 3: each scenario with unobserved traffic ×1.0 and ×its own stationary multiplier.
    @param scen Scenario name -> scenario.
    @return List of (name, multiplier, metrics).
    """
    rows = []
    for n, s in scen.items():
        for mult in [1.0, stationary_multiplier(s)]:
            rows.append((n, mult, metrics(s, simulate(s, unobserved_traffic=mult))))
    return rows


# ------------------------------------------------------------------ report

def run_all() -> str:
    """@brief Run all experiments.
    @return The markdown text of results/experiments.md.
    """
    scen = {n: load_scenario(ROOT / "Scenarios" / n) for n in NAMES}
    out: list[str] = ["# Experiments", "",
                      "Generated by `python -m src.experiments`. All times are EXPECTED seconds under "
                      "the model unless stated otherwise. Weighted missed = Σ priority weight "
                      "(Critical 8, High 4, Medium 2, Low 1) of late or unhandled incidents.", ""]

    out += ["## 1. Threshold tau (CSP)", "",
            f"| Scenario | tau | Usable roads | {METRIC_HEADER} | Change vs tau 0.5 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in NAMES:
        for tau in TAUS:
            m = metrics(scen[n], simulate(scen[n], tau=tau))
            out.append(f"| {n} | {tau} | {len(usable_roads(scen[n], tau))}/{len(scen[n].roads)} | "
                       f"{_fmt(m)} | {tau_changes(scen[n], tau)} |")

    out += ["", "## 2. Unreported roads (decision 2), CSP, tau 0.5", "",
            f"| Scenario | Unreported roads | {METRIC_HEADER} |", "|---|---|---|---|---|---|---|"]
    for n in NAMES:
        for mode in ["prior", "open"]:
            out.append(f"| {n} | {mode} | {_fmt(metrics(scen[n], simulate(scen[n], unreported=mode)))} |")

    out += ["", "## 3. Unobserved traffic (decision 5), CSP, tau 0.5", "",
            "The alternative multiplier is the expected multiplier under each scenario's OWN "
            "stationary distribution (medium has different traffic tables).", "",
            f"| Scenario | Unobserved multiplier | {METRIC_HEADER} |", "|---|---|---|---|---|---|---|"]
    for n, mult, m in unobserved_traffic_rows(scen):
        out.append(f"| {n} | {mult:.4f} | {_fmt(m)} |")

    out += ["", "## 4. CSP vs greedy, tau 0.5", "",
            f"| Scenario | Policy | {METRIC_HEADER} |", "|---|---|---|---|---|---|---|"]
    for n in NAMES:
        for policy in ["csp", "greedy"]:
            out.append(f"| {n} | {policy} | {_fmt(metrics(scen[n], simulate(scen[n], policy=policy)))} |")

    out += ["", f"## 5. Monte Carlo robustness of the CSP plan (tau 0.5, N = {N_WORLDS}, seed {SEED})",
            "", "| Scenario | Planned deadlines met | P(≥1 planned route blocked) | Mean deadlines met "
            "| 5th–95th percentile | Most often missed (fraction of worlds) |",
            "|---|---|---|---|---|---|"]
    i2_line = ""
    for n in NAMES:
        res = simulate(scen[n])
        planned = metrics(scen[n], res).deadlines_met_pct
        mc = monte_carlo(scen[n], res.assignments)
        top = ", ".join(f"{i} ({f:.2f})" for i, f in mc.most_missed)
        out.append(f"| {n} | {planned:.1f}% | {mc.p_any_route_blocked:.3f} | "
                   f"{mc.mean_deadlines_met:.1f}% | {mc.p5_deadlines_met:.1f}–{mc.p95_deadlines_met:.1f}% "
                   f"| {top} |")
        if n == "small":
            a = next(x for x in res.assignments if x.incident_id == "I2")
            i2_line = (f"Small I2 (Critical): {a.responder_id}'s route {'→'.join(a.path)} is blocked in "
                       f"{mc.route_blocked['I2']:.3f} of the worlds "
                       f"(two unreported roads: 1 − 0.865² = {1 - 0.865 ** 2:.3f}).")
    out += ["", i2_line, ""]

    out += [f"## 6. Planned vs Monte Carlo per tau (CSP, N = {N_WORLDS}, seed {SEED})", "",
            "A higher tau makes the PLANNED numbers look better by using roads that are probably "
            "blocked; the sampled worlds show what that costs.", "",
            "| Scenario | tau | Planned deadlines met | Monte Carlo mean | 5th–95th percentile "
            "| P(≥1 planned route blocked) |", "|---|---|---|---|---|---|"]
    for n in NAMES:
        for tau in [0.5, 0.6, 0.9]:
            res = simulate(scen[n], tau=tau)
            mc = monte_carlo(scen[n], res.assignments)
            out.append(f"| {n} | {tau} | {metrics(scen[n], res).deadlines_met_pct:.1f}% | "
                       f"{mc.mean_deadlines_met:.1f}% | {mc.p5_deadlines_met:.1f}–"
                       f"{mc.p95_deadlines_met:.1f}% | {mc.p_any_route_blocked:.3f} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    """@brief Entry point of `python -m src.experiments`: write results/experiments.md and print it."""
    text = run_all()
    path = ROOT / "results" / "experiments.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    print(text)
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
