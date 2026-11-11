# Emergency Response Decision-Support System (AI practicum, UAntwerpen)

Student project. The student must be able to explain and defend every line in an oral exam,
so understanding matters more than features.

## How to work with me
- Build ONE phase at a time (order below). Don't write the whole system in one go.
- Before writing code for a phase: explain the concept in at most 5 lines and link it to the course
  (Notes 6 = Bayes nets / sampling, Notes 8 = HMMs, practicum 1 = search + CSP).
- Keep code simple, typed, commented where the theory shows up. Standard library only (+ pytest).
- Every phase ends with a test that passes. Show me the command to run it.
- If you are unsure about a number or a property (admissible, optimal, ...), write a test instead of
  claiming it.
- At the end of each phase, suggest 2-3 sentences for the report.

## Run
Setup: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`; tests: `pytest`.
`python run.py Scenarios/<name> [--out output/<name>] [--tau 0.5] [--policy csp|greedy] [--unobserved-traffic 1.0] [--node-limit 100000]`
-> writes plan.json + statistics.txt, then validates the plan (exit code 1 on errors).
`python -m src.experiments` -> results/experiments.md (sensitivity + Monte Carlo).
Units: metres, metres/second, seconds (fractional allowed). Deadlines are ABSOLUTE times.

## Layout
run.py, src/{data_loader, bayes, hmm, cost_model, search, allocation, simulator, validator, statistics,
experiments}.py, tests/test_<module>.py (+ test_mutation_sweep.py, opt-in: RUN_MUTATION_SWEEP=1),
tools/mutation_sweep.py (35 mutants, all must be caught), Scenarios/{small = scenario 1, medium = 2, large = 3}/ (capital S),
results/<scenario>-<policy>/{plan.json, statistics.txt} + results/experiments.md (committed runs, tau 0.5),
output/ (git-ignored default output), report.md, PROMPTS.md (prompt log), README.md.

## Build order
1. data_loader: dataclasses; CSVs contain blank lines (skip them); capabilities split on ';'.
2. bayes: P(S | weather, reports) ∝ P(S|weather) · Π P(report|S). JSON CPTs: OUTER key = true status.
   Unreported road: sum out weather with the prior.
3. hmm: forward algorithm; `initial` = P(X1), the state that emits observation 1; normalise each step;
   one extra time-elapse step after the last observation = belief at t = 0.
4. cost_model + search: time = distance/speed × status mult × traffic mult; road unusable if
   P(Blocked) ≥ tau. A* with h = euclidean / 30 m/s (graph search, closed set, alphabetical ties).
5. allocation: CSP per event. Variables = incidents appearing at time t; domain = capable responders
   (with timelines, so busy responders can be queued) + LATE value. MRV, forward checking on deadlines,
   branch and bound on (priority-weighted misses, total response time). Weights: Critical 8, High 4,
   Medium 2, Low 1.
6. simulator + run.py: incidents grouped by timestamp, solved only when known (no clairvoyance).
7. validator: recompute everything from the raw files (never reuse planner internals); fault-injection tests.
8. statistics: handled, responders used, avg response time (arrival − incident timestamp, s),
   deadline % over ALL incidents (unhandled = missed), per-priority breakdown.

## Design decisions (already chosen)
1. Blocked if P(B) ≥ tau = 0.5; else status mult = (P(O)·1 + P(C)·1.5) / (P(O)+P(C)).
2. Unreported roads: prior marginal over weather (O .685, C .18, B .135 -> mult ≈ 1.104).
3. Traffic "now" = filtered belief + one prediction step.
4. Traffic mult: Low 1.0, Medium 1.25, High 1.5; expected value over the belief; multiplied with status mult.
5. Roads without traffic observations: mult 1.0.
6. Roads are directed; never share data between A->B and B->A.
7. Commit assignments immediately (rolling horizon = future improvement).
8. Busy responders can be queued; after a job a responder stays at the incident location.
9. Deadlines are soft (late help > no help).
10. Response time = arrival − incident timestamp.

## Known test values
- Bayes, small N1->N2 (Rain; Blocked, Blocked, Congested): O 0.014, C 0.459, B 0.527.
- HMM, practicum 2 Q5 umbrella (with Fraction): P(Y) = 15/128, P(Rain3 | +,+,-) = 2/5.
- A* must equal UCS (h = 0) on every node pair.
- Loader counts (nodes/roads/responders/incidents): small 5/14/4/7, medium 12/30/10/12, large 26/60/10/20.
