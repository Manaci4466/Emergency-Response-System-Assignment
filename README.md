# Emergency Response Decision-Support System

Artificial Intelligence Practicum, UAntwerpen. Report: [`report.md`](report.md) · LLM prompt log:
[`PROMPTS.md`](PROMPTS.md)

The system plans which emergency responder goes to which incident, and along which route, in a city
whose road conditions are uncertain. It combines the course techniques in one pipeline:

1. **Bayesian network** (Notes 6): noisy road reports (citizen, police, sensor) + weather → P(road status).
2. **Hidden Markov model** (Notes 8): historical traffic observations → belief over the current traffic level.
3. **A\* search** (practicum 1): fastest route on the directed road graph with expected travel times.
4. **Constraint satisfaction** (practicum 1): responders are assigned to incidents with backtracking,
   MRV, forward checking and branch and bound, round by round as incidents become known.

`run.py` writes the plan (`plan.json`) and its statistics (`statistics.txt`), and an independent
validator re-checks the written plan.

---

## 1. Scenarios

The three provided datasets live in `Scenarios/`. The folder names map to the assignment's numbering:

| Assignment | Folder | Nodes | Roads (directed) | Responders | Incidents |
|---|---|---|---|---|---|
| Scenario 1 | `Scenarios/small` | 5 | 14 | 4 | 7 |
| Scenario 2 | `Scenarios/medium` | 12 | 30 | 10 | 12 |
| Scenario 3 | `Scenarios/large` | 26 | 60 | 10 | 20 |

Each folder contains `city_graph.json`, `incidents.csv`, `responders.csv`, `road_reports.csv`,
`observed_traffic.csv`, `road_probabilities.json` and `traffic_probabilities.json`
(see [`Scenarios/README.md`](Scenarios/README.md)). Any folder with the same files can be used.

## 2. Installation

Tested with Python 3.12. Only the standard library is used; `pytest` is needed for the tests.

```
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## 3. Running

```
python run.py Scenarios/small
```

Console output:

```
small (policy csp, tau 0.5, unobserved traffic 1.0): 7/7 handled, 4 responders, avg response 112.8 s, deadlines met 28.6% -> output/small
plan valid: 0 errors, 5 warnings
```

The 5 warnings are the 5 incidents that are reached after their deadline (late help is still given).

| Option | Default | Meaning |
|---|---|---|
| `scenario` (positional) | – | scenario folder, e.g. `Scenarios/large` |
| `--out DIR` | `output/<scenario name>` | where `plan.json` and `statistics.txt` are written |
| `--tau T` | `0.5` | a road is unusable if P(Blocked) ≥ T; must be in (0, 1] |
| `--policy csp\|greedy` | `csp` | the CSP solver, or the greedy baseline (incidents in id order, earliest arrival) |
| `--unobserved-traffic M` | `1.0` | traffic multiplier for roads without traffic observations; must be ≥ 1 |
| `--node-limit N` | `100000` | CSP search nodes per round; if hit, the best plan found so far is used and a warning is printed |

Exit code: `0` if the written plan is valid, `1` if the validator finds errors, `2` for invalid options.

More examples:

```
python run.py Scenarios/large --policy greedy --out output/large-greedy
python run.py Scenarios/medium --tau 0.9
python -m src.experiments          # all experiments -> results/experiments.md
python -m pytest                   # all tests
```

## 4. Output files

**`plan.json`**: a list of assignments with exactly the four keys from the assignment, sorted by time
and then responder id. `time` is the simulation time (s) at which the responder **starts** the job.

```json
[
  {"responder_id": "R1", "incident_id": "I2", "path": ["N1", "H1", "N2"], "time": 0.0},
  {"responder_id": "R2", "incident_id": "I4", "path": ["H1", "N2"], "time": 0.0}
]
```

**`statistics.txt`** starts with the four required lines (small, CSP):

```
System statistics:

 * 7 incidents handled
 * 4 responders used
 * Average response time: 112.8 s
 * Deadlines met: 28.6%
```

followed by details (on time, Critical on time, priority-weighted missed, per-priority breakdown,
late and unhandled incidents, the route-search heuristic used, node-limit warnings) and a
"Definitions" block:

- **Units**: seconds; all times are absolute simulation times from t = 0.
- **Incidents handled**: incidents with an assigned responder.
- **Responders used**: distinct responders with at least one assignment.
- **Average response time**: mean of (arrival − incident timestamp) over **handled** incidents only.
- **Deadlines met**: on-time arrivals (arrival ≤ deadline) / **all** incidents; unhandled = missed.
- **Priority-weighted missed**: Σ priority weights (Critical 8, High 4, Medium 2, Low 1) of late or
  unhandled incidents.
- All times are **expected** times under the probabilistic model, not observed ground truth.

## 5. How it works

```
road_reports + weather ──► bayes.py ──► P(Open / Congested / Blocked) per directed road ─┐
observed_traffic ────────► hmm.py ───► P(Low / Medium / High) per directed road ────────┤
                                                                                        ▼
                         cost_model.py: expected travel time per road (None if P(Blocked) ≥ tau)
                                                                                        ▼
incidents (by timestamp) ► simulator.py ─► allocation.py (CSP per round) ─► search.py (A*)
                                                                                        ▼
                                  run.py ─► plan.json + statistics.txt ─► validator.py
```

| Module | What it does | Course link |
|---|---|---|
| `src/data_loader.py` | reads one scenario into typed dataclasses; checks probability rows, ids, weather consistency, observation numbering and value names at load time | – |
| `src/bayes.py` | Weather → Status → {Citizen, Police, Sensor}; P(S \| w, reports) ∝ P(S\|w) · Π P(report\|S) by enumeration; weather summed out for unreported roads | Notes 6 |
| `src/hmm.py` | forward algorithm (time elapse, observe, normalise), one prediction step to t = 0, stationary distribution | Notes 8 |
| `src/cost_model.py` | time = distance/speed × status multiplier × traffic multiplier; unusable if P(Blocked) ≥ tau | – |
| `src/search.py` | A\* graph search with a closed set, h = straight-line distance / fastest road speed; falls back to uniform-cost search if a road is shorter than the straight line | practicum 1 |
| `src/allocation.py` | per-round CSP: backtracking, MRV, forward checking on deadlines, branch and bound on (priority-weighted late, total response time); greedy baseline | practicum 1 |
| `src/simulator.py` | rounds by incident timestamp, no clairvoyance, assignments committed immediately | – |
| `src/statistics.py` | the statistics and the `statistics.txt` text | – |
| `src/validator.py` | independent plan check: reads only `plan.json` + raw files (never the planner's search or allocation code) | – |
| `src/experiments.py` | sensitivity analyses and Monte Carlo robustness | Notes 6 (sampling) |
| `tools/mutation_sweep.py` | injects 35 known bugs, one at a time, and checks the tests catch each one | – |

### Design decisions and assumptions

1. A road is unusable if P(Blocked) ≥ tau (0.5); otherwise the status multiplier is
   (P(O)·1 + P(C)·1.5) / (P(O) + P(C)), the expected slowdown given that the road is passable.
2. Roads without reports get the prior marginal over weather: Open 0.685, Congested 0.18,
   Blocked 0.135 (multiplier ≈ 1.104).
3. Traffic "now" = filtered HMM belief + one prediction step; road states are fixed for the whole run
   (all evidence is available at t = 0).
4. Traffic multipliers Low 1.0, Medium 1.25, High 1.5; the expected value over the belief is
   multiplied with the status multiplier.
5. Roads without traffic observations get traffic multiplier 1.0.
6. Roads are directed: A→B and B→A never share reports or observations.
7. Assignments are committed immediately (no re-planning when a later incident arrives).
8. Busy responders can be queued; after a job a responder waits at the incident location.
9. Deadlines are soft: late help is better than no help.
10. Response time = arrival − incident timestamp.

Additional safeguards: the A\* heuristic uses the fastest road speed of the scenario and is replaced by
h = 0 (uniform-cost search, with a warning) if any road is shorter than the straight line between its
ends, because then h could overestimate. If the CSP hits its node limit before finding any complete
plan for a round, that round falls back to the greedy plan, so no incident is silently dropped.

## 6. Testing

```
python -m pytest                                                        # 110 passed, 1 skipped
RUN_MUTATION_SWEEP=1 python -m pytest tests/test_mutation_sweep.py -s   # the slow overarching check
python tools/mutation_sweep.py                                          # the same, as a script
```

- **Hand-calculated values**: e.g. the Bayes posterior for small N1→N2 (0.014 / 0.459 / 0.527),
  the practicum umbrella HMM exactly with fractions (2/5 and likelihood 15/128), the stationary
  traffic distribution (45, 49, 33)/127, the small t = 0 allocation (late weight 12 vs greedy 16).
- **Differential testing**: A\* = uniform-cost search = an independent Floyd–Warshall in the test, on
  every node pair; the CSP = exhaustive brute force on all three t = 0 rounds.
- **Properties**: the heuristic is consistent for every goal, tau sets are nested, no clairvoyance
  (changing a future incident never changes an earlier assignment).
- **Validator fault injection**: 13 injected plan faults, each caught with exactly the expected error.
- **Mutation sweep** (`tools/mutation_sweep.py`): 35 typical bugs across all modules (flipped CPTs,
  off-by-one in the HMM, an inadmissible heuristic, no MRV, ignoring busy responders, clairvoyance,
  wrong statistics denominators, a validator that forgets checks, ...). Each is applied to a temporary
  copy of the project and the whole suite is run: **35 of 35 caught**.

## 7. Results

Committed runs (tau 0.5, default options) are in `results/<scenario>-<policy>/`; compare a fresh run
with `diff output/small/plan.json results/small-csp/plan.json`. All times are expected seconds.

| Scenario | Policy | Handled | Responders used | Avg response (s) | Deadlines met | Priority-weighted missed | Critical on time |
|---|---|---|---|---|---|---|---|
| 1 (small) | CSP | 7/7 | 4 | 112.8 | 28.6% | 24 | 1/2 |
| 1 (small) | greedy | 7/7 | 4 | 112.2 | 28.6% | 28 | 0/2 |
| 2 (medium) | CSP | 12/12 | 8 | 34.6 | 91.7% | 4 | 6/6 |
| 2 (medium) | greedy | 12/12 | 8 | 37.2 | 91.7% | 8 | 5/6 |
| 3 (large) | CSP | 20/20 | 8 | 1319.2 | 65.0% | 36 | 5/8 |
| 3 (large) | greedy | 20/20 | 8 | 1319.2 | 65.0% | 36 | 5/8 |

The unweighted deadline percentage is the same for both policies, but the CSP keeps a Critical
incident on time where greedy does not (small and medium). Small misses several deadlines by only
1–5 s.

`results/experiments.md` (regenerate with `python -m src.experiments`) contains:

- **tau** 0.1 / 0.5 / 0.6 / 0.9: 0.1 closes every unreported road and disconnects the maps;
  0.6 opens the roads at P(Blocked) = 0.527; 0.9 reconnects medium, whose two halves are linked only
  by N1↔N2 (P(Blocked) = 0.83).
- **Assumptions**: treating unreported roads as open lifts small from 28.6% to 85.7%; the stationary
  traffic multiplier for unobserved roads (×1.25 in medium) drops medium from 91.7% to 33.3%.
- **Monte Carlo** (5000 sampled worlds, seed 2026): true road states and traffic levels are sampled
  from the beliefs and the fixed plan is replayed. At least one planned route is blocked in 68% /
  64% / 97% of worlds (small / medium / large). R1's route to the Critical I2 in small is blocked in
  24.9% of worlds (analytically 1 − 0.865² = 25.2%). Raising tau to 0.6 doubles small's planned
  deadlines met (28.6% → 57.1%) but not the sampled one (36.2% → 35.1%).

## 8. Limitations

- All statistics are **expected** values under the model; a "valid" plan is valid under the believed
  road states, not the true ones (see the Monte Carlo results).
- Road states and traffic are estimated once at t = 0 and used for the whole simulation.
- Assignments are never re-planned when a more urgent incident arrives later.
- The CSP is optimal within one round, but the job order of a shared responder follows the search
  order, so it is not exhaustive over all job orders (brute force agrees on all three t = 0 rounds).
- Reports are assumed conditionally independent given the road status; the traffic and status
  multipliers are chosen, not learned.

## 9. Repository layout

```
run.py                     entry point
src/                       data_loader, bayes, hmm, cost_model, search, allocation,
                           simulator, statistics, validator, experiments
tests/                     one test file per module + test_mutation_sweep.py (opt-in)
tools/mutation_sweep.py    overarching mutation test
Scenarios/                 small (1), medium (2), large (3)
results/                   committed runs <scenario>-<policy>/ and experiments.md
output/                    default output folder (git-ignored)
report.md, PROMPTS.md      report and LLM prompt log
requirements.txt, pyproject.toml
```
