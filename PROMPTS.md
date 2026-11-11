# LLM prompt log

Log every prompt that shaped the design, code or report. One entry each: what you asked,
what came back, and what you did with it (accepted / modified / rejected + why).
The report needs Prompt 1 **exactly as sent**, typos included, so don't edit it.

**Tools used**
- Claude (claude.ai chat): concept explanations, design, build guide, report help.
- Claude Code in JetBrains Air (Claude subscription): implementation in the repo. *(fill in if you use it)*

---

## Before the project: concept questions (2026-11-11, 14:48-15:01, Claude chat)

Questions about notation in the course notes (Notes 6 and 8), asked before starting the assignment:
"sum out" (marginalisation), what "PDF" stands for, what a CPT is, the upside-down T (independence),
whether independence still holds if X and Y depend on a third variable, random vs stochastic
variables, the proportional sign, and argmax.

**Response:** explanations with worked examples. It noted that the notes use "PDF" where, strictly,
a PMF (discrete) is meant.
**What I did:** used them to understand Notes 6 and 8 before designing the system.

---

## Prompt 1: first substantive prompt (2026-11-11 15:30, Claude chat)

> I need to complete this assignment but I do not know exactly how. For this prompt I am not going to explain everything thorougly, but I just want you to work with me and give me a guide that maximalizes both the project generated as well as my understanding of it. I still need to add all the scenarios so wait a bit before generating or doing anything. When I send a prompt with only 'ready' in it. You may begin!

Attached: Assignment1.pdf, AI_Practica1-search_CSP.pdf, AI_Practica2-BN_HMM.pdf, Notes_6.pdf,
Notes_8.pdf, and the files of two scenarios.

**Response:** confirmed it would wait for "ready".

---

## Prompt 2: "ready" (2026-11-11 15:30, Claude chat)

> ready

Attached: the third scenario (large map) and the scenario README.

**Response:** read the assignment and practica, ran a data-check script on all three scenarios, and
wrote a build guide (doc): time plan, architecture, 10 design decisions, 8 build phases, data
findings, pitfalls, report plan and defence questions.
Key findings: the A* heuristic is admissible and consistent in all scenarios; two roads sit right at
P(Blocked) = 0.527; several deadlines depend on assumptions (e.g. large I4, I5).
While drafting, it first claimed medium I2 is on time by chaining (19.8 s), then corrected this to
20.8 s once the unreported-road multiplier was included.
**What I did:** *(fill in: e.g. checked the N1->N2 posterior by hand)*

---

## Prompt 3: design decisions (2026-11-11 16:09, comment in the guide doc)

> The decisions look good for now

**What I did:** accepted all 10 recommended decisions for version 1. *(Note any you change later, and why.)*

---

## Prompt 4: report help + JetBrains Air setup (2026-11-11 16:11, Claude chat)

> *Yes, it is tomorrow, the date.  I also forgot to add the ReadME from the scenarios. I don't know if that changes anything. I need your help writing the report. I already have a file ready which I added just now. I want you to give me hints on what to write and also refine the few sentences I write into something more refined and concrete. I also want to use Jetbrains Air and log in to Claude using this account. I don't know how to. Help me do this before starting the first fase.*

**Response:** report skeleton with section hints, this log, a CLAUDE.md for the repo, Air setup steps.
**What I did:** Looked for a way to link Claude Code to PyCharm

---


## Prompt 5 (Phase 1 data loader): data loader (2026-11-11 16:52, Claude Code)

> Read CLAUDE.md. Start phase 1: explain the plan, then write the data loader and its test.

**Response:** Phase 1 is finished: the data loader works and all 5 tests pass. The
  node, road, responder and incident counts match CLAUDE.md for small,
  medium and large.
**What I did:** loaded a one-shot prompt with the help of Claude Chat

---


## Prompt 6 (Phase 1 review): fixes before phase 2 (2026-11-11 17:50, Claude Code)

> Fixes before phase 2:
> 1. The CPT orientation test cannot fail: Police[Blocked][Blocked] is diagonal and the citizen table is symmetric. Test an asymmetric entry instead (Police["Open"]["Congested"] == 0.06; flipped it would be 0.08) and explain in a comment why the old check proved nothing.
> 2. Plain `pytest` fails with "No module named src". Add a pyproject.toml with [tool.pytest.ini_options] pythonpath = ["."] and show that plain `pytest` now passes.
> 3. Check that all reports for one directed road share the same weather; raise a ValueError otherwise. Add a test.
> 4. Fix the Node.type comment (the large map also has police and fire_station).
> 5. Add requirements.txt (pytest), keep the Scenarios/ folder name and update CLAUDE.md to match.

**Response:** replaced the orientation test with asymmetric entries (also for the HMM transition and
emission tables), added pyproject.toml, requirements.txt and the weather check + test, fixed the
comment, updated CLAUDE.md. Plain `pytest`: 6 passed.
**What I did:** modified the LLM's phase 1 code after reviewing it: its orientation test could
never fail (diagonal entry + symmetric table), so it proved nothing. Lesson: check that a test can fail.
Phase 1 fixes: gave 5 fixes from my review. All applied; it also added HMM orientation checks on its
own initiative (transition Low→Medium 0.18, flipped 0.15; emission Low→Medium 0.14, flipped 0.10)
→ **accepted** after checking the values in the JSON.

---


## Prompt 7 (Phase 2 Bayes): Bayesian road status (2026-11-11 18:00, Claude Code)

> Phase 2: Bayesian road status (src/bayes.py).
> First explain in max 5 lines: the network Weather -> Status -> {Citizen, Police, Sensor}, which variables are query / evidence / hidden, and why P(Weather) is not needed when weather is observed.
> Implement:
> - status_posterior(model, weather, reports): P(S | w, reports) ∝ P(S|w) · Π P(report|S), normalised. If weather is None, sum out weather with the prior.
> - road_status_beliefs(scenario) -> dict[Edge, dict[str, float]] for EVERY road (unreported roads: no weather, no reports).
> Tests:
> - small N1->N2 (Rain; Blocked, Blocked, Congested) -> O 0.014, C 0.459, B 0.527 (tolerance 1e-3); put the unnormalised products in a comment
> - no reports + Clear -> exactly P(S | Clear)
> - no reports + no weather -> O 0.685, C 0.18, B 0.135
> - a road with one source missing still works; every result sums to 1

**Response:** src/bayes.py (`status_prior`, `status_posterior`, `road_status_beliefs`) and
tests/test_bayes.py; 13 tests pass. Unnormalised products for N1→N2: 0.00012 / 0.00384 / 0.00441.
Roads with P(Blocked) ≥ 0.5: small H1↔N3 (0.998), N1→N2 and N2→N4 (0.527); medium N1↔N2 (0.83);
large N16↔N5 (0.527). Also tested that every unreported road in all three scenarios equals the prior.
**What I did:** Accepted. Recomputed the N1→N2 products by hand. Mutation-tested with a second LLM: flipping the CPT index gives 0.614 and removing the dict() copy corrupts the CPT. The tests catch both.

---


## Prompt 8 (Phase 2 fixes + Phase 3 HMM): traffic model (2026-11-11, Claude Code)

> Phase 3: HMM traffic model (src/hmm.py). First apply the 3 small fixes: sharper docstring in bayes.py (S is each report's ONLY parent, no edges between reports), compute the expected prior per scenario in test_every_road_gets_a_distribution, and validate reported_status and weather values in the loader.
>
> Then explain in max 5 lines: hidden state vs evidence, the two steps of the forward algorithm (time elapse, then observe + normalise), why we normalise at every step, and why forward and not Viterbi.
> Implement:
> - forward(model, observations) -> (belief over X_n after the last observation, likelihood P(e_1..e_n)). Convention: model.initial = P(X1), the state that emits observation 1 (no transition before it). Normalise after every step; the likelihood is the product of the normalising constants. Must also work with fractions.Fraction.
> - predict(model, belief, steps=1): time-elapse only.
> - traffic_beliefs(scenario) -> dict[Edge, dict[str, float]]: forward + one predict step, ONLY for roads with observations.
> - expected_multiplier(belief, multipliers={"Low": 1.0, "Medium": 1.25, "High": 1.5}).
> Tests:
> - practicum 2 Q5 umbrella HMM with Fraction: P(R1)=1/2, T(R->R)=T(D->D)=3/4, E(+|R)=E(-|D)=3/4, observations +,+,- -> P(Rain3 | obs) = 2/5 and likelihood = 15/128, both exact
> - small N4->H1: predicted P(High) ≈ 0.548; large N16->N17: predicted P(High) ≈ 0.704
> - predicting 50 steps converges to a fixed distribution (print it: the stationary distribution)
> - every belief sums to 1

(The 3 fixes came from my review of phase 2, done with Claude chat.)

**Response:** Phase 2 fixes applied, plus a load-time check on traffic levels (its own addition);
3 typo tests. src/hmm.py (`predict`, `observe`, `forward`, `traffic_beliefs`, `expected_multiplier`)
and tests/test_hmm.py: umbrella 2/5 and 15/128 exact; N4→H1 0.548, N16→N17 0.704; stationary
distribution (45, 49, 33)/127. Its first stationary test (tolerance 10⁻⁹ after 50 steps) failed;
it explained the geometric convergence (≈0.7⁵⁰ ≈ 2·10⁻⁸), loosened to 10⁻⁶ and added an exact
Fraction check π·T = π. 24 tests pass.
**What I did:** Modified. Checked 2/5, 15/128 and π·T = π for (45, 49, 33)/127 by hand. Mutation-tested
four typical HMM bugs: transition table flipped (6 tests fail), no normalisation (6 fail), no
prediction step to t = 0 (2 fail), transition before observation 1 (0 fail). Mutation testing showed
the P(X₁) convention was untested (both existing tests are blind to it); added a one-observation
test. Corrected the LLM's eigenvalue claim (0.742, not 0.7). See Prompt 9.

---


## Prompt 9 (Phase 3 fixes + Phase 4 cost model + A*) (2026-11-11, Claude Code)

> Phase 3 fixes first:
> 1. Add a test pinning the convention initial = P(X1): small model, ONE observation "Low" must give B ∝ emission·initial = (0.941, 0.0554, 0.0037). Explain in a comment why the existing tests cannot catch a transition applied before observation 1 (umbrella prior is stationary for its symmetric T; 10 observations wash out the start). Prove it: temporarily apply the transition before observation 1, show only the new test fails, then revert.
> 2. Fix the comment: the 2nd eigenvalue of T is 0.742, so 0.742^50 ≈ 3e-7.
> 3. In report.md replace the last HMM sentence with: "Predicting further ahead converges to the stationary distribution (45, 49, 33)/127, so beliefs lose their road-specific information. Because observation_nr has no time unit, the t = 0 belief is used for the whole simulation, which is likely overconfident for late incidents."
> Commit, merge phase-2-bayes and phase-3-hmm into main, push.
>
> Phase 4: cost model + A* (src/cost_model.py, src/search.py) on a new branch.
> Explain in max 5 lines: f = g + h, why h = straight-line distance / 30 m/s is admissible AND consistent here, and why consistency matters for graph search with a closed set.
> cost_model.py: edge_costs(scenario, tau=0.5, unobserved_traffic=1.0) -> per directed road: base = distance/speed, p_blocked, status_mult = (P(O)·1 + P(C)·1.5) / (P(O)+P(C)), traffic_mult (expected_multiplier, or unobserved_traffic without observations), cost = base·status_mult·traffic_mult, or None if p_blocked >= tau. Keep the parts so they can be printed.
> search.py: astar(scenario, costs, start, goal) -> (path, cost, expanded) or None. Graph search with closed set, ties broken alphabetically; start == goal -> ([start], 0). ucs = the same with h = 0.
> Tests:
> - every road in all 3 scenarios: distance >= straight-line distance (print the smallest ratio)
> - consistency: for every goal and every usable road (u,v): cost(u,v) >= h(u) - h(v)
> - A* cost == UCS cost for EVERY node pair in all 3 scenarios; A* never expands more nodes than UCS (print totals)
> - small: H1<->N3 unusable (P(Blocked) 0.998); N1->N2 unusable at tau 0.5, usable at tau 0.6
> - a tiny hand-made graph where the shortest path in metres is NOT the fastest (main 30 m/s vs secondary 15 m/s): A* picks the fastest
> - unreachable goal -> None

(Written after my own review of phase 3 in Claude chat, including the mutation testing above.)

**Response (phase 3 fixes):** verified the numbers first (eigenvalues 1, 0.742, 0.508; one-observation
belief 0.941 / 0.0554 / 0.0037, off-by-one 0.925 / 0.069 / 0.005). Added the convention test with the
explanation comment; injected the bug → only the new test failed (1 failed, 24 passed); reverted →
25 passed. Fixed the eigenvalue comment and the report sentence.
**Response (phase 4):** src/cost_model.py (`EdgeCost` with base / p_blocked / status_mult /
traffic_mult / cost, `edge_costs`) and src/search.py (`astar`, `ucs`, closed set, alphabetical ties);
38 tests pass. Smallest distance/straight-line ratio: small 1.017, medium 1.132, large 1.116.
Expanded A* vs UCS over all pairs: small 46/50, medium 144/180, large 7780/8450. Found that medium
splits in two halves at tau 0.5 (N1↔N2 blocked, P = 0.83): only 72 of 144 pairs reachable.
Mutation check: heuristic with 15 m/s (inadmissible) → consistency and straight-line tests fail in
all scenarios, A* = UCS fails only on large.
**What I did:** *accepted; modified the tests after mutation testing showed the shared search function was a blind spot.*

---


## Prompt 10 (Phase 4 fix: differential testing) (2026-11-11, Claude Code)

> Phase 4 fix first: A* and UCS share _best_first, so a bug inside it makes both wrong the same way and "A* == UCS" still passes (a reviewer injected a cost bug in _best_first; only test_fastest_not_shortest caught it). In the every-pair test: (a) recompute each returned path's cost by summing costs[(a,b)].cost over its roads (all usable) and compare with the returned cost; (b) compare with an independent all-pairs reference written in the test itself (Floyd–Warshall). Clear __pycache__, run pytest (check its exit code), commit, merge phase-4-search into main, push.

(From my review of phase 4 in Claude chat. My mutation test: "goal tested when generated" was caught
by A* = UCS on medium and large; "some roads costed wrong during search" passed A* = UCS in all 3
scenarios and was only caught by the tiny hand-made graph.)

**Response:** added a Floyd–Warshall reference in the test file and, for every pair, checks that the
path's roads exist, are usable and sum to the returned cost, and that A* and UCS both equal the
reference; unreachable pairs must be ∞ in the reference too. Re-ran the cost mutant (0.9 × cost on some
roads inside `_best_first`): now caught in all 3 scenarios. Reverted, cleared `__pycache__`, 38 pass
(pytest exit code 0).
Also logged the phase 4 commit incident in report.md (LLM process error: `pytest | tail` hid the exit
code, a stale .pyc from its mutation check caused the failures).
**What I did:** modified: found the blind spot by mutation testing; asked for differential testing.

---



<!-- Template for new entries:

## Prompt N: <topic> (<date time>, <tool>)

> <exact prompt>

**Response:** <1-2 lines>
**What I did:** accepted / modified / rejected, because ...
-->
