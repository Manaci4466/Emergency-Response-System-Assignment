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



## Prompt 11 (Phase 5 CSP allocation) (2026-11-11, Claude Code)

> Phase 5: CSP allocation (src/allocation.py), new branch phase-5-allocation.
> First explain in max 8 lines: variables, domains, hard vs soft constraints, why MRV, what forward checking removes here, and why an empty on-time set does not fail the branch (soft deadlines) but adds a guaranteed penalty used by branch and bound.
> Design (CLAUDE.md decisions 7-10):
> - ResponderState(location, free_at); initially start_location and time_until_available.
> - allocate(scenario, costs, now, incidents, states) -> (assignments, unhandled, new_states, stats). Solves ONE round: the incidents that become known at `now`.
> - Variables = this round's incidents. Domain = responders with the capability that can reach the incident (unary constraints, filter first). Empty domain -> unhandled.
> - Assigning r to i in a branch: start = max(now, r.free_at), arrival = start + A* time from r.location, finish = arrival + time_required; in that branch r becomes (incident location, finish). So one responder can take several incidents in a round, in assignment order (queueing).
> - Objective, lexicographic: (1) sum of priority weights of LATE incidents (Critical 8, High 4, Medium 2, Low 1), (2) total response time Σ(arrival − timestamp).
> - Backtracking. MRV: the unassigned incident with the fewest ON-TIME responders given the branch's current states (ties: higher weight, earlier deadline, id). Values by earliest arrival. Forward checking: after each assignment recompute the on-time sets of the remaining incidents; each empty one adds its weight to a lower bound; prune when (bound, response time so far) >= best. Node limit 100000 -> return best found and flag it.
> - Assignment record: responder_id, incident_id, path, start (= the plan's "time"), arrival, finish, late.
> - greedy(...) baseline with the same signature: incidents in id order, each gets the capable responder with the earliest arrival.
> - Document the limitation: the job order on one responder follows the search order, so the CSP is not exhaustive over all job orders.
> Tests (round t = 0) — verify my numbers against the cost model first:
> - small: I4->R2 (≈56.4, on time), I1->R4 (≈81.1, late by ≈1.1 s), I2->R1 via N1->H1->N2 (≈92.2, late), I3->R3 (≈131.2, on time); missed weight 12
> - greedy on the same round sends R2 to I1 and misses I4; CSP missed weight < greedy's
> - medium: I3->R10 (arrival 0), I2->R10 queued (start 10, arrival ≈20.78, late by 0.78 s); R9 cannot reach the N2 side
> - an incident with a capability nobody has -> unhandled, the rest still assigned
> - brute force on small t=0: every order of the 4 incidents × every responder choice; CSP objective must equal the brute-force optimum; print nodes explored by CSP vs brute forceµ

**Response:** verified the numbers against the cost model first: small and medium all correct, but
the reviewing LLM's greedy prediction was wrong: greedy does not leave I4 to R4 (147.4), it queues R2
behind I1 (arrival 89.6, still late; late weight 16 vs CSP 12). Tests use the verified value.
src/allocation.py (`allocate`, `greedy`, `initial_states`) and tests/test_allocation.py. Brute force
extended to all three t = 0 rounds: CSP equals the optimum (nodes 7 / 7 / 28 vs 96 / 1,440 / 15,360
plans). Own mutation check: an over-strong bound survived every test (the first plan found is already
optimal on real rounds) → added a hand-made assignment trap (first solution 11 s, optimum 4 s) that
catches it. 46 tests pass.
**What I did:** accepted after my own review. Mutation-tested: no MRV (caught), busy responder ignored
(caught), handling time forgotten when queueing (caught), no pruning (not caught, see Prompt 12).
Traced MRV on small t = 0 (I2 → I4 → I1 → I3; forward checking empties I1's on-time set after R2
takes I4). My greedy prediction (R4 at 147.4) was wrong; Claude Code caught it by checking against
the cost model.

---

## Prompt 12 (Phase 5 additions + log) (2026-11-11 19:20, Claude Code)

> Yes: add phase 5 to report.md and PROMPTS.md (log that the reviewing LLM's greedy prediction (R4 at 147.4) was wrong and that you caught it by checking against the cost model), commit, merge into main, push. Two small additions first:
> - assert in the brute-force test that the CSP explores fewer nodes than brute force evaluates plans (removing pruning currently goes unnoticed)
> - add to the allocation.py docstring: because assignment order also fixes the job order of a shared responder, MRV here affects solution quality, not only speed (removing MRV makes medium t=0 worse)

Followed by: "don't merge and push yet. I still want to verify something."

**Response:** checked the suggested assertion first by removing pruning: the CSP still needs only
11 / 7 / 202 nodes against 96 / 1,440 / 15,360 brute-force plans, so "CSP < brute force" cannot detect
missing pruning (one incident order per branch vs all orders). Kept it (true, commented as weak) and
added a `prune` flag plus a test: same answer with and without pruning, strictly fewer nodes on small
(7 vs 11) and large (28 vs 202). Removing pruning is now caught. Verified the MRV claim by mutation
(id order: R10 does I2 first, I3 late at 41.6, late weight 8 vs 4) and added it to the docstring.
49 tests pass. Committed, not merged or pushed.
**What I did:** accepted; my suggested assertion was weak; Claude added prune=False and a real pruning
test, verified by mutation. (Re-checked on my copy: 49 pass; removing pruning again makes the new test
fail on small and large.) This is the second time the reviewing LLM was wrong in phase 5: the
assertion sounded reasonable but tested the wrong thing (brute force counts every incident order, the
CSP only one per branch, so the CSP always has fewer nodes, with or without pruning).

---

## Prompt 13 (Phase 5 merge + Phase 6 simulator, statistics, run.py) (2026-11-11, Claude Code)

> Merge phase-5-allocation into main and push. Then:
>
> Phase 6: simulator + statistics + run.py, new branch.
> - simulator.py: simulate(scenario, tau=0.5, unobserved_traffic=1.0, policy="csp"|"greedy") -> result with all assignments, unhandled ids, per-round stats. Costs computed once (t = 0 beliefs, decision 3). Rounds = distinct incident timestamps in ascending order; each round only sees incidents with timestamp == t and the current responder states (no clairvoyance).
> - statistics.py: from the assignments + scenario: incidents handled; responders used (distinct); average response time = mean(arrival − incident timestamp) over handled incidents; deadlines met % = on-time arrivals / ALL incidents (unhandled count as missed); plus a per-priority breakdown and the lists of late and unhandled incidents. statistics.txt starts with the four required lines in the assignment's example format, then the extras, then a short "Definitions" block: units (seconds), the formulas above, and that times are EXPECTED times under the model, not observed ground truth.
> - run.py: `python run.py Scenarios/<name> [--out DIR] [--tau 0.5] [--policy csp|greedy] [--unobserved-traffic 1.0]`; default out = output/<name>. Writes plan.json with EXACTLY the four keys from the assignment (responder_id, incident_id, path, time), sorted by time then responder id, and statistics.txt. Prints a one-line summary.
> Tests:
> - no clairvoyance: change a future incident (small I7, t=60) and check that all assignments from rounds before t=60 are identical
> - statistics on a hand-made result: 2 handled (one late) + 1 unhandled -> deadlines met = 1/3 = 33.3%, average over the 2 handled only
> - run.py end-to-end on all 3 scenarios (subprocess, tmp out dir): both files exist, plan.json is a list of objects with exactly those 4 keys and the right types
> Finally print a table: per scenario, CSP vs greedy: handled, responders used, avg response time, deadlines met %.

**Response:** merged and pushed phase 5. Read the statistics example format from the assignment PDF
(section 5.2). src/simulator.py, src/statistics.py, run.py and tests/test_simulation.py; 60 tests
pass. The no-clairvoyance test also checks that I7's own assignment changes (otherwise it proves
nothing); mutation check: rounds processed latest-first → caught. Results: CSP and greedy tie on the
required deadline % (small 28.6%, medium 91.7%, large 65.0%), but the CSP keeps a Critical incident on
time where greedy does not (weighted lateness small 24 vs 28, medium 4 vs 8).
**What I did:** accepted after verifying independently: a separate script reading only plan.json and
the raw scenario files recomputed every path cost, start/end node, start time, capability and job
order per responder: 0 errors in all 6 runs, every number matched (including weighted lateness and
Critical on time). Mutation: made the simulator clairvoyant (all incidents known at t = 0) → 7 tests
fail.

---

## Prompt 14 (Phase 6: committed results) (2026-11-11, Claude Code)

> Also: after the phase 6 commit, copy the six final runs (3 scenarios × csp/greedy, tau 0.5) to a committed results/<scenario>-<policy>/ folder and mention it in README.
>
> Do this before phase 7. I know it says I should paste it above phase 7 prompt, but I find it safe to do the fix for phase 6 first before moving on to the validator.

(Suggested by my reviewing LLM: output/ is git-ignored, so the results the report cites were not in
the repo; a grader can now compare their own run with ours.)

**Response:** added phase 6 to report.md (simulation paragraph, Table 3 with weighted lateness and
Critical on time, test rows), committed phase 6, generated the six runs with run.py into
results/<scenario>-<policy>/ and documented them in README.md (install with `python -m pip`, run
options, how to diff a fresh run against results/; a fresh run is byte-identical).
**What I did:** *after review found errors then prompted for test to check and modify*

---

## Prompt 15 (Phase 7 validator) (2026-11-11, Claude Code)

> Phase 7: plan validator (src/validator.py), new branch.
> It must be INDEPENDENT of the planner: it reads plan.json and the raw scenario files, and may use data_loader and cost_model (the model's definition of travel time and P(Blocked)), but NOT search, allocation or simulator. It recomputes travel times by summing edge costs along the given path.
> validate(plan, scenario, tau) -> (errors, warnings).
> Errors:
>  1. schema: a list of objects with exactly responder_id (str), incident_id (str), path (non-empty list of str), time (number >= 0)
>  2. unknown responder or incident id; an incident assigned more than once
>  3. responder lacks the required capability
>  4. path: first node = where the responder is at that moment (start_location for its first job, the previous incident's location afterwards); last node = incident location; every consecutive pair is a directed road; no road with P(Blocked) >= tau; no repeated node
>  5. time < incident timestamp (clairvoyance) or time < the responder's time_until_available
>  6. overlap: per responder, sorted by time, each job must start >= previous arrival + previous time_required (recomputed; tolerance 1e-6)
> Warnings: late arrivals (recomputed arrival > deadline) and incidents missing from the plan (unhandled).
> Also recompute the four required statistics from plan.json alone and check that they match statistics.txt.
> run.py validates after writing and prints "plan valid: E errors, W warnings"; exit code 1 if errors.
> Document the limitation in the module docstring: validity is relative to the BELIEVED road states (tau) and expected travel times, not the true road states.
> Tests:
> - the real plans of all 3 scenarios with both policies: 0 errors
> - fault injection, one test per fault, each must produce exactly the expected error: drop a middle path node; route through small N1->N2 (blocked at tau 0.5); wrong first node; wrong last node; incapable responder; time before the incident timestamp; R4 (small) starting before 10; the same incident twice; two overlapping jobs of one responder; an unknown id; a missing key / wrong type
> - print a table: fault -> caught (this goes in the report)

**Response:** merged and pushed phase 6 first. src/validator.py (`validate` with coded `Issue`s,
`check_statistics`), run.py validates the written plan (exit 1 on errors), tests/test_validator.py:
independence via `ast`, real plans 0 errors with warnings = the planner's late/unhandled, 13 faults
each caught with exactly the expected error (it had to search for a middle-node drop that removes the
road, since dropping H1 from I3's path gives the blocked N2→N4). Own planner mutations: ignoring busy
responders → caught in all scenarios; A* using blocked roads → caught in small and large, not medium
(no route changes). 82 tests pass.
**What I did:** accepted after my own review: 82 pass; run.py gives 0 errors with 5 / 1 / 7 warnings
(exactly the late incidents). Broke the validator itself: forgets the responder moved (18 tests fail),
no overlap check (2 fail), ignores multipliers (9 fail). Found a blind spot: setting Congested to ×1.0
in the cost model, the validator still says "valid" with fewer late warnings (1 / 0 / 2), because it
shares the cost model; 8 hand-checked unit tests fail. Lesson: testing in layers. Also: R1's route to
the Critical I2 (N1→H1→N2) uses two unreported roads at P(Blocked) = 0.135, so P(both open) =
0.865² ≈ 0.75: valid under our beliefs, but blocked in about one world in four.

---

## Prompt 16 (Phase 7 merge + Phase 8 experiments) (2026-11-11 20:00, Claude Code)

> (Commit, merge and push phase 7, then:)
>
> Phase 8: experiments (src/experiments.py, run with `python -m src.experiments`, writes results/experiments.md with markdown tables), new branch.
> 1. tau sensitivity: tau in {0.1, 0.5, 0.6, 0.9} x 3 scenarios, CSP: deadlines met %, priority-weighted late, Critical on time, avg response, unhandled. Explain each change in one line (which roads open/close: e.g. 0.6 opens the 0.527 roads, 0.9 reconnects medium, 0.1 closes every unreported road at 0.135).
> 2. Unreported roads (decision 2): add a parameter to edge_costs, unreported="prior"|"open" ("open" = P(Blocked) 0, status multiplier 1.0). Compare both on all scenarios.
> 3. Unobserved traffic (decision 5): 1.0 vs the stationary expected multiplier (compute it from the HMM stationary distribution, ≈1.23).
> 4. CSP vs greedy on all scenarios incl. priority-weighted late and Critical on time.
> 5. Monte Carlo robustness (Notes 6, sampling): for the CSP plan at tau 0.5, sample N = 5000 worlds with a fixed seed: each road's TRUE status from its Bayes posterior, each observed road's traffic level from its HMM belief (unobserved roads: Low). Replay the plan with its fixed routes: a job starts at max(its planned time, the responder's finish time in this world); a route that contains a road that is blocked in this world fails (incident counts as missed). Report per scenario: P(at least one planned route is blocked), mean deadlines met % with the 5th–95th percentile, and the 3 incidents most often missed. Also report, for small I2, the fraction of worlds in which R1's route is blocked (expected ≈ 0.25).
> Tests: same seed -> identical Monte Carlo result; a scenario copy where every road's posterior is certain Open and every traffic belief is certain Low gives exactly the deterministic plan's numbers; tau monotonicity: a higher tau never makes fewer roads usable.
> Print all tables and add a short "what this shows" paragraph per table to report.md (Testing & results).

**Response:** committed phase 7 (report Table 2 incl. the layered-testing finding, Prompt 15), merged
and pushed. Phase 8: `unreported` option in edge_costs/simulate, `stationary_distribution` in hmm.py
(×1.2264 = 155.75/127), src/experiments.py with the 5 experiments. Design choice in the replay: a
responder whose route is blocked is stranded, so its later jobs (fixed routes starting where it never
arrived) are missed too. Added section 6 on its own initiative: Monte Carlo per tau, because τ 0.6
doubles small's planned deadlines met (28.6% → 57.1%) but is slightly worse in the sampled worlds
(36.2% → 35.1%). Small I2: route blocked in 0.249 of worlds (analytic 0.252). Tests: the requested
three plus the unreported option, the stationary distribution, I2 ≈ 0.25, and a forced-blocked-road
stranding test (its first version was wrong: in the certain world R1 drives N1→N2 directly; fixed;
a no-stranding mutant is caught). 95 tests pass. Report: Tables 4–6 with "what this shows".
**What I did:** *after review created tests and modified code.*

---

## Prompt 17 (Phase 8 fixes before commit) (2026-11-11, Claude Code)

> Before committing phase 8, fix these (each with a test), then finish:
> 1. Experiment 3 bug: the stationary multiplier is computed from small's traffic model and applied to all scenarios. Medium has a different traffic_probabilities.json (its stationary multiplier is 1.25). Compute it per scenario and show it per row.
> 2. Robustness for unseen scenarios: compute MAX_SPEED from the scenario's roads instead of hard-coding 30. At the start of a run, check that every road is at least as long as the straight line between its ends; if not, print a warning and use h = 0 (UCS). Mention the heuristic used in the statistics.txt details. Tests: a scenario copy with a 40 m/s road; a copy with one road shorter than the straight line (must fall back and still equal Floyd–Warshall).
> 3. If the CSP node limit is hit in any round, print a warning and add a line to statistics.txt. Test with a tiny node_limit.
> 4. statistics.txt extras: add "Priority-weighted missed (late or unhandled)" and "Critical on time: x/y" (requested earlier, missing in statistics.py).
> 5. Regenerate and commit results/<scenario>-<policy>/ (3 scenarios × csp/greedy, tau 0.5) and results/experiments.md.
> 6. Put the attached README.md in the repo root. Verify EVERY command, number, file name and claim in it against the code and a fresh run (test count, console output, results table, section 4 point 4 about the fallback), and fix any mismatch in the README. Update the layout section of CLAUDE.md.
> 7. Log in PROMPTS.md (this prompt word for word + what changed). Then commit phase 8, merge into main, push.

**Response (what changed):**
1. Confirmed the bug (Claude Code's own: it applied small's 1.2264 to every scenario and labelled it
   with small's π). Medium's own value is 1.25. `stationary_multiplier(scenario)` per scenario, shown
   per row; medium's row keeps 33.3% / 52 but avg response 41.8 s (was 41.2 s). Tests: helper per
   scenario and the experiment's rows (medium 1.25).
2. `max_speed(scenario)`, `straight_line_violations`, `heuristic_description` in search.py; `astar`
   falls back to UCS when a road is shorter than the straight line; run.py warns; statistics.txt
   "Route search: ..." line. Tests: 40 m/s copy (consistent, = Floyd–Warshall); short-road copy
   (astar == ucs, = Floyd–Warshall). Removing the fallback is caught. While checking, found that on
   the requested copy (large N16→N17 = 1 m) plain A* happens to stay optimal; searched all
   single-road copies: 22 make plain A* suboptimal, e.g. medium N9→N7 = 1 m gives N2→N7 10.8 s
   instead of 7.7 s; added that as a test showing the fallback is really needed.
3. Found a real bug: with node_limit < incidents in a round, no complete plan was found and the
   round's incidents silently vanished (neither assigned nor unhandled). Now falls back to greedy for
   that round and flags it; `--node-limit` option; warning printed and written to statistics.txt.
   Test: node_limit = 1 (nothing vanishes, warning in stdout and file, no warning by default).
4. `weighted_missed` and `critical_on_time` in Statistics and statistics.txt (+ definition);
   experiments reuse them. Hand-made test: 4 + 1 = 5, Critical 1/1.
5. Regenerated results/ (plans unchanged, statistics.txt +4 lines) and results/experiments.md.
6. CLAUDE.md run/layout updated. The README attachment did not arrive: waiting for it.
102 tests pass.
**What I did:** *(fill in)*

---

<!-- Template for new entries:

## Prompt N: <topic> (<date time>, <tool>)

> <exact prompt>

**Response:** <1-2 lines>
**What I did:** accepted / modified / rejected, because ...
-->
