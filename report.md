# Emergency Response Decision-Support System

<!-- TITLE PAGE (does not count toward the 3 pages)
     - Your name, course (Artificial Intelligence, UAntwerpen), date 12/11/2026
     - The GitHub link, big and visible: **Repository: https://github.com/<you>/<repo>**
     - Before submitting, invite nick.wils@uantwerpen.be and lowie.rombouts@uantwerpen.be to the repo. -->
**Full Name:** Manasseh Ophalfens

**student number:** s0231648


**Repository:** https://github.com/Manaci4466/Emergency-Response-System-Assignment

---

## Approach & implementation

<!-- BUDGET: ~1.2 pages (~600 words + 1 small figure + 1 table).
     OPENING SENTENCE = the whole system in one line. Rewrite this in your own words, e.g.:
     "The system turns noisy road reports and historic traffic data into expected travel times,
      finds routes with A*, and assigns responders online with a constraint solver each time new
      incidents become known."

     Then ONE paragraph per component. Each paragraph follows the same pattern:
       theory (1 sentence) -> where it is in the code (function name) -> the assumption -> its consequence.

     1. Bayesian road model (bayes.py, Notes 6)
        - Network: Weather -> Status -> {Citizen, Police, Sensor}. Inference by enumeration.
        - Why the reports can be multiplied: conditionally independent given Status.
        - Why P(Weather) drops out: weather is observed, so it is a constant that normalisation removes.
        - Unreported roads: weather unknown, so sum it out -> P(Blocked) = 0.135 (decision 2).
        - Give ONE worked number: N1->N2 (small) -> P(Blocked) = 0.527.
     2. HMM traffic model (hmm.py, Notes 8)
        - Hidden = true traffic level, evidence = observed level, forward algorithm + one prediction step.
        - Why forward and not Viterbi (current state vs most likely history).
        - Your multipliers (Low 1.0 / Medium 1.25 / High 1.5) and WHY you chose them.
     3. Cost model + A* (cost_model.py, search.py)
        - Edge time formula, threshold tau.
        - Heuristic = straight-line distance / 30 m/s; one sentence proving admissible + consistent
          (road distance >= straight line on every road; multipliers >= 1).
     4. CSP allocation (allocation.py)
        - Variables, domains, constraints (capability, reachability, no overlap), soft deadlines.
        - MRV + forward checking + branch and bound; what the objective is.
        - Online: incidents are only solved when their timestamp is reached; assignments are committed.

     FIGURE: export the architecture diagram from the guide doc (small, ~1/4 page).
     TABLE: the 10 design decisions condensed to 3 columns: Decision | Choice | Consequence.
     Use the doc's table but cut every cell to <= 10 words. -->

**Data model (`data_loader.py`).** All scenario files are parsed into immutable, typed dataclasses and
checked once at load time: every probability row must sum to one, every road, responder and incident
must refer to an existing node, and all reports for one road must agree on the weather (weather is a
single evidence variable per road in the Bayes net). Roads are stored by their directed `(from, to)`
key, so reports and traffic observations for A→B and B→A are never mixed. The CPTs keep the
orientation of the input files (outer key = true state), so the Bayes and HMM code reads directly
like the formulas in Notes 6 and 8.

## LLM collaboration

<!-- BUDGET: ~0.6 page (~300 words).
     1. Tools: name them exactly. Claude chat on claude.ai (model shown in the chat) for concept
        explanations, design and the build guide; Claude Code in JetBrains Air for implementation
        (if you use it). One sentence on the role of each.
     2. First prompt: "The exact first prompt is in Appendix A (Prompt 1)." Don't paste it here too.
     3. Three or four concrete interactions, each in ONE compact line or table row:
          Asked -> LLM answered -> I accepted / modified / rejected -> why.
        Real candidates from this session:
        - Before the project, used the LLM as a tutor for notation in the course notes
          ("sum out", CPT, the independence symbol, proportional to, argmax). It pointed out
          that the notes say "PDF" where, strictly, a discrete PMF is meant.
        - The LLM ran data checks on all three scenarios and found that the A* heuristic is
          admissible, that two roads sit right on the blocked threshold (P = 0.527), and that
          several deadlines flip depending on assumptions. -> I verified N1->N2 by hand: accepted.
        - While drafting the guide, the LLM first claimed incident I2 (medium) would be on time
          by chaining (19.8 s), then corrected itself to 20.8 s once the unreported-road
          multiplier was included. -> Shows why every LLM number needs a test.
        - Add at least one where YOU changed or rejected a suggestion during the build
          (the grading's 14-17 band explicitly asks for this). Log these in PROMPTS.md as they happen. -->

| Asked | LLM answered | What I did | Why |
|---|---|---|---|
| Phase 1: write the data loader and its test (Appendix A, Prompt 5) | Loader + 5 passing tests; CPT orientation checked with `Police[Blocked][Blocked]` and a Citizen entry | **Modified**: replaced with an asymmetric entry (`Police[Open][Congested]` = 0.06, flipped 0.08), added a weather-consistency check (Prompt 6) | A diagonal entry and a symmetric table give the same value either way round, so that test could never fail |

## Testing & results

<!-- BUDGET: ~0.8 page. Mostly tables, 1-2 sentences of interpretation under each.
     TABLE 1 - Component tests: Test | Expected | Result
        Bayes hand-check N1->N2 (0.014 / 0.459 / 0.527) | HMM umbrella example (P(R3|+,+,-) = 2/5)
        | A* == UCS on all node pairs | no-clairvoyance test | loader counts per scenario
     TABLE 2 - Validator fault injection: Injected fault | Caught? (one row per fault type)
     TABLE 3 - Scenario results: Scenario | Handled | Responders used | Avg response (s) | Deadlines met (%)
        Two rows per scenario: your CSP vs the greedy baseline. All times in seconds.
     SENSITIVITY: one sentence or mini-table for tau = 0.3 / 0.5 / 0.7, and decision 2 switched.
     INTERPRET: explain WHY deadlines are missed (e.g. medium has 20 s windows), not only that they are.
     Say clearly: these are EXPECTED times under the model, not measured ground truth. -->

**Table 1 – Component tests** (`pytest`, all tests in `tests/`)

| Test | Expected | Result |
|---|---|---|
| Loader counts (nodes/roads/responders/incidents) | small 5/14/4/7, medium 12/30/10/12, large 26/60/10/20 | pass |
| CPT orientation (asymmetric entries) | Police P(Congested\|Open) = 0.06; HMM T(Low→Medium) = 0.18, E(Medium\|Low) = 0.14 | pass |
| Blank CSV lines / `;`-split capabilities | blank and comma-only rows skipped | pass |
| Inconsistent weather on one road | `ValueError`; opposite direction allowed | pass |

## Critical reflection

<!-- BUDGET: ~0.4 page (~200 words). Three short paragraphs:
     1. Limitations, each with its consequence: statistics are expectations; one static traffic
        belief for the whole simulation; commit-immediately policy (no re-planning when a more urgent
        incident arrives); reports assumed independent given status (false if citizens copy police);
        multipliers chosen by you, not learned.
     2. LLM: where it helped (speed, theory explanations, finding edge cases) and where it misled
        (a concrete example, see above), and how you caught it (tests, hand calculation).
     3. Next improvements, each WITH how you would measure it:
        rolling-horizon re-planning -> compare deadline % on all 3 scenarios;
        Monte Carlo replay with sampled road states -> % of routes that hit a blocked road. -->

---

## Appendix A: LLM prompts

<!-- One page max. Copy from PROMPTS.md: Prompt 1 verbatim, then the 3-5 most important follow-ups
     (short). Refer to them from the main text as "(Appendix A, Prompt 3)". -->
