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
**What I did:** **modified** the LLM's phase 1 code after reviewing it: its orientation test could
never fail (diagonal entry + symmetric table), so it proved nothing. Lesson: check that a test can fail.

---



<!-- Template for new entries:

## Prompt N: <topic> (<date time>, <tool>)

> <exact prompt>

**Response:** <1-2 lines>
**What I did:** accepted / modified / rejected, because ...
-->
