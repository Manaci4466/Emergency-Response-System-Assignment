# Emergency-Response-System-Assignment
Artificial Intelligence Practicum

## Install and run

Python 3.10+ (standard library only; pytest for the tests).

```
python -m pip install -r requirements.txt
python run.py Scenarios/small                      # writes output/small/plan.json + statistics.txt
python run.py Scenarios/large --policy greedy --tau 0.5 --out output/large-greedy
python -m pytest
```

Options: `--out DIR` (default `output/<scenario>`), `--tau` (road unusable if P(Blocked) ≥ tau,
default 0.5), `--policy csp|greedy` (default `csp`), `--unobserved-traffic` (traffic multiplier for
roads without observations, default 1.0). All times are in seconds.

## Results

The runs cited in the report are committed in `results/<scenario>-<policy>/` (3 scenarios × CSP and
greedy baseline, tau 0.5, default options), each with `plan.json` and `statistics.txt`.
`output/` is git-ignored, so to compare your own run with ours:

```
python run.py Scenarios/small --policy csp --out output/small-csp
diff output/small-csp/plan.json results/small-csp/plan.json
```
