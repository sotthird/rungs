# rungs

> Deciding how hard to think about a problem, before thinking about it.

Optimization systems usually commit to one solving method and apply it to
every instance. But instances differ enormously in how much effort they
deserve, and you can't tell which is which without solving them. This
project predicts per-instance difficulty from features that cost nothing to
compute, and allocates solver effort accordingly — on capacitated facility
location, a sequential "run the cheap method, look at what happened, then
decide whether to escalate" policy cuts mean solve time by **26%** (0.117s →
0.087s) for a **6.8-point** quality-gap cost, capturing **13–22%** of the
improvement available to a perfect oracle across the range where compute
actually costs something.

This is not a new idea — **algorithm selection** was formalized by Rice
(1976); the best-known success is **SATzilla**, which won SAT competitions
predicting per-instance solver runtime from cheap features. The theoretical
frame for "is more thinking worth it" is Russell & Wefald's rational
metareasoning. What's here: sequential escalation solved by backward
induction rather than learned, an oracle-bounded metric (gain fraction, never
an unbounded "X% better"), and an ablation isolating the value of evidence
purchased by running the cheap method first.

![Pareto frontier](figures/fig1_pareto_all_policies.png)

## Honest limitations

Read this before the numbers above impress you more than they should.

- **Small synthetic instances only** (6 candidate sites, 15 customers) —
  chosen empirically, not from the original design: at the originally
  planned 15×60 scale, the exact MILP solver fails to prove optimality
  within 5s on ~48% of instances even with a stable solver backend. Nothing
  here is claimed about larger or real-world instances.
- **The one-shot ("predict, then commit") policy underperforms the naive
  baseline at every λ tested**, once judged by the combined quality+time
  loss rather than simple Pareto dominance — a real, cross-validated
  negative result, not a bug. The sequential ("run, observe, then decide")
  policy is the one that delivers positive gain, consistent with it being
  the intended headline policy rather than the fallback.
- **Feature-extraction time is the same order of magnitude as the cheapest
  solver** (~60μs vs ~70μs), not "orders of magnitude" smaller as originally
  hoped — a side effect of the instance-size reduction above.
- **The second domain (knapsack) got a lightweight pass, not the full
  treatment**: correctness-tested, proven to plug into the same `Allocator`
  machinery with one small, genuine generalization required (a `sense`
  field, since knapsack maximizes value while facility location minimizes
  cost), and has its own Pareto and gain-fraction figures (`figures/
  knapsack_*.png`) — but no 5-fold cross-validation, just a single
  train/test split. Its own pipeline also surfaced a new limitation: at
  microsecond solve times, wall-clock timing noise (~50% relative stdev)
  makes some cross-policy time comparisons unreliable, and one lambda
  (10,000) produces a gain fraction above 1 — logically impossible for a
  metric bounded by the oracle, and excluded from the headline figure for
  exactly that reason.
- **Three of the four analysis scripts are not yet "thin drivers"** over the
  `rungs` library — only `precompute.py` was refactored to route solving
  through `Domain.rungs()`; `evaluate_one_shot.py`, `evaluate_sequential.py`,
  and `generate_figures.py` still import solvers directly.
- Three fixed rungs, not a continuous effort dial. The sequential policy is
  optimal only for the *discretized* state; discretization loss is
  unquantified.

## Reproduce everything

```bash
./run_all.sh
```

Installs dependencies, runs the test suite, solves 800 facility-location
instances across 3 methods once, evaluates both policies via 5-fold
cross-validation, regenerates every figure in `figures/`, and runs the
lightweight knapsack pipeline proving the same library works on a second
domain. Takes a few minutes, dominated by the exact MILP solves.

## The idea, in three sentences

Every practical optimization system applies one solving method to every
instance, wasting effort on easy cases and giving up too early on hard ones.
The difficulty is that you can't observe how hard an instance is without
paying to solve it. This project predicts effort allocation from cheap,
pre-solve features instead — and, more interestingly, from evidence
purchased by actually running the cheapest method first and looking at what
happened.

## What's here

```
config.yaml             # pre-registered constants: seed, lambda grid, gap penalty
src/rungs/               # the library
  core.py                #   Rung / Domain protocols, Allocator (plug in a new domain here)
  policies.py             #   one-shot selector (LightGBM regression per rung)
  sequential.py            #   tabular backward induction for the sequential policy
  evaluate.py               #   quality_gap, loss, gain_fraction
  domains/
    facility_location*.py    # primary domain: capacitated facility location
    knapsack*.py               # second domain, proving the plugin interface
src/scripts/             # drivers
  precompute.py            #   solve everything once, build data/cache.parquet
  evaluate_one_shot.py       #   one-shot CV + first Pareto plot
  evaluate_sequential.py       #   sequential CV + cost-accounting check + ablation
  generate_figures.py            #   the full lambda sweep and all 5 figures
  run_knapsack_pipeline.py         #   full pipeline + 2 figures for the second domain
tests/                  # 41 tests, TDD throughout
figures/                # generated plots
internal/               # design doc, implementation guide, build-order TODO
```

## Future work

- The LLM cascade routing second domain (verifiable-benchmark model
  escalation) named in the original design as preferred over knapsack —
  skipped this round for lack of API budget.
- Finish routing the three remaining scripts through `Allocator`.
- A real write-up (`paper/paper.pdf` in the original design) — this README
  is currently the only narrative document.

---

## Development

```bash
uv sync --group dev        # install runtime + dev dependencies
pre-commit install         # activate local pre-commit hooks (once, after cloning)
uv run pytest              # run tests
```

### Local hooks (run on every `git commit`)

| Hook | What it does |
|---|---|
| detect-secrets | Blocks commits containing credentials or secrets |
| ruff | Lints and auto-fixes Python code |
| ruff-format | Formats Python code |
| trailing-whitespace / end-of-file-fixer / check-yaml / check-toml | General hygiene |
| commitizen | Enforces Conventional Commits message format |

### GitHub Actions CI (runs on every push and PR)

| Job | What it does |
|---|---|
| Lint | ruff check + ruff format --check |
| Type Check | mypy |
| Security | detect-secrets + pip-audit + Trivy (vuln/misconfig/secret scan) |
| Tests | pytest |

## Commit message format

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore, ci
```

Examples:
```
feat(solvers): add LP-relax-and-round medium rung
fix(policies): accumulate escalation cost across all rungs run
chore(deps): bump ruff to v0.17.0
```
