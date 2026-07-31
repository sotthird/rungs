# rungs

> Deciding how hard to think about a problem, before thinking about it.

![CI](https://github.com/sotthird/rungs/actions/workflows/ci.yml/badge.svg)

Instance-adaptive solver selection: predict per-instance solve difficulty from
cheap, pre-solve features, and route each instance to the cheapest solving
method ("rung") that reaches acceptable quality — instead of committing one
fixed method for every instance regardless of how much effort it deserves.

Full writeup, headline plot, and repro command land at the end of the build
(see project TODO). This README will be filled in per the design spec's
README skeleton once results exist — no numbers are reported before then.

---

## Development

```bash
uv sync --group dev        # install runtime + dev dependencies
bash setup-pipeline.sh     # activate local pre-commit hooks (once, after cloning)
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
