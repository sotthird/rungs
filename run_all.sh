#!/usr/bin/env bash
# Regenerates every result (cache, policy evaluations, figures) from a clean
# checkout. See internal/IMPLEMENTATION.md §1, §13.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

echo "Installing dependencies..."
uv pip install --python .venv/bin/python -e ".[dev]"

echo
echo "Running tests..."
.venv/bin/python -m pytest tests/

echo
echo "Precomputing solver results (facility location, ~800 instances x 3 rungs)..."
.venv/bin/python src/scripts/precompute.py

echo
echo "Evaluating one-shot selector..."
.venv/bin/python src/scripts/evaluate_one_shot.py

echo
echo "Evaluating sequential escalation policy..."
.venv/bin/python src/scripts/evaluate_sequential.py

echo
echo "Generating figures and gain-fraction sweep..."
.venv/bin/python src/scripts/generate_figures.py

echo
echo "Running the second-domain (knapsack) pipeline — proves the Allocator"
echo "library generalizes beyond facility location..."
.venv/bin/python src/scripts/run_knapsack_pipeline.py

echo
echo "Done. Figures written to figures/, cache written to data/cache.parquet."
