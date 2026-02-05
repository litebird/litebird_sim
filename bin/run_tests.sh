#!/bin/bash

set -o errexit

# Run the test suite and print the N slowest tests
uv run python -m pytest --cov=litebird_sim --cov-report=xml --cov-report=term --durations=5 -vv

# Run the doctests
uv run make -C docs/ doctest
