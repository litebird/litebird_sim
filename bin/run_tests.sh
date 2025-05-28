#!/bin/bash

set -o errexit

# Verify that the code is properly formatted
ruff format --diff .

# Check for common errors
ruff check .

# Run the test suite and print the N slowest tests
python3 -m pytest --cov=litebird_sim --cov-report=xml --cov-report=term --durations=5 -vv

# Run the doctests
make -C docs/ doctest
