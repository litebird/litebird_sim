#!/bin/bash

set -o errexit

# Verify that the code is properly formatted
ruff format --diff . --config ./pyproject.toml

# Check for common errors
ruff check . --config ./pyproject.toml

# Run the test suite
python3 -m pytest -vv

# Run the doctests
make -C docs/ doctest
