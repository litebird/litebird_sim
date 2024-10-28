#!/bin/bash

set -o errexit

# Verify that the code is properly formatted
ruff format --diff .

# Check for common errors
ruff check .

# Run the test suite
python3 -m pytest -vv

# Run the doctests
make -C docs/ doctest
