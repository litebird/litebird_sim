#!/bin/bash

set -o errexit

# Verify that the code is properly formatted
black --check --diff .

# Check for common errors
flake8

# Run the test suite
python3 -m pytest -vv

# Run the doctests
make -C docs/ doctest
