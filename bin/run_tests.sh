#!/bin/bash

set -o errexit

# Verify that the code is properly formatted
poetry run black --check --diff .

# Check for common errors
poetry run flake8

# Run the test suite
poetry run python3 -m pytest -vv

# Run the doctests
poetry run make -C docs/ doctest
