#!/bin/bash

set -o errexit

readonly MAX_LINE_LENGTH=88

# Verify that the code is properly formatted
poetry run black --check --line-length=${MAX_LINE_LENGTH} .

# Check for common errors
poetry run flake8 --max-line-length=${MAX_LINE_LENGTH} --extend-exclude docs

# Run the test suite
poetry run python3 -m pytest -vv

# Run the doctests
poetry run make -C docs/ doctest
