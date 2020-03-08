#!/bin/sh

readonly MAX_LINE_LENGTH=88

# Verify that the code is properly formatted
poetry run black --check --line-length=${MAX_LINE_LENGTH} -q .

# Check for common errors
poetry run flake8 --max-line-length=${MAX_LINE_LENGTH}

# Run the test suite
poetry run python3 -m pytest -vv
