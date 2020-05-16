#!/bin/sh

set -o nounset
set -o errexit

readonly MAX_LINE_LENGTH=88
readonly LITEBIRD_SIM_PATH="$HOME/litebird_sim"
readonly POETRY_PATH="$HOME/.poetry/bin/poetry"

cd $LITEBIRD_SIM_PATH

# Verify that the code is properly formatted
python3 "$POETRY_PATH" run black --check --line-length=${MAX_LINE_LENGTH} -q .

# Check for common errors
python3 "$POETRY_PATH" run flake8 --max-line-length=${MAX_LINE_LENGTH}

# Run the test suite
python3 "$POETRY_PATH" run python3 -m pytest --doctest-modules --ignore=test/mpi -vv
