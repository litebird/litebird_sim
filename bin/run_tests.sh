#!/bin/bash

# Description: This script allows the user to run different tests depending
# on the command line arguments
# Typical usage: `./bin/run_tests.sh [OPTION(S)]`, where the options could be 
# `--all`, `--docs`, `--notebooks`, `--codestyle`, `--unittests`, 
# `--mpitests`, `--help`

set -euo pipefail

# Ensure the script runs from the repository root
cd "$(dirname "$0")/.."

# Use `uv run` only if it is available
if command -v uv >/dev/null 2>&1; then
    UV_RUN="uv run"
else
    UV_RUN=""
fi

doc_test() {
    echo "Running doctests..."
    $UV_RUN make -C docs/ doctest
}

notebook_test() {
    echo "Running notebook tests..."
    $UV_RUN python -m pytest --nbmake -n auto --durations=5 -vv notebooks/*.ipynb --deselect=notebooks/litebird_sim_smarties.ipynb
}

codestyle_test() {
    echo "Running codestyle tests..."
    
    # ruff linter and formatter
    $UV_RUN ruff check . --config ./pyproject.toml
    $UV_RUN ruff format --diff . --config ./pyproject.toml

    # type checker
    $UV_RUN ty check .
}

serial_utest(){
    echo "Running serial unit tests..."
    $UV_RUN python -m pytest --cov=litebird_sim --cov-report=xml --cov-report=term --durations=5 -vv
}

mpi_utest() {
    echo "Running MPI related unit tests..."
    
    # See issue #457: <https://github.com/litebird/litebird_sim/issues/457>
    if [[ "$(uname)" == "Darwin" ]]; then
        export HDF5_USE_FILE_LOCKING=FALSE
    fi

    # Pass --map-by :OVERSUBSCRIBE only when using OpenMPI
    if mpiexec --version 2>&1 | grep -Eqi "OpenRTE|Open MPI|OpenMPI|PRTE"; then
        MPI_ARGS="--map-by :OVERSUBSCRIBE"
    else
        MPI_ARGS=""
    fi
    
    for proc in 1 2 3 4 6; do
        PYTHONPATH=. mpiexec $MPI_ARGS -n $proc $UV_RUN python -m pytest ./test/test_mpi.py -v
    done
    
    for proc in 1 2 4; do
        PYTHONPATH=. mpiexec $MPI_ARGS -n $proc $UV_RUN python -m pytest ./test/test_detector_blocks.py -v
    done

    for proc in 1 2; do
        PYTHONPATH=. mpiexec $MPI_ARGS -n $proc $UV_RUN python -m pytest ./test/test_shared_memory_pointings.py -v
    done

    for proc in 1 4; do
        PYTHONPATH=. mpiexec $MPI_ARGS -n $proc $UV_RUN python -m pytest ./test/test_mpi_n4.py -v
    done
}

run_all() {
    echo "Running all tests..."
    
    codestyle_test
    doc_test
    serial_utest
    mpi_utest
    notebook_test
}

# if no arguments are passed, run fast checks and unit tests 
# (skipping notebooks and mpi unittest)
if [[ "$#" -eq 0 ]]; then
    codestyle_test
    doc_test
    serial_utest
    exit 0
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --all)
            run_all
            shift
            ;;
        --docs)
            doc_test
            shift
            ;;
        --notebooks)
            notebook_test
            shift
            ;;
        --codestyle)
            codestyle_test
            shift
            ;;
        --unittests)
            serial_utest
            shift
            ;;
        --mpitests)
            mpi_utest
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--all] [--docs] [--notebooks] [--codestyle] [--unittests] [--mpitests] [--help]"
            exit 0
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done


