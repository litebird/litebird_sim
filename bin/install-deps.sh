#!/bin/sh

# This script is based on the one in mpi4py's distribution
# It is currently used by GitHub Actions

set -e
case `uname` in
    Linux)
        sudo apt-get install libfftw3-dev
        ;;
    Darwin)
        brew install fftw
        ;;
esac
