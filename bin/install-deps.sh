#!/bin/sh

# This script is based on the one in mpi4py's distribution
# It is currently used by GitHub Actions

set -e
case `uname` in
Linux)
  case $1 in
    none)
      echo "Skipping the installation of a MPI library"
      sudo apt-get install -y -q libfftw3-dev
      ;;      
    mpich) set -x;
      echo "Installing mpich"
      sudo apt-get install -y -q mpich libmpich-dev libfftw3-dev
      ;;
    openmpi) set -x; 
      echo "Installing openmpi"
      sudo apt-get install -y -q openmpi-bin libopenmpi-dev libfftw3-dev
      ;;
    *)
      echo "Unknown MPI implementation: \"$1\""
      exit 1
      ;;
  esac
  ;;
Darwin)
  case $1 in
    none)
      brew install fftw
      ;;      
    openmpi) set -x;
      echo "Installing openmpi"
      brew install openmpi fftw
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
esac
