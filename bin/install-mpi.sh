#!/bin/sh

# This script is based on the one in mpi4py's distribution
# It is currently used by GitHub Actions

set -e
case `uname` in
Linux)
  case $1 in
    none)
      echo "Skipping the installation of a MPI library"
      ;;      
    mpich) set -x;
      echo "Installing mpich"
      sudo apt-get install -y -q mpich libmpich-dev
      ;;
    openmpi) set -x; 
      echo "Installing openmpi"
      sudo apt-get install -y -q openmpi-bin libopenmpi-dev
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
Darwin)
  case $1 in
    none)
      echo "Skipping the installation of a MPI library"
      ;;      
    mpich) set -x;
      echo "Installing mpich"
      brew install mpich
      ;;
    openmpi) set -x;
      echo "Installing openmpi"
      brew install openmpi
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
esac
