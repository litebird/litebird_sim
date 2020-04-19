#!/bin/sh

# This script is based on the one in mpi4py's distribution
# It is currently used by the Travis CI script

set -e
case `uname` in
Linux)
  case $1 in
    none)
      ;;      
    mpich) set -x;
      sudo apt-get install -y -q mpich libmpich-dev
      ;;
    openmpi) set -x;
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
      ;;      
    mpich) set -x;
      brew install mpich
      ;;
    openmpi) set -x;
      brew install openmpi
      ;;
    *)
      echo "Unknown MPI implementation:" $1
      exit 1
      ;;
  esac
  ;;
esac
