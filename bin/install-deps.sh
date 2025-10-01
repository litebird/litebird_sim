#!/bin/sh

set -e

MPI_IMPL=${1:-none}

case $(uname) in
Linux)
    # Update package list
    sudo apt-get update -q
    
    # Install FFTW if not present
    if ! dpkg -l | grep -q libfftw3-dev; then
        echo "Installing FFTW3..."
        sudo apt-get install -y -q libfftw3-dev
    else
        echo "FFTW3 already installed"
    fi
    
    case $MPI_IMPL in
        none)
            echo "Skipping MPI installation"
            ;;
        mpich)
            if ! command -v mpiexec >/dev/null 2>&1 || ! dpkg -l | grep -q libmpich-dev; then
                echo "Installing MPICH..."
                sudo apt-get install -y -q mpich libmpich-dev
            else
                echo "MPICH already installed"
            fi
            ;;
        openmpi)
            if ! command -v mpiexec >/dev/null 2>&1 || ! dpkg -l | grep -q libopenmpi-dev; then
                echo "Installing OpenMPI..."
                sudo apt-get install -y -q openmpi-bin libopenmpi-dev
            else
                echo "OpenMPI already installed"
            fi
            ;;
        *)
            echo "Error: Unknown MPI implementation: '$MPI_IMPL'"
            echo "Valid options: none, mpich, openmpi"
            exit 1
            ;;
    esac
    ;;
Darwin)
    # Install FFTW if not present
    if ! brew list fftw >/dev/null 2>&1; then
        echo "Installing FFTW..."
        brew install fftw
    else
        echo "FFTW already installed"
    fi
    
    case $MPI_IMPL in
        none)
            echo "Skipping MPI installation"
            ;;
        openmpi)
            if ! command -v mpiexec >/dev/null 2>&1 || ! brew list openmpi >/dev/null 2>&1; then
                echo "Installing OpenMPI..."
                brew install openmpi
            else
                echo "OpenMPI already installed"
            fi
            ;;
        *)
            echo "Error: Unknown MPI implementation for macOS: '$MPI_IMPL'"
            echo "Valid options: none, openmpi"
            exit 1
            ;;
    esac
    ;;
*)
    echo "Error: Unsupported operating system: $(uname)"
    exit 1
    ;;
esac

# Verify installations
echo "Verifying installations..."
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists fftw3; then
    echo "FFTW3 found"
fi

if [ "$MPI_IMPL" != "none" ] && command -v mpiexec >/dev/null 2>&1; then
    echo "MPI found"
fi

echo "Dependencies installation completed"
