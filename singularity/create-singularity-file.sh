#!/bin/bash

readonly ubuntu_version="$1"
readonly mpi_library="$2"
if [ "$3" == "" ]; then
    readonly branch="master"
else
    readonly branch="$3"
fi

if [ "$mpi_library" == "" ]; then
    cat <<EOF
Usage: $(basename $0) UBUNTU_VERSION MPI_LIBRARY [BRANCH]

where:

  UBUNTU_VERSION is the number of the Ubuntu distribution to use (e.g., 24.04)
  MPI_LIBRARY    is either "openmpi", "mpich", or "none"
  BRANCH         specifies the branch to clone, or "master" if nothing is provided
EOF

    exit 1
fi

case "$mpi_library" in
    openmpi)
        mpi_lib='-DMPI_LIB=openmpi-bin'
        mpi_lib_name='-DMPI_LIB_NAME=OpenMPI'
        poetry_mpi='-DPOETRY_MPI=--extras=mpi'
        ;;
    
    mpich)
        mpi_lib='-DMPI_LIB=mpich'
        mpi_lib_name='-DMPI_LIB_NAME=MPICH'
        poetry_mpi='-DPOETRY_MPI=--extras=mpi'
        ;;

    none)
        mpi_lib='-DMPI_LIB'
        mpi_lib_name=""
        poetry_mpi='-DPOETRY_MPI'
        ;;
    
    *)
        echo "Unknown MPI library \"$mpi_library\""
        exit 1
        ;;
esac

m4 \
    -DUBUNTU_VERSION="$ubuntu_version" \
    -DBRANCH="$branch" \
    $mpi_lib $mpi_lib_name $poetry_mpi \
    Singularity.m4 > Singularity

cat <<EOF
File "Singularity" has been created. Now build an image using the command

    singularity build --fakeroot litebird_sim.img Singularity
EOF
