# Build this using the command
#
#     apptainer build --fakeroot litebird_sim.img Apptainer
#
# Run the container using the command
#
#     Apptainer run -H /tmp/$HOME ./litebird_sim.img COMMAND...
#
# where COMMAND... can be one of the following:
#
# -  `python3 FILENAME`: run a Python script
# -  `ipython`: start the IPython prompt
# -  `jupyter`: start the notebook interface
# -  `jupyter-lab`: start the JupyterLab interface
# -  `bash`: start a shell within the container

Bootstrap: docker
From: ubuntu:UBUNTU_VERSION

%help
    This container provides an installation of the LiteBIRD Simulation
    Framework, running Ubuntu Linux UBUNTU_VERSION with dnl
    ifdef(`MPI_LIB_NAME', MPI_LIB_NAME, `no MPI library').

    Running the container without arguments starts the IPython REPL;
    otherwise, the arguments are executed like if they were typed on the
    shell. Examples:

        # Start a bash shell
        apptainer run ./litebird_sim.img bash

        # Run Jupyter Lab. You must specify the home directory, otherwise
        # a read-only directory will be used as default, preventing you
        # from creating/modifying notebooks
        apptainer run ./litebird_sim.img \
            jupyter-lab --notebook-dir=$HOME

        # Avoid messing with your $HOME directory. Use this if you
        # see conflicts with your Miniconda/Anaconda installation
        apptainer run -H /tmp/$USER ./litebird_sim.img

%files
    runscript.py /opt/

%environment
    export LC_ALL=C
    export LC_NUMERIC=en_GB.UTF-8
    export XDG_CONFIG_HOME=/opt
    export XDG_CACHE_HOME=/tmp
    export MPLCONFIGDIR=/tmp/matplotlib
    export PYSM_LOCAL_DATA=/root/pysm3-data


%runscript
    export MPLCONFIGDIR=/tmp/matplotlib
    export PATH="/opt/litebird_sim/.venv/bin:$PATH"
    exec python3 /opt/runscript.py "$@"

%post

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
    apt-get install -y build-essential curl git python3 python3-dev python3-pip python3-venv

    # install MPI binaries
    case MPI_LIB_NAME in
        OpenMPI)
            echo "Installing OpenMPI..."
            apt-get install -y openmpi-bin libopenmpi-dev
            ;;
        MPICH)
            echo "Installing MPICH..."
            apt-get install -y mpich libmpich-dev
            ;;
        None)
            echo "Skipping MPI installation. BrahMap will also not be installed as it is dependent on MPI binaries."
            ;;
        *)
    esac

    # Configure pip
    export PIP_DISABLE_PIP_VERSION_CHECK=on
    export PIP_NO_CACHE_DIR=off

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
    uv --version

    # Clone source
    git clone -b BRANCH https://github.com/litebird/litebird_sim.git /opt/litebird_sim
    cd /opt/litebird_sim

    # Create environment with dependencies (including optional dependencies)
    if [[ MPI_LIB_NAME == OpenMPI || MPI_LIB_NAME == MPICH ]]; then
        uv sync --extra docs --extra dev --extra mpi --locked
    else
        uv sync --extra docs --extra dev --locked
    fi

    # Install a few handy packages
    uv pip install jupyterlab tqdm rich pudb

    # Install BrahMap
    # Only installs if openmpi or mpich are present in the container
    if [[ MPI_LIB_NAME == OpenMPI || MPI_LIB_NAME == MPICH ]]; then
        pip install git+https://github.com/anand-avinash/BrahMap.git
    fi

    # Define default IMo location for tests
    mkdir -p $HOME/.config/litebird_imo
    printf '[[repositories]]\nlocation = "%s/test/mock_imo/"\nname = "Mock IMO"\n' "$(pwd)" > "$HOME/.config/litebird_imo/imo.toml"

    #Cache pysm3 files for tests
    echo "Caching pysm3 data..."
    export PYSM_LOCAL_DATA=/root/pysm3-data
    git clone --depth 1 https://github.com/galsci/pysm-data $PYSM_LOCAL_DATA

    echo "Regenerating the documentation..."
    uv run sh bin/refresh_docs.sh

    # The reason why we run tests here is because AstroPy
    # needs to download a few files, and if we postpone this
    # to %test, the filesystem will be read-only.
    echo "Running the tests..."
    uv run python -m pytest -vv

    # Print some information
    echo "Information about this Apptainer image:"
    python3 --version
    gcc --version
    uv --version
    uv run python -c "import litebird_sim as lbs; print('Litebird_sim version: ', lbs.__version__)"

%test
    export PATH="/root/.local/bin:$PATH"
    export PYSM_LOCAL_DATA=/root/pysm3-data
    (cd /opt/litebird_sim && uv run python -m pytest)
