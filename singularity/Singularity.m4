# Build this using the command
#
#     singularity build --fakeroot litebird_sim.img Singularity
#
# Run the container using the command
#
#     singularity run -H /tmp/$HOME ./litebird_sim.img COMMAND...
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
    singularity run ./litebird_sim.img bash

    # Run Jupyter Lab. You must specify the home directory, otherwise
    # a read-only directory will be used as default, preventing you
    # from creating/modifying notebooks
    singularity run ./litebird_sim.img \
        jupyter-lab --notebook-dir=$HOME

    # Avoid messing with your $HOME directory. Use this if you
    # see conflicts with your Miniconda/Anaconda installation
    singularity run -H /tmp/$USER ./litebird_sim.img

%files
        runscript.py /opt/

%environment
        export LC_ALL=C
        export LC_NUMERIC=en_GB.UTF-8
        export XDG_CONFIG_HOME=/opt
        export XDG_CACHE_HOME=/opt

%runscript
        exec python3 /opt/runscript.py "$@"

%post

        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
        apt-get install -y build-essential curl git python3 python3-dev python3-pip python3-venv MPI_LIB

        # Configure pip
        export PIP_DISABLE_PIP_VERSION_CHECK=on
        export PIP_NO_CACHE_DIR=off

        # Install poetry. Do *not* use the default destination, as
        # Singularity exports the host's home directory by default,
        # thus incrementing the chance of clashes
        export POETRY_VERSION=1.1.11
        export POETRY_HOME=/opt/poetry

        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
        export PATH="$POETRY_HOME/bin:$PATH"

        # Install all the dependencies, build a .tar.gz file, install
        # it, run the tests and build all the documentation. (The
        # reason why we run tests here is because AstroPy needs to
        # download a few files, and if we postpone this to %test the
        # filesystem will be read-only).
        git clone https://github.com/litebird/litebird_sim.git /opt/litebird_sim

        cd /opt/litebird_sim

        poetry export --without-hashes POETRY_MPI -E docs -E jupyter -o requirements.txt
        pip3 install -r requirements.txt
        poetry build -f sdist
        pip3 install dist/litebird_sim-$(poetry version --short).tar.gz

        # Install a few handy packages
        pip3 install jupyterlab tqdm rich pudb

        echo "Regenerating the documentation..."
        sh bin/refresh_docs.sh

        echo "Running the tests..."
        python3 -m pytest -vv

        # Print some information
        echo "Information about this Singularity image:"
        python3 --version
        gcc --version
        python3 -c "import litebird_sim as lbs; print('Litebird_sim version: ', lbs.__version__)"

%test
	(cd /opt/litebird_sim && python3 -m pytest)
