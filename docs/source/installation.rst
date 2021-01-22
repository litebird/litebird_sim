.. _installation_procedure:

Installing the framework
========================

Until the framework is not published on PyPI, you have to grab a
``.tar.gz`` file from the `litebird_sim release page
<https://github.com/litebird/litebird_sim/releases>`_ and install it
manually with the following procedure:

.. code-block:: text

   # Create a directory where you're going to write your scripts,
   # notebooks, etc.
   mkdir -p ~/litebird && cd ~/litebird

   # Create a virtual environment
   virtualenv venv

   # Activate the environment
   . venv/bin/activate

   # Here you must specify the path to the .tar.gz file
   # that you downloaded before
   pip install ~/Downloads/litebird_sim-X.Y.Z.tar.gz

When the command is completed, check that everything works by issuing
the following command at the terminal prompt:

.. code-block:: text

   python -c "import litebird_sim; print(litebird_sim.__version__)"

If it prints a version number that matches the one of the ``.tar.gz``
file, this means you are done!


Hacking litebird_sim
--------------------

To develop ``litebird_sim``, you should first install `poetry
<https://poetry.eustace.io/>`_ and then build the framework with the
following commands:

.. code-block:: text

   git clone https://github.com/litebird/litebird_sim litebird_sim
   cd litebird_sim && poetry install

Be sure **not** to work within a Conda environment, nor to create a
virtual environment! The purpose of the ``poetry`` command is exactly
to transparently manage virtual environments.

To produce new releases, you can use ``poetry build -f sdist``: this
will create a ``.tar.gz`` file in the directory ``dist``, which can be
uploaded in the `release page
<https://github.com/litebird/litebird_sim/releases>`_ of
``litebird_sim``.


Using Singularity
-----------------

`Singularity <https://sylabs.io/docs/>`_ is a container platform that
helps user in creating isolated environments where programs can be ran
without interfering with other libraries installed on the system.

You can use Singularity 3+ to run the LiteBIRD Simulation Framework;
the advantages are the following:

- No need to install Python;
- No need to use virtual environments;
- Existing Python versions won't conflict with Singularity's
  containers (but see below for a caveat);
- All the framework, its dependencies, and the Python compiler itself
  are bundled in **one** file, which you can keep in your home
  directory;
- It supports MPI, and thus it can be used on HPC clusters.

Be aware that Singularity has a few disadvantages:

- It only runs under Linux;
- You must run a recent version of the Linux kernel (at least 3.18,
  which was released in 2015);
- It is *still* possible to have conflicts with other programs
  installed on your machine (although there are simple workarounds);
- The container is huge (~0.5 GB) and could be a waste of disk space
  if you already have a working Python distribution;
- You cannot install it system-wide and call it within other Python
  programs you are already using on your system. (It's a *container*,
  after all.)

To use Singularity, you must follow these steps:

1. Build a ``Singularity`` file; using the scripts provided by the
    LiteBIRD Simulation Framework, it is a matter of a few seconds;

2. Build the container; this requires a working internet connection
   and will take a few minutes;

3. Once the container is built, a new huge executable file is ready to
   be used: with it, you can start IPython, JupyterLab, or run Python
   programs calling the LiteBIRD Simulation Framework.

Let's see the details of each step.

Build a ``Singularity`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter the directory ``litebird_sim/singularity`` and run the script
``create-singularity-file.sh``. It takes the following arguments:

- The version number of the Ubuntu Linux distribution to use. Valid
    choices are ``18.04`` and ``20.04``; you should use the most
    recent LTS release, which is currently ``20.04``.

- A flag telling which version of MPI to install. Possible choices
  are:

  - ``openmpi``;
  - ``mpich``;
  - ``none`` (no MPI support).

  You should choose the same MPI implementation you are running on
  your system.

Here are a few usage examples; each of them creates a ``Singularity``
file in the current directory (i.e., ``litebird_sim/singularity``):

.. code-block: text

   # Use Ubuntu Linux 20.04 and OpenMPI
   $ ./create-singularity-file.sh 20.04 openmpi

   # Use Ubuntu Linux 20.04 and MPICH
   $ ./create-singularity-file.sh 20.04 mpich

   # Use Ubuntu Linux 18.04 without MPI
   $ ./create-singularity-file.sh 18.04 none
   
Build the container
~~~~~~~~~~~~~~~~~~~

Once you have executed ``create-singularity-file.sh``, you will have a
``Singularity`` file. It's time to run ``singularity`` and create the
container:

.. code-block: text

   singularity build --fakeroot litebird_sim.img Singularity

(The file name ``litebird_sim.img`` is the container to create. Of
course, you can pick the name you want.) The flag ``--fakeroot``
permits to create an image even if you do not have superuser powers.

If everything works as expected, in a few minutes you will have a
working container in file ``litebird_sim.img`` (which should be about
~0.5 GB in size).

Running the container
~~~~~~~~~~~~~~~~~~~~~

Once the container has been created, you can run it directly: the
IPython prompt will appear, and you can use ``litebird_sim``
immediately.

.. asciinema:: singularity_demo1.cast
   :preload: 1

You can use it to run scripts as well:

.. asciinema:: singularity_demo2.cast
   :preload: 1

.. note::

   You might wonder how could the container run the script
   ``test.py``, if the file was create *outside* the container. The
   reason is because Singularity by default mounts the home directory
   and the current directory in the container, so that you can always
   access whatever you have in these directories while running stuff
   from the container.

   This might lead to undesired effects, though. Suppose you have
   installed Anaconda/Miniconda under your home directory: in this
   case, clashes between the Python packages installed within the
   container and Anaconda might happen!

   In this case, you can run the container using the syntax
   ``singularity run -H /tmp/$USER``: this will mount the home
   directory on a directory under ``/tmp``. (You can specify another
   directory, of course.)
             
To use MPI, you must call ``mpirun`` *outside* the container:

.. asciinema:: singularity_demo3.cast
   :preload: 1
