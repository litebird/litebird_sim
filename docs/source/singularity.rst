Using Singularity
=================

`Singularity <https://sylabs.io/docs/>`_ is a container platform that
helps users create isolated environments where programs can be run
without interfering with other libraries installed on the system. You
can consider a container as a zipped file containing a Linux
distribution (Ubuntu), a Python distribution, and the LiteBIRD
Simulation Framework, which is already installed and ready to be used.
It might look similar to a virtual machine but is way faster to start
(no boot time delay) and easier to use.

Let’s first state what are the *drawbacks* of running the framework in
a container:

- It is not meant to be a tool used by people developing the LiteBIRD
  Simulation Framework; it is only used by users because the source
  code for ``litebird_sim`` is stored in a read-only directory. (You
  could, however, use it to develop some new module in your home
  directory and merge it later.)

- It only runs under Linux.

- You must run a recent version of the Linux kernel (at least 3.18,
  released in 2015).

- The container takes considerable space (~0.5 GB) and could be a
  waste of disk space if you already have a working Python
  distribution.

- You cannot install it system-wide and call it within other Python
  programs you already use on your system. For instance, if you have
  installed library XYZ in your system, you cannot call XYZ *and*
  functions/classes in ``litebird_sim`` from the same Python script.
  (It's a *container*, after all.)

- Although it is a container, it is *still* possible to have conflicts
  with other programs installed on your machine (although there are
  simple workarounds);

However, the reason why we are providing this solution is because of
some significant advantages:

- You do not need to install/upgrade Python;

- No need to mess with virtual environments;

- Existing Python versions won’t conflict with Singularity containers
  (but see below for some caveats);

- All the framework, its dependencies, and the Python compiler itself
  are bundled in **one** file, which you can keep in your home
  directory or move around;

- It supports MPI, and thus it can be used on HPC clusters.

Typically, you might want to use our Singularity container if you just
want to run a Python script that calls ``litebird_sim`` but do not
want/cannot install the framework because of conflicts on your system.

To use the Singularity container, you must follow these steps:

1. Build a ``Singularity`` file; using the scripts provided by the
   LiteBIRD Simulation Framework is a matter of an instant;

2. Build the container; this requires a working internet connection
   and will take a few minutes;

3. Once the container is built, a new executable file is ready. With
   it, you can start IPython and JupyterLab or run Python programs
   calling the LiteBIRD Simulation Framework.

Let’s see the details of each step.

Build a ``Singularity`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build a file for Singularity, you must first clone the
``litebird_sim`` repository:

.. code-block:: text

   git clone https://github.com/litebird/litebird_sim litebird_sim

Enter the directory ``litebird_sim/singularity`` and run the script
``create-singularity-file.sh``. It takes the following arguments:

- The version number of the Ubuntu Linux distribution to use. Some
    possible choices are ``22.04``, ``24.04``, …; you should use the
    most recent LTS release, which is currently ``24.04``.

- A flag telling which version of MPI to install. Possible choices
  are:

  - ``openmpi``;

  - ``mpich``;

  - ``none`` (no MPI support).

  You should choose the same MPI implementation you are running on
  your system.

Here are a few usage examples; each of them creates a ``Singularity``
file in the current directory (i.e., ``litebird_sim/singularity``):

.. code-block:: text

   # Use Ubuntu Linux 24.04 and OpenMPI
   $ ./create-singularity-file.sh 24.04 openmpi

   # Use Ubuntu Linux 22.04 and MPICH
   $ ./create-singularity-file.sh 22.04 mpich

   # Use Ubuntu Linux 20.04 without MPI
   $ ./create-singularity-file.sh 20.04 none

Build the container
~~~~~~~~~~~~~~~~~~~

Once you have executed ``create-singularity-file.sh``, you will have a
``Singularity`` file. It's time to run ``singularity`` and create the
container:

.. code-block:: text

   singularity build --fakeroot litebird_sim.img Singularity

(The file name ``litebird_sim.img`` is the container to create. Of
course, you can pick the name you want; for example, if you are
creating several containers, you might name them
``litebird_sim_20.04_openmpi.img`` and so on.) The flag ``--fakeroot``
allows you to create an image even if you do not have superuser
powers.

If everything works as expected, in a few minutes you will have a
working container in file ``litebird_sim.img`` (which should be about
~0.5 GB in size).

To check that the container works correctly, run a self-test on it:

.. code-block:: text

   singularity test litebird_sim.img


Running the container
~~~~~~~~~~~~~~~~~~~~~

Once the container has been created, you can run it directly: the
IPython prompt will appear, and you can use ``litebird_sim``
immediately.

.. asciinema:: singularity_demo1.cast

You can use it to run scripts as well:

.. asciinema:: singularity_demo2.cast

.. note::

   You might wonder how the container could run the script
   ``test.py`` if the file was created outside the container. The
   reason is that Singularity, by default, mounts the home directory
   and the current directory in the container, so you can always
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

To obtain a short help about how to use the container, you can use the
command ``singularity run-help``:

.. asciinema:: singularity_help.cast

Finally, the following demo shows how to test the correctness of the
LiteBIRD Simulation Framework and browse a local copy of the
documentation. The key feature shown here is that running
``singularity shell litebird_sim.img`` starts a shell within the
container; you can then move to ``/opt/litebird_sim`` (the directory
where the framework has been installed) and run commands from there.

.. asciinema:: singularity_shell.cast

Running ``python3 -m http.server`` starts an HTTP server connected to
http://0.0.0.0:8000/: browsing to that URL will open your own local
copy of the User's manual for the LiteBIRD Simulation Framework.


Accessing the IMO from the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are exporting your home directory (the default), you should
have no problem accessing the IMO, provided that one of these
conditions apply:

- You are accessing a remote copy of the IMO;

- You are accessing a local copy of the IMO that resides within your
  home directory.
