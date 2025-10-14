.. _installation_procedure:

Installation
============

LBS can be used on Linux or Mac OS X machines. Windows is supported
only through `Windows Subsystem for Linux
<https://learn.microsoft.com/en-us/windows/wsl/>`_. You must have
Python 3.10 at least to use the framework; on some systems, you must
ensure to have the Python development libraries, otherwise some
package dependencies might not be installed correctly. (For instance,
on Fedora Linux you must install the package ``python3-devel`` with
the command ``sudo dnf install python3-devel``.)

The framework is `registered on PyPI <https://pypi.org/project/litebird-sim/>`_,
so it can be installed using ``pip``:

.. code-block:: text

   # Create a directory where you're going to write your scripts,
   # notebooks, etc.
   mkdir -p ~/litebird && cd ~/litebird

   # Create a virtual environment with
   python3 -m venv lbs_env

   # Activate the environment
   . lbs_env/bin/activate

   # Finally install litebird_sim with pip
   pip install litebird_sim

When the command is completed, check that everything works by issuing
the following command at the terminal prompt:

.. code-block:: text

   python3 -c "import litebird_sim as lbs; print(lbs.__version__)"

A similar procedure can be used with conda:

.. code-block:: text

   # Create a conda environment
   conda create -n lbs_env python=3.12

   # Activate the environment
   conda activate lbs_env

   # Finally install litebird_sim with pip
   pip install litebird_sim

If you plan to use any of the facilities provided by ``ducc``, you are
advised to compile it from source; follow the instructions in
:ref:`maximize-performance`.

Optional dependencies can be installed along with LBS as following:

.. code-block:: text

   pip install litebird_sim[<dependency_name>]

where you can substitute the ``<dependency_name>`` with one of the
dependencies (or comma-separated list of the dependencies) listed in
:ref:`install-dependencies`


Hacking LBS
-----------

If you plan to work on the source code of LBS, you should clone the
repository and create a virtual environment for it. The virtual
environment must be prepared by installing all the packages needed by
LBS; we use `uv <https://docs.astral.sh/uv/>`_
for this. Uv is a fast package manager and virtualenv tool, similar to
a combination of ``pip`` and ``venv``.

.. code-block:: text

   # Install uv (if not already installed)
   pip install uv

   # Clone the code locally
   git clone https://github.com/litebird/litebird_sim litebird_sim
   cd litebird_sim

   # Install all dependencies and create virtual environment
   uv sync --all-extras

At this point, you will have two ways to run use LBS:

- As ``uv`` created a virtual environment under ``.venv``, you can
  activate it and forget about ``uv`` for the rest of the session:

  .. code-block:: sh

     # Activate the environment
     source .venv/bin/activate

- Otherwise, you add ``uv run`` before *any* command invoking
  ``python`` or ``python3``:

  .. code-block:: sh

     # Or simply run commands with uv run (recommended)
     uv run python3 your_script.py

To install LBS with optional dependencies, you can use the ``--extra`` option
with ``uv sync --extra <dependency_name>``; for instance::

    uv sync --extra docs --extra mpi

See the section on :ref:`install-dependencies` for more details.

Run code validators
~~~~~~~~~~~~~~~~~~~

As every commit and pull request is validated through `ruff
<https://github.com/astral-sh/ruff>`_, you might want to run them
before pushing modifications to the GitHub repository. In this case
enter the ``litebird_sim`` directory and run the following command:

.. code-block:: text

   # Always remember to activate your virtual environment!
   . my_venv/bin/activate

   # Install pre-commit using `uv`
   uv tool install pre-commit --with pre-commit-uv

   # Install some useful hooks for git
   pre-commit install

What this command does is to install a few «pre-commit» hooks: they
are programs that are run whenever you run ``git commit`` and do some
basic checks on your code before actually committing it. These checks
are the same that are run by GitHub once you push your changes in a
pull request, so they can save you several back-and-forth iterations.

.. _install-dependencies:

Installing LBS with optional dependencies
-----------------------------------------

The LiteBIRD Simulation Framework offers additional functionalities that can
be enabled optionally. These optional functionalities are supported via
optional dependencies that can be installed by the users as required.

LBS offers 3 optional dependencies:

1. ``mpi``

   As explained in the chapter :ref:`using_mpi`, the LiteBIRD Simulation
   Framework supports MPI. To use it, you must ensure that `mpi4py
   <https://mpi4py.readthedocs.io/en/stable/>`_ is installed.

   If you are using ``uv`` (recommended), you can install the MPI
   optional dependency:

   .. code-block:: text

       uv sync --extra mpi

   Alternatively, if you are within a virtual environment you can
   install mpi4py directly:

   .. code-block:: text

       pip install mpi4py

   That's it: the next time you run a script that uses ``litebird_sim``,
   MPI functions will be automatically enabled in the framework. See the
   chapter :ref:`using_mpi` for more details.

2. ``docs``

   This dependency installs the packages that are used to build the documentation.

3. ``brahmap``

   BrahMap is an external map-making framework and it supports optimal map-making
   with LBS simulations. LBS in turn, offers a high level interface to call
   BrahMap. The additional packages needed to use BrahMap can be installed with
   ``brahmap`` dependency. See the section on :ref:`mapmaking` for details on using
   BrahMap with LBS.

.. _maximize-performance:

Maximize the performance
------------------------

*This part is optional and mostly relevant only for power users
running large simulations.*

For some of the most CPU-intensive tasks, LBS relies on the `ducc
<https://gitlab.mpcdf.mpg.de/mtr/ducc>`_ library, which is written in
C++. When you run ``pip install litebird_sim``, you are downloading a
prebuilt binary of the library which is portable among many
architectures but might not exploit the CPU you are using to its
maximum potential.

If you plan to use CPU-intensive tasks like beam convolution (see
chapter :ref:`beamconvolution`), you will surely take advantage of a
natively compiled binary. To do this, you must have a valid C++
compiler; check the most up-to-date requirements in `ducc’s README
<https://gitlab.mpcdf.mpg.de/mtr/ducc>`_.

To use a natively-compiled binary for ``ducc``, create a virtual
environment using the commands listed above and install
``litebird_sim`` as usual, then *uninstall* ``ducc`` and re-install it
again, this time telling ``pip`` to compile it from source.

.. code-block:: text

   mkdir -p ~/litebird && cd ~/litebird
   python3 -m venv lbs_env
   . lbs_env/bin/activate
   pip install litebird_sim

   # Remove the version downloaded by default
   pip uninstall ducc0

   # Re-install ducc0 forcing to skip the download of the binary
   pip install --no-binary ducc0 ducc0

If you experience problems with the last command because of
compilation errors, please open an issue on the `ducc repository page
<https://gitlab.mpcdf.mpg.de/mtr/ducc/-/issues>`_.

