.. _installation_procedure:

Installation
============

The framework is `registered on PyPI <https://pypi.org/project/litebird-sim/>`_,
it can be installed with the following procedure:

.. code-block:: text

   # Create a directory where you're going to write your scripts,
   # notebooks, etc.
   mkdir -p ~/litebird && cd ~/litebird

   # Create a virtual environment with
   virtualenv lbs_env
   # or with
   python3 -m venv lbs_env

   # Activate the environment
   . lbs_env/bin/activate

   # Finally install litebird_sim with pip
   pip install litebird_sim

When the command is completed, check that everything works by issuing
the following command at the terminal prompt:

.. code-block:: text

   python -c "import litebird_sim"

A similar procedure can be used with conda:

.. code-block:: text

   # Create a conda environment
   conda create -n lbs_env python=3.9

   # Activate the environment
   conda activate lbs_env

   # Finally install litebird_sim with pip
   pip install litebird_sim

If you plan to use any of the facilities provided by ``ducc``, you are
advised to compile it from source, follow the instructions in
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
LBS; we use `Poetry <https://python-poetry.org/docs/basic-usage/>`_
for this.

.. code-block:: text

   # Create a virtual environment with
   virtualenv lbs_env
   # or with
   python3 -m venv lbs_env

   # Activate the environment
   . lbs_env/bin/activate

   # Clone the code locally
   git clone https://github.com/litebird/litebird_sim litebird_sim

   # If you are running Poetry 2.0 or above, run this to install the `export` plugin
   pip install poetry-plugin-export

   # Generate requirements.txt and install all the dependencies
   poetry export --without-hashes -o requirements.txt
   pip install --upgrade pip

   # Install litebird_sim in the environment
   pip install -e .

To install LBS with optional dependencies, one can add the option
``-E <dependency_name>`` while calling ``poetry export``. See the section on
:ref:`install-dependencies` for more details.

Run code validators
~~~~~~~~~~~~~~~~~~~

As every commit and pull request is validated through `ruff
<https://github.com/astral-sh/ruff>`_, you might want to run them
before pushing modifications to the GitHub repository. In this case
enter the ``litebird_sim`` directory and run the following command:

.. code-block:: text

   # Always remember to activate your virtual environment!
   . my_venv/bin/activate

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

   If you have created a virtual environment to work with
   ``litebird_sim`` (as you should have), just install it using ``pip``:

   .. code-block:: text

       pip install mpi4py

   That's it: the next time you run a script that uses ``litebird_sim``,
   MPI functions will be automatically enabled in the framework. See the
   chapter :ref:`using_mpi` for more details.

2. ``jupyter``  

   This dependency installs the packages that can be used to work with LBS in a
   jupyter notebook.

3. ``brahmap``  

   BrahMap is an external map-making framework and it supports optimal map-making
   with LBS simulations. LBS in turn, offers a high level interface to call
   BrahMap. The additional packages needed to use BrahMap can be installed with
   ``brahmap`` dependency. See the section on :ref:`mapmaking` for details on using
   BrahMap with LBS.

.. _maximize-performance:

Maximize the performance
------------------------

For some of the most CPU-intensive tasks, LBS relies on the `ducc
<https://gitlab.mpcdf.mpg.de/mtr/ducc>`_ library, which is written in
C++. When you run ``pip install litebird_sim``, you are downloading a
prebuilt binary of the library which is portable among many
architectures but might not exploit the CPU you are using to its
maximum potential.

If you plan to use CPU-intensive tasks like beam convolution (see
chapter :ref:`beamconvolution`), you will
surely take advantage of a natively compiled binary. To do this, you
must have a valid C++ compiler, as it is specified in `ducc’s README
<https://gitlab.mpcdf.mpg.de/mtr/ducc>`_.

To use a natively-compiled binary for ``ducc``, create a virtual
environment and install ``litebird_sim`` as usual, then *uninstall*
``ducc`` and re-install it again, this time telling ``pip`` to compile
it from source.

.. code-block:: text

   mkdir -p ~/litebird && cd ~/litebird
   python3 -m venv lbs_env
   . lbs_env/bin/activate
   pip install litebird_sim

   # Remove the version downloaded by default
   pip uninstall ducc0

   # Re-install ducc0 forcing to skip the download of the binary
   pip3 install --no-binary ducc0 ducc0

If you experience problems with the last command because of
compilation errors, please open an issue on the `ducc repository page
<https://gitlab.mpcdf.mpg.de/mtr/ducc/-/issues>`_.

