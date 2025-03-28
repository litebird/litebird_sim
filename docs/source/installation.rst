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
   poetry export --without-hashes $EXTRAS -o requirements.txt
   pip install --upgrade pip

   # Install litebird_sim in the environment
   pip install -e .


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


Development with MPI
~~~~~~~~~~~~~~~~~~~~

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
