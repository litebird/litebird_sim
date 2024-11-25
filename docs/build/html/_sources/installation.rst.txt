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


Hacking litebird_sim
--------------------

To develop ``litebird_sim``, you can create an enviroment, as described
above, then checkout and install a local copy of the framework.

.. code-block:: text

   # Create a virtual environment and activate it
   virtualenv my_venv && . my_venv/bin/activate

   # First clone the code
   git clone https://github.com/litebird/litebird_sim litebird_sim

   # Then install it with pip
   cd litebird_sim && pip install .


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
