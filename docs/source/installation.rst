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
