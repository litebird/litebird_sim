.. _tutorial:

Tutorial
========

This section contains a short tutorial that describes how to get ready
to use the framework. It assumes that you have already installed the
``litebird_sim`` framework; refer to :ref:`installation_procedure`.
For a nice and exhaustive example on how to use the framework in the 
LiteBIRD case see the `example notebook <https://github.com/litebird
/litebird_sim/blob/master/notebooks/litebird_sim_example.ipynb>`_.

A «Hello world» example
-----------------------

In this section we assume that you are running these command
interactively, either using the REPL (``python`` or IPython are both
fine) or a Jupyter notebook.

The first thing to do is to create a folder where you will write your
script. The LiteBIRD Simulation Framework is a *library*, and thus it
should be listed as one of the dependencies of your project. It's
better to use virtual environments, so that dependencies are properly
tracked:

.. code-block:: sh

  python -m virtualenv ./lbs_tutorial_venv
  source lbs_tutorial_venv/bin/activate

The last line of code might vary, depending on the Operating System
and the shell you are using; refer to the `Python documentation
<https://docs.python.org/3/tutorial/venv.html>`_ for more information.

Once you have activated the virtual environment, you should install
the LiteBIRD Simulation Framework. As it is `registered on PyPI
<https://pypi.org/project/litebird-sim/>`_, it's just a matter of
calling ``pip``:

.. code-block:: sh

  pip install litebird_sim

To ensure reproducibility of your results, it is good to keep track of
the version numbers used by your program. We will immediately create a
file ``requirements.txt``, which can be used by other people to ensure
that they are using the very same version of the LiteBIRD Simulation
Framework (as well as any other package you might want to install with
``pip``) as ours:

.. code-block:: sh

  pip freeze > requirements.txt

(If you use a Version Control System like ``git``, it is a good idea
to add ``requirements.txt`` to the repository.) Anybody will then be
able to install the same versions of each package as you by using the
command ``pip install -r requirements.txt``.

If you got no errors, you are ready to write your first program! To
follow an ancient tradition, we will write a «Hello world!» program.
Create a new file called ``my_script.py`` in the folder you just
created, and write the following::

  # File my_script.py
  import litebird_sim as lbs

  print("Starting the program...")
  sim = lbs.Simulation(base_path="./tut01")
  sim.append_to_report("Hello, world!")
  sim.flush()
  print("Done!")

Surprisingly, the program did not output ``Hello world`` as you might
have expected! Instead, it created a folder, named ``tut01``, and
wrote a few files in it:

.. code-block:: sh

  $ ls ./tut01
  report.html    report.md    sakura.css
  $

Open the file ``report.html`` using your browser (e.g., ``firefox
tut01/report.html``), and the following page will appear:

.. image:: images/tutorial-bare-report.png
   :width: 512
   :align: center
   :alt: Screenshot of part of the tutorial produced by our script

Among the many lines of text produced by the report, you can spot the
presence of our «Hello, world!» message. Hurrah!

Let's have a look at what happened. The first line imports the
``litebird_sim`` framework; since the name is quite long, it's
customary to shorten it to ``lbs``::

  import litebird_sim as lbs

The next interesting stuff happens when we instantiate a
:class:`.Simulation` object::

  sim = lbs.Simulation(base_path="./tut01")

Creating a :class:`.Simulation` object makes a lot of complicated
things happen beyond the scenes. In this short example, what happens
is the following:

1. The code checks if a directory named ``tut01`` exists; if not, it
   is created.
2. An empty report is created.

The report is where the results of a simulation will be saved, and
sections can be appended to it using the method
:meth:`.Simulation.append_to_report`, like we did in our example::

  sim.append_to_report("Hello, world!")

The report is actually written to disk only when
:meth:`.Simulation.flush` is called::

  sim.flush()

This is the most basic usage of the :class:`.Simulation` class; for
more information, refer to :ref:`simulations`.

In the next section, we will make something more interesting using the
framework.


Interacting with the IMO
------------------------

It's not clear why we should want to install a whole framework just to
create a HTML file, no matter how nice it looks. Things begin to get
interesting once we start using other facilities provided by our
framework.

Simulations for real-life experiments often require to use several
parameters that describe the instruments being simulated: how many
detectors there are, what are their properties, etc. These information
are usually kept in an Instrument MOdel database, IMO for short.

The LiteBIRD IMO is managed using `instrumentdb
<https://github.com/ziotom78/instrumentdb>`_, a web-based database,
but it can be retrieved also as a bundle of files. The LiteBIRD
simulation framework seamlessy interacts with the IMO database and
permits to retrieve all the parameters that describe the LiteBIRD
instruments.

The best way to interact with the IMO is to have a local copy
installed on your laptop. You should ask permission to the LiteBIRD
Simulation Team for downloading the IMO from the (protected) site
`litebird_imo <https://github.com/litebird/litebird_imo>`_. Save it in
a folder on your computer, e.g., ``/storage/litebird_imo``, and then
run the following command:

.. code-block:: text

  python -m litebird_sim.install_imo

and run the program interactively to configure the IMO. You typically
want to use a «local copy»; specify the folder where the file
``schema.json`` you downloaded before resides (under
``/storage/litebird_imo`` in our case). Save the changes by pressing
``s``, and you will have your IMO configured.

Our next example will use the IMO to run something more interesting::

  import litebird_sim as lbs

  sim = lbs.Simulation(base_path="./tut02")
  lft_file = sim.imo.query(
      "/releases/v1.0/satellite/LFT/instrument_info"
  )
  sim.append_to_report(
      "The instrument {{ name }} has {{ num }} channels.",
      name=lft_file.metadata["name"],
      num=lft_file.metadata['number_of_channels'],
  )
  sim.flush()

If you run this program, it will produce a report containing the
following message:

.. code-block:: text

  The instrument LFT has 12 channels.

Let's dig into the code of the example. The first line looks almost
the same as in the previous example::

  # Previous example
  sim = lbs.Simulation(base_path="./tut01")

  # This example
  sim = lbs.Simulation(base_path="./tut02")

Yet a big difference went unnoticed: since you configured the IMO
using the ``install_imo`` module, the :class:`.Simulation` class
managed to read the database contents and initialize a set of member
variables. This is why we have been able to write the next line::

  lft_file = sim.imo.query(
      "/releases/v1.0/satellite/LFT/instrument_info"
  )

Although the parameter looks like a path to some file, it is a
reference to a bit of information in the IMO; specifically, a set of
parameters characterizing the instrument LFT (Low Frequency
Telescope). This call retrieves the parameters and returns a
:class:`.DataFile` object, which contains the information in its
``metadata`` field. These are used to fill the report::

  sim.append_to_report(
      "The instrument {{ name }} has {{ num }} channels.",
      name=lft_file.metadata["name"],
      num=lft_file.metadata['number_of_channels'],
  )

The code should be self-evident: the keywords ``name`` and ``num`` are
used in the text to put some actual values within the placeholders
``{{ … }}``. This is the syntax used by `Jinja2
<https://jinja.palletsprojects.com/en/2.11.x/>`_, a powerful
templating library.

This example showed you how to retrieve information from the IMO and
introduced some features of the method
:meth:`.Simulation.append_to_report`. To learn a bit more about the
the IMO, read :ref:`imo`; for reporting facilities, read
:ref:`reporting`.


Creating a coverage map
-----------------------

We're now moving to something more «astrophysical»: we will write a
program that computes the sky coverage of a scanning
strategy over some time.

The code is complex because it uses several concepts explained in the
section :ref:`scanning-strategy`; in fact, this example is very
similar to the one shown in that section. It's not needed that you
understand everything, just have a look at the code that generates the
report::

  import litebird_sim as lbs
  import healpy, numpy as np
  import matplotlib.pylab as plt
  import astropy.units as u

  sim = lbs.Simulation(
      base_path="./tut04",
      name="Simulation tutorial",
      start_time=0,
      duration_s=86400.,
  )

  sim.set_scanning_strategy(
      scanning_strategy=lbs.SpinningScanningStrategy(
          spin_sun_angle_rad=np.deg2rad(30), # CORE-specific parameter
          spin_rate_hz=0.5 / 60,     # Ditto
          # We use astropy to convert the period (4 days) in
          # seconds
          precession_rate_hz=1.0 / (4 * u.day).to("s").value,
      )
  )

  sim.set_instrument(
      InstrumentInfo(
          name="core",
          spin_boresight_angle_rad=np.deg2rad(65),
      ),
  )

  sim.set_hwp(IdealHWP(ang_speed_radpsec=0.1))

  sim.create_observations(
      detectors=lbs.DetectorInfo(name="foo", sampling_rate_hz=10),
  )

  sim.compute_pointings()

  for cur_obs in sim.observations:
      nside = 64
      pixidx = healpy.ang2pix(nside, pointings[:, 0], pointings[:, 1])
      m = np.zeros(healpy.nside2npix(nside))
      m[pixidx] = 1
      healpy.mollview(m)

  sim.append_to_report("""

  ## Coverage map

  Here is the coverage map:

  ![](coverage_map.png)

  The fraction of sky covered is {{ seen }}/{{ total }} pixels
  ({{ "%.1f" | format(percentage) }}%).
  """,
      figures=[(plt.gcf(), "coverage_map.png")],
      seen=len(m[m > 0]),
      total=len(m),
      percentage=100.0 * len(m[m > 0]) / len(m),
  )

  sim.flush()

This example is interesting because it shows how to interface Healpy
with the report-generation facilities provided by our framework. As
explained in :ref:`scanning-strategy`, the code above does the
following things:

1. It sets the scanning strategy, triggering the computation of set
   of quaternions that encode the orientation of
   the spacecraft for the whole duration of the simulation (86,400
   seconds, that is one day);
2. It creates an instance of the class :class:`.InstrumentInfo` and
   it registers them using the method
   :meth:`.Simulation.set_instrument`;
3. It instantiates a new class that represents an ideal Half-wave Plate
   (HWP);
4. It sets the detectors to be simulated and allocates the TODs through
   the call to :meth:`.Simulation.create_observations`;
5. It generates a pointing information matrix through the call to
   :meth:`.Simulation.compute_pointings`;
6. It produces a coverage map by setting to 1 all those pixels that
   are visited by the directions encoded in the pointing information
   matrix. To do this, it iterates over all the instances of the
   class :class:`.Observation` in the
   :class:`.Simulation` object. (In this simple example, there is only
   one :class:`.Observation`, but in more complex examples there can
   be many of them.)

If you run the example, you will see that the folder ``tut04`` will be
populated with the following files:

.. code-block:: sh

  $ ls tut04
  coverage_map.png  report.html  report.md  sakura.css
  $

A new file has appeared: ``coverage_map.png``. If you open the file
``report.html``, you will get the map in the report (here the image
has been cropped a bit, because the report is longer):

.. image:: images/tutorial-coverage-map.png
   :width: 512
   :align: center
   :alt: Screenshot of part of the tutorial produced by our script

The elements shown in this tutorial should allow you to generate more
complex scripts. The next section detail the features of the framework
in greater detail.
