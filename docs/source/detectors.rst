.. _detectors:

Detectors, channels, and instruments
====================================

The core of the LiteBIRD instruments is the set of *detectors* that
perform the measurement of the electromagnetic power entering through
the telescope. Several simulation modules revolve around the concept
of «detector», «frequency channel», and «instrument», and the LiteBIRD
Simulation Framework implements a few classes and functions to handle
the information associated with them.

Consider this example::

  import litebird_sim as lbs

  detectors = [
      lbs.DetectorInfo(
          name="Dummy detector #1",
          net_ukrts=100.0,
          bandcenter_ghz=123.4,
          bandwidth_ghz=5.6,
      ),
      lbs.DetectorInfo(
          name="Dummy detector #2",
          net_ukrts=110.0,
          fwhm_arcmin=65.8,
      ),
  ]

  # Now simulate the behavior of the two detectors
  # …

Here you see that the :class:`.DetectorInfo` class can be used to
store information related to single detectors, and that you can choose
which information provide for each detector: the example specifies
bandshape information only for the first detector (``Dummy detector
#1``), but not for the other. Missing information is usually
initialized with zero or a sensible default. The simulation modules
provided by the framework typically expect one or more
:class:`.DetectorInfo` objects to be passed as input.

Detectors are grouped according to their nominal central frequency in
*frequency channels*; there are several frequency channels in the
three LiteBIRD instruments (LFT, MFT, HFT), and sometimes one does not
need to simulate each detector within a frequency channel, but only
the channel as a whole. In this case there is the
:class:`.FreqChannelInfo` class, which can produce a mock
:class:`.DetectorInfo` object by taking out the «average» information
of the frequency channel:

.. testcode::

  import litebird_sim as lbs

  chinfo = lbs.FreqChannelInfo(
      bandcenter_ghz=40.0,
      net_channel_ukrts=40.0, # Taken from Sugai et al. 2020 (JLTP)
      number_of_detectors=64, # 32 pairs
  )

  # Return a "mock" detector that is representative of the
  # frequency channel
  mock_det = chinfo.get_boresight_detector(name="mydet")

  # Look, ma, a detector!
  assert isinstance(mock_det, lbs.DetectorInfo)
  assert mock_det.name == "mydet"

  print("The NET of one detector at 40 GHz is {0:.1f} µK·√s"
        .format(mock_det.net_ukrts))

.. testoutput::

  The NET of one detector at 40 GHz is 320.0 µK·√s

Finally, the :class:`.Instrument` class holds information about one of
the three instruments onboard the LiteBIRD spacecraft (LFT, MFT, and
HFT).


Reading from the IMO
--------------------

The way information about detectors and frequency channels is stored
in the IMO (see :ref:`imo`) closely match the :class:`.DetectorInfo` and
:class:`.FreqChannelInfo` classes. In fact, they can be retrieved
easily from the IMO using the static methods
:meth:`.DetectorInfo.from_imo` and :meth:`.FreqChannelInfo.from_imo`.

The following example uses the PTEP IMO to show how to use the API:

.. testcode::

  import litebird_sim as lbs

  # The location of the PTEP IMO is stored in the constant
  # PTEP_IMO_LOCATION
  imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

  det = lbs.DetectorInfo.from_imo(
      imo=imo,
      url="/releases/vPTEP/satellite/LFT/L1-040/"
          "000_000_003_QA_040_T/detector_info",
  )

  print(f"The bandcenter for {det.name} is {det.bandcenter_ghz} GHz")

  freqch = lbs.FreqChannelInfo.from_imo(
      imo=imo,
      url="/releases/vPTEP/satellite/LFT/L1-040/channel_info",
  )

  print(
      f"The average bandcenter for {freqch.channel} "
      f"is {freqch.bandcenter_ghz} GHz"
  )

.. testoutput::

    The bandcenter for 000_000_003_QA_040_T is 40.0 GHz
    The average bandcenter for L1-040 is 40.0 GHz


Detectors in parameter files
----------------------------

It is often the case that the list of detectors to simulate must be
read from a parameter file. There are several situations that
typically need to be accomodated:

1. In some simulations, you just need to simulate one detector (whose
   parameters can be taken either from the IMO or from the parameter
   file itself);
2. In other cases, you want to simulate all the detectors in one or
   more frequency channels: in this case, you would like to avoid
   specifying them one by one!
3. In other cases, you might want to specify just a subset
4. Finally, you might base your simulation on the IMO definition of a
   detector/channel but twiddle a bit with some parameters.

The LiteBIRD simulation framework provides a way to read a set of
detectors/channels from a dictionary, which can be used to build a
list of :class:`.DetectorInfo`/:class:`.FreqChannelInfo` objects.

In the following examples, we will refer to a «mock» IMO, which
contains information about a few fake detectors. It's included in the
source distribution of the framework, under the directory
``litebird_sim/test/mock_imo``, and it defines one frequency channel
with four detectors. Here is a summary of its contents:

.. testcode::

   from pathlib import Path
   import litebird_sim as lbs

   # This ensures that the mock IMO can be found when generating this
   # HTML page
   imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

   # This UUID refers to a 140 GHz "channel_info" object.
   ch = imo.query("/data_files/463e9ea9-c1f0-484d-9bfd-05092851d8f4")
   metadata = ch.metadata

   print("Here are the contents of the mock IMO:")
   print(f'Channel: {metadata["channel"]}')

   detector_names = metadata["detector_names"]
   print(f'There are {len(detector_names)} detectors')
   print("Here are the first 5 of them:")
   for name, obj in list(zip(metadata["detector_names"],
                             metadata["detector_objs"]))[0:5]:
       det_obj = imo.query(obj)
       det_metadata = det_obj.metadata
       bandcenter_ghz = det_metadata["bandcenter_ghz"]
       print(f'  {name}: band center at {bandcenter_ghz:.1f} GHz')

.. testoutput::

    Here are the contents of the mock IMO:
    Channel: M1-140
    There are 366 detectors
    Here are the first 5 of them:
      001_002_030_00A_140_T: band center at 140.0 GHz
      001_002_030_00A_140_B: band center at 140.0 GHz
      001_002_031_15B_140_T: band center at 140.0 GHz
      001_002_031_00B_140_B: band center at 140.0 GHz
      001_002_022_15A_140_T: band center at 140.0 GHz


Now, let's turn back to the problem of specifying a set of detectors
in a parameter file. The following TOML file shows some of the
possibilities granted by the framework. The parameter `random_seed`
is mandatory for the :class:`.Simulation` constructor.

.. literalinclude:: ../det_list1.toml
   :language: toml

If the TOML file you are using in your simulation follows this
structure, you can use the function
:func:`.detector_list_from_parameters`, which parses the parameters
and uses an :class:`.Imo` object to build a list of
:class:`.DetectorInfo` objects.

The following code will read the TOML file above and produce a list of
6 detectors:

.. testcode::

   from pathlib import Path
   import litebird_sim as lbs

   imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

   # Tell Simulation to use the PTEP IMO and to
   # load the TOML file shown above
   sim = lbs.Simulation(imo=imo, parameter_file="det_list1.toml")

   det_list = lbs.detector_list_from_parameters(
       imo=sim.imo,
       definition_list=sim.parameters["detectors"],
   )

   for idx, det in enumerate(det_list):
       print(f"{idx + 1}. {det.name}: band center at {det.bandcenter_ghz} GHz")

.. testoutput::

    1. 000_000_003_QA_040_T: band center at 40.0 GHz
    2. 001_002_030_00A_140_T: band center at 140.0 GHz
    3. 001_002_030_00A_140_B: band center at 140.0 GHz
    4. foo_boresight: band center at 140.0 GHz
    5. planck30GHz: band center at 28.4 GHz
    6. 000_000_003_QA_040_T: band center at 40.0 GHz

You are not forced to use ``detectors`` as the name of the parameter
in the TOML file, as :func:`.detector_list_from_parameters` accepts a
generic list. As an example, consider the following TOML file. Note
again the mandatory parameter `random_seed`.

.. literalinclude:: ../det_list2.toml
   :language: toml

Its purpose is to provide the parameters for a two-staged simulation,
and each of them requires its own list of detectors. For this purpose,
it uses the TOML syntax ``[[simulation1.detectors]]`` to specify that
the ``detectors`` list belongs to the section ``simulation1``, and
similarly for ``[[simulation2.detectors]]``. Here is a Python script
that interprets the TOML file and prints the detectors to be used at
each stage of the simulation:

.. testcode::

   from pathlib import Path
   import litebird_sim as lbs

   imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

   # Tell the Simulation object that we want to use the PTEP IMO
   sim = lbs.Simulation(imo=imo, parameter_file="det_list2.toml")

   det_list1 = lbs.detector_list_from_parameters(
       imo=sim.imo,
       definition_list=sim.parameters["simulation1"]["detectors"],
   )

   det_list2 = lbs.detector_list_from_parameters(
       imo=sim.imo,
       definition_list=sim.parameters["simulation2"]["detectors"],
   )

   print("Detectors to be used in the first simulation:")
   for det in det_list1:
       print(f"- {det.name}")

   print("Detectors to be used in the second simulation:")
   for det in det_list2:
       print(f"- {det.name}")


.. testoutput::

    Detectors to be used in the first simulation:
    - 000_000_004_QB_040_B
    Detectors to be used in the second simulation:
    - 000_000_004_QB_040_T


API reference
-------------

.. automodule:: litebird_sim.detectors
    :members:
    :undoc-members:
    :show-inheritance:
