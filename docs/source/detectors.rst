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

  mock_det = chinfo.get_boresight_detector(name="mydet")
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
in the IMO closely match the :class:`.DetectorInfo` and
:class:`.FreqChannelinfo` classes. In fact, they can be retrieved
easily from the IMO using the static methods
:meth:`.DetectorInfo.from_imo` and :meth:`.FreqChannelInfo.from_imo`::

  import litebird_sim as lbs

  # You must have configured the IMO before using this!
  imo = lbs.Imo()

  det = lbs.DetectorInfo.from_imo(
      imo=imo,
      url="/releases/v1.0/satellite/LFT/L1-040/"
          "L00_008_QA_040T/detector_info",
  )

  freqch = lbs.FreqChannelInfo.from_imo(
      imo=imo,
      url="/releases/v1.0/satellite/LFT/L1-040/channel_info",
  )


API reference
-------------

.. automodule:: litebird_sim.detectors
    :members:
    :undoc-members:
    :show-inheritance:
