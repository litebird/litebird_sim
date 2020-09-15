The Instrument Model Database (IMO)
===================================

To run a realistic simulation of an instrument, one needs to known its
details: the noise level of the detectors, the angular resolution of
the beams, etc. This kind of information is stored in an «instrument
model database», called IMO, and the LiteBIRD Simulation Framework
provides a few facilities to access it.

Let's start from a simple example, which will guide us in the
following paragraphs::

  from litebird_sim import Imo

  imo = Imo(flatfile_location="/storage/litebird/my_imo")
  scan_params = imo.query(
      "/releases/v0.10/satellite/scanning_parameters"
  )
  metadata = scan_params["metadata"]
  print(metadata["spin_sun_angle_deg"])

  # Output: the angle between the sun and the spin axis, in degrees

This example shows how to retrieve the parameters of the scanning
strategy followed by the LiteBIRD spacecraft. Everything revolves
around the class :class:`.Imo`, which reads the IMO from the files
saved in the folder ``/storage/litebird/my_imo``.

The call to ``imo.query`` retrieves a specific bit of information;
note that the access to the parameters is done using a file-like path,
``/releases/v0.10/satellite/scanning_parameters``. This is not a real
file, but a way to tell the IMO which kind of information is
requested: the ``/releases/v0.10`` specifies the IMO version to use,
and the remaining path ``/satellite/scanning_parameters`` points to
the information you're looking for.


Local/remote access to the IMO
----------------------------------
  
The IMO can be accessed either through an Internet connection or by
reading it directly from a file. Each approach has its own advantages
and disadvantages:

1. Having the IMO saved in a local file (like in the example above) is
   the fastest way to access its contents. However, it might not
   contain the latest version of the data you're looking for.

2. Using a remote IMO through an Internet connection ensures that you
   have access to the most updated version of the instrument; however,
   accessing it can be slow, and you're out of luck if your internet
   connection is unstable. Here is an example::

     from litebird_sim import Imo

     imo = Imo(
         url="https://dummy-litebird-imo.org",
         user="username",
         password="12345",
     )
     scan_params = imo.query(
         "/releases/v0.10/satellite/scanning_parameters"
     )

Once the :class:`.Imo` object has been created, accessing information
follows the same rules and uses the same syntax for paths.


How objects are stored
----------------------
   
Information in the IMO is stored using a hierarchical format, and
every datum is versioned. There are three fundamental concepts that
you need to grasp:

1. The IMO can store data files and Python-like dictionaries, called
   «metadata».
2. Different versions of the same data file can be kept at the same
   time in the IMO; we use the term **quantity** to refer to a data
   file but we don't care about its version.
3. Quantities can be stored in hierarchical structures, using the
   concept of **entity**, which enable to structure entities in a
   tree-like shape.

Here is an example:

.. code-block:: text

   satellite
   |
   +--- spacecraft
   |    |
   |    +--- *scanning_parameters*
   |
   +--- LFT
   |    |
   |    +--- 40_ghz
   |    |    |
   |    |    +--- *noise_characteristics*
   |    |    |
   |    |    +--- *band_response*
   |    |
   |    +--- 50_ghz
   |    |    |
   |    |    +--- *noise_characteristics*
   |    |    |
   |    |    +--- *band_response*
   |    ...
   |
   +--- MFT
   |    |
   |    ...
   |
   +--- HFT
        |
        ...

The diagram above shows how different quantities (marked using
asterisks in the diagram: ``scanning_parameters``,
``noise_characteristics``, ``band_response``) can be structured in a
tree-like structure using entities (the branches in the tree, e.g.,
``40_ghz``, ``LFT``). Different data files can be associated with
quantities like ``BAND_RESPONSE``.

In the code example at the top of this page, we accessed the scanning
strategy parameters using the string
``/releases/v0.10/satellite/scanning_parameters``. The meaning of the
string in terms of entities, quantities, and data files is the
following:

1. ``Satellite`` is an entity;
2. ``scanning_parameters`` is a quantity, because it's the last part
   of the path;
3. Of all the possible versions of the data file that have been saved
   in the quantity ``scanning_parameters``, we're asking for the one
   that is part of IMO 0.10 (the string ``v0.10`` in the path).

Apart from paths like
``/releases/v0.10/satellite/scanning_parameters``, there is a more
low-level method to access data files, using UUIDs. Each quantity and
each datafile is identified by a unique UUID, an hexadecimal string
that it's granted to be unique. This string is assigned automatically
when information is added to the IMO, and it can be used to retrieve
the information later. In Python, you can use the ``UUID`` type from
the `uuid library <https://docs.python.org/3/library/uuid.html>`_ to
encode this information from a string::

  from uuid import UUID

  # The string below is usually read from a parameter file
  my_uuid = UUID("5b9e3155-72f2-4e18-95d4-9881bc3e592d")

  # Use "my_uuid" to access data in the IMO
  scan_params = imo.query(my_uuid)

The advantage of the latter method is that you can access data files
that have not been formally included in a versioned IMO.

IMO and reports
---------------

When constructing a :class:`.Simulation` object, you should pass an
instance of a :class:`Imo` class. In this way, simulation modules can
take advantage of an existing connection to the IMO.

This has the additional advantage that the report produced at the end
of the simulation will include a list of all the data files in the IMO
that were accessed during the simulation.

There are cases when you want to query some information from the IMO,
but you do not want it to be tracked. (For instance, you are just
navigating through the tree of entities.) In this case, you can pass
``track=False`` to the :meth:`.Imo.query` method: the object you have
queried will not be included in the final report produced by the
:class:`.Simulation` object.

API reference
-------------

.. automodule:: litebird_sim.imo
    :members:
    :undoc-members:
    :show-inheritance:
