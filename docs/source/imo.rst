.. _imo:

The Instrument Model Database (IMO)
===================================

To run a realistic simulation of an instrument, one needs to known its
details: the noise level of the detectors, the angular resolution of
the beams, etc. This kind of information is stored in an «instrument
model database», called IMO, and the LiteBIRD Simulation Framework
provides a few facilities to access it.

.. note::

   The type of information stored in the LiteBIRD Instrument Model
   Database is extremely diverse: it goes from CAD/optical/thermal
   models to high-level parameters representing some general
   characteristics of the instrument.

   The LiteBIRD Simulation Framework enables to access any information
   stored in the IMO, but it only provides full support for those
   parameters that are actually used by the framework itself. (As an
   example, you can use this interface to download a CAD file, but the
   framework does not implement any facility to render/analyze the
   file.)

Let's start from a simple example, which will guide us in the
following paragraphs::

  from litebird_sim import Imo

  imo = Imo(flatfile_location="/storage/litebird/my_imo")
  scan_params = imo.query(
      "/releases/v1.3/satellite/scanning_parameters"
  )
  metadata = scan_params.metadata
  print(metadata["spin_sun_angle_deg"])

  # Output: the angle between the sun and the spin axis, in degrees


This example shows how to retrieve the parameters of the scanning
strategy followed by the LiteBIRD spacecraft. Everything revolves
around the class :class:`.Imo`, which reads the IMO from the files
saved in the folder ``/storage/litebird/my_imo``.

The call to ``imo.query`` retrieves a specific bit of information;
note that the access to the parameters is done using a file-like path,
``/releases/v1.3/satellite/scanning_parameters``. This is not a real
file, but a way to tell the IMO which kind of information is
requested: the ``/releases/v1.3`` specifies the IMO version to use,
and the remaining path ``/satellite/scanning_parameters`` points to
the information you're looking for.


.. _imo-configuration:

Configuring the IMO
-------------------

The LiteBIRD Simulation Framework comes with a bundled IMO, which contains
only public information about the mission and the instruments. It was built
using the numbers reported in the paper
`Probing cosmic inflation with the LiteBIRD cosmic microwave background
polarization survey <https://academic.oup.com/ptep/article/2023/4/042F01/6835420>`_
(PTEP, 2022). You can use this database by passing the path
``lbs.PTEP_IMO_LOCATION`` to the constructor of the :class:`.Imo` class::

    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

However, to run serious simulations you should grab a copy of the
official IMO database released by the IMO team and install it on your
computer. If you just need basic information, it is enough to download
the JSON file associated with any data release from the site
https://litebirdimo.ssdc.asi.it (authentication is required). For
extensive simulations that use data files like beams and bandpasses,
you should ask the ASI SSDC for a tarball bundle containing the whole
database and decompress it in a folder on your computer.

Assuming that you put the JSON file or the decompressed tarball on a
local folder like ``/storage/litebird_imo``, run the following
command:

.. code-block:: text

  python -m litebird_sim.install_imo

This is an interactive program that lets you to configure the IMO.
Choose the item “local copy” and specify the folder. Save the changes
by pressing ``s``, and you will have your IMO configured.

To sum up, there are three possibilities to access an IMO:

1. Use the bundled PTEP IMO by passing
   ``flatfile_location=lbs.PTEP_IMO_LOCATION`` to the constructor of
   the :class:`.Imo` class. In this case, the only IMO release tag
   that you will see is ``vPTEP``.

2. Download a JSON file from the ASI SSDC website, save it in a folder
   and run ``python -m litebird_sim.install_imo`` to make it visible.
   In this case, only basic information will be available.

3. Ask the ASI SSDC for a bundled tarball containing one or more IMO
   versions. Decompress the tarball in a folder and run ``python -m
   litebird_sim.install_imo`` to make it visible.


Local/remote access to the IMO
----------------------------------
  
The IMO can be accessed either through an Internet connection or by
reading it directly from a file. Each approach has its own advantages
and disadvantages:

1. Having the IMO saved in a local file (like in the example above) is
   the fastest way to access its contents. However, it might not
   contain the latest version of the data you're looking for. Note
   that the framework does not require that a *full* database be
   available, as only the data that are actually needed in a
   simulation are retrieved: for this reason, you might opt to
   download a reduced version of the IMO containing only those
   high-level parameters that you want to use in your simulation.

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
         "/releases/v1.3/satellite/scanning_parameters"
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
``/releases/v1.3/satellite/scanning_parameters``. The meaning of the
string in terms of entities, quantities, and data files is the
following:

1. ``Satellite`` is an entity;
2. ``scanning_parameters`` is a quantity, because it's the last part
   of the path;
3. Of all the possible versions of the data file that have been saved
   in the quantity ``scanning_parameters``, we're asking for the one
   that is part of IMO 1.3 (the string ``v1.3`` in the path).

Apart from paths like
``/releases/v1.3/satellite/scanning_parameters``, there is a more
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

Browsing the IMO database
-------------------------

The LiteBIRD Simulation Framework provides a text-mode program to
navigate the contents of the IMO. You can start it using the following
command:

.. code-block:: text

    python3 -m litebird_sim.imobrowser

Here is a short demo of its capabilities:

.. asciinema:: imobrowser.cast
   :preload: 1

When «opening» a data file, you can copy either the full path of the
data file or its UUID (the hexadecimal string uniquely identifying it)
in the clipboard: this can be handy when you are developing codes that
need to access specific objects. On Ubuntu, clipboard copying only
works if you have ``xclip`` or ``xsel`` installed; on
Ubuntu/Mint/Debian Linux, you can install ``xclip`` with the following
command:

.. code-block:: text

    sudo apt-get install xclip

If ``xclip`` is not installed, clipboard functions are automatically
disabled.


IMO and reports
---------------

When constructing a :class:`.Simulation` object, you should pass an
instance of an :class:`.Imo` class. In this way, simulation modules can
take advantage of an existing connection to the IMO.

This has the additional advantage that the report produced at the end
of the simulation will include a list of all the data files in the IMO
that were accessed during the simulation.

There are cases when you want to query some information from the IMO,
but you do not want it to be tracked. (For instance, you are just
navigating through the tree of entities but are not going to *use* the
quantities you are querying.) In this case, you can pass
``track=False`` to the :meth:`.Imo.query` method: the object you have
queried will not be included in the final report produced by the
:class:`.Simulation` object.

API reference
-------------

.. automodule:: litebird_sim.imo
    :members:
    :undoc-members:
    :show-inheritance:
