Introduction
============

`LiteBIRD <https://www.litebird.jp/eng/>`_ is a JAXA-led satellite mission that
will map the polarization of the Cosmic Microwave Background over the full sky,
searching for the primordial *B*-mode signature of cosmic inflation. Meeting its
requirements relies on detailed end-to-end simulations of the instruments and of
the observation process.

The **LiteBIRD Simulation Framework** (LBS) is the collaboration's toolkit for
those simulations: it turns *sky models* and *instrument descriptions* into
*time-ordered detector data* (TOD), injects instrumental noise and systematic
effects, and reduces the TOD back into sky maps. The rest of this manual
documents these steps part by part.

If you are new to the framework, start with the installation instructions below
and then work through the :doc:`tutorial <tutorial>`. If you use LBS in your
work, please cite it (:cite:`2025:litebird:tomasi`).

.. toctree::
   :maxdepth: 1

   installation.rst
   singularity.rst
