.. _beamconvolution:

Convolve Alms with a Beam to fill a TOD
=======================================

Minimal documentation should contain:
- quick descrition of the math (reference to papers)
- conventions 
- details on the interface and examples 
- description of the SphericalHarmonics class and how to store alms
- example with high level interface


The framework provides :func:`.add_convolved_sky`, a routine which 
takes sky and beam alms perform harmonic space convolution and fills
the detector timestreams. It implements both the algebra for...  

Methods of the Simulation class
-------------------------------

The class :class:`.Simulation` provides the function
:func:`.Simulation.convolve_sky`, which takes sky alms and perform the
convolution 

API reference
-------------

.. automodule:: litebird_sim.beam_convolution
    :members:
    :undoc-members:
    :show-inheritance:
