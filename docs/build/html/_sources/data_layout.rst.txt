Data layout
===========

This page discusses how data is laid down in memory, and it provides
some hints about how to use NumPy efficiently with the memory layout
used by the LiteBIRD Simulation Framework. We focus mostly on the
detectors time-ordered-data (TOD), as they are the largest object we
expect to handle frequently.

The LiteBIRD Simulation Framework stores TOD as a matrix with one row
per detector and the other detector attributes as arrays with one
entry per detector. This means that, given an observation ``obs``, you
access quantities with patterns like ``obs.fknee[12]`` (as opposed to
``obs.detectors[12].fknee``). Note you can easily write single lines
that operate on all the detectors::

  # Apply to each detector its own calibration factor
  obs.tod *=  obs.calibration_factors[:, None]
  # Generate white noise at different level for each detector
  obs.tod = (np.random.normal(size=obs.tod.shape)
             * obs.wn_level[:, None])


TOD as two-dimensional arrays
-----------------------------

Organizing the TOD in a matrix is ideal because it enables fast access
to (1) many samples for one detector and (2) many detectors for one
time sample. In fact, TOD are a large set of samples. Each sample is
identified by the time at which it was taken and by the detector that
took it. If all the detectors are synchronously sampled, these samples
can be organized in a matrix D-by-T, where D is the number of
detectors and T is the number of time samples. Following
``numpy.ndarray`` notation, we denote it as ``tod[row_index,
col_index]``. Accessing the :math:`i`-th row can be done using the
notation ``tod[i]``: for instance, ``tod[3]`` gets the fourth row (the
TOD of the fourth detector); accessing the fourth column, i.e., the
fourth sample of each detector, can be done with ``tod[:, 3]`` (remember that in Python we start counting from 0). So, the
way the framework keeps TODs in memory makes easy to operate on the
«detector dimension» as well as on the «time dimension».

You can take advantage of NumPy functions to easily propagate
operations along one dimension or another. For instance, taking the
time stream of the mean signal across the focal plane is done with
``tod.mean(axis=0)``. In general, an extremely large number of
operations can be expressed in terms of ``numpy`` functions and they
can operate easily and efficiently over axes of multi-dimensional
arrays. Similarly, if ``C`` is a cross-talk matrix, the operation ``C
@ tod`` mixes the TODs.
   
The previous examples are not only easy to write, they are also (very likely)
as fast as they can be (compared to different data layouts). This is
particularly true for the last example. The matrix multiplication ``C @ tod``
calls internally some highly optimized dense linear algebra routines.

Even when there is not really communications between detectors
involved, having data in a 2-dimensional array produces (arguably)
cleaner code and sometimes faster code (never slower). Suppose you
have a *list* of ``calibration_factors`` and a *list* of time streams
``tod``. You can apply the calibration factors with

.. code-block:: python 

  for calibration_factor, det_tod in zip(calibration_factors, tod):
      det_tod *= calibration_factor

but if ``calibration_factors`` is an *array* and ``tod`` a *matrix* you can
achieve the same result with the easier and (typically) faster

.. code-block:: python 

  tod *= calibration_factors[:, None]

Splitting the TOD matrix
------------------------

When working on a laptop (or a single compute node) we can live in the
ideal case discussed above. We can benefit both from the API
simplifications and the performance of the compiled codes that the
numerical libraries typically call. Moreover, these libraries are
often threaded (or easily threadable) and therefore we can in
principle leverage on all the processors available.

However, when working on clusters, we have to split this matrix into
pieces. We resort to supercomputers either when we want more CPU power
than a laptop or because we need more memory to store our data.
Both motivations apply to full-scale LiteBIRD simulations (4000
detectors, sampled at 20 GHz for 3 years take approximately 10 TB).
Therefore, we have to distribute the matrix and a compute node has
access only to the portion of the matrix that is in its memory.

To do this, we split the matrix into blocks that are stored as
(sub-)matrices. At the present stage, there is no constraint on the
shape of these blocks, they can also be a single row or a single
column. The important thing is that, if the block spans more than a
row, it is stored as a matrix (a 2-dimensional array). This means
that, for the time- and detector-chunk assigned to a compute node, all
the benefits discussed in the previous section apply.


Choosing the shape of the blocks
--------------------------------

The most convenient block shape depends on the application:

- Some operations prefer to store an entire row. For example,
  :math:`4\pi` beam convolution has a memory overhead of the order of
  the GB for each beam, which is in principle different for each
  detector. For this operation is therefore more convenient to hold a
  detector for the entire length of the mission.

- However, other operations prefer to have an entire column. For
  example, computing the common mode across the focal plane requires
  information from all the detectors at every time sample. This is
  trivial if the full detector column is in the memory, but it is very
  complicated if it is scattered across many compute nodes.

  Another example is cross-talks. Mixing the TOD of different
  detectors is trivial when everything is in memory, but otherwise it
  requires sending large messages across the network. Sending messages
  is not only a performance issue, it means that probably a custom
  algorithm has to be written, and this algorithm will probably
  require MPI communications, which are notoriously hard to develop
  and debug.

Since there is no one-size-fit-all solution, the LiteBIRD Simulation
Framework keeps the most generic approach. The shape of the blocks
depends on the application, and it is possible to change it during the
execution. (However, this takes time and should thus be done only when
strictly necessary.)


Caveats
-------

An important thing to remember is that, if the TOD matrix is chunked
along the detector dimension, only the corresponding portion of the
property array is detained in memory.
  
This has a few important implications:

1. Regardless if and how the TOD is distributed, both ``obs.tod[i]``
   and ``obs.wn_level[i]`` refer to the same detector;

2. ``obs.tod`` and ``obs.wn_level`` have the same length;

3. Operations like ``obs.tod * obs.wn_level[:, None]`` are correct.
