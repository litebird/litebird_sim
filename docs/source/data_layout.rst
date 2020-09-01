Data layout
===========

This page discusses how data is laid down in memory.
We focus mostly on the detectors time-ordered-data (TOD):
They are the largest object we expect to handle frequently
and therefore they are the driver of our decisions.

.. toggle-header::
 :header: In summary, we store TOD as a matrix with one row per detector and the other detector attributes as arrays with one entry per detector

   This means that, given an observation ``obs``, you access quantities
   with patterns like ``obs.fknee[12]`` (as opposed to ``obs.detectors[12].fknee``).
   Note you can easily write single lines that operate on all the detectors

   .. code-block:: python 

     # Apply to each detector its own calibration factor
     obs.tod *=  obs.calibration_factors[:, None]
     # Generate white noise at different level for each detector
     obs.tod = (np.random.normal(size=obs.tod.shape)
                * obs.wn_level[:, None])

TOD as two-dimensional arrays
-----------------------------
.. toggle-header::
 :header: Organizing the TOD in a matrix is ideal because 1) we can and 2) it allows fast access to both many time samples for fixed detector and many detectors for fixed time sample.

    TOD are a large set of samples. Each sample is identified by the time at which
    it was taken and by the detector that took it. If all the detectors are
    synchronously sampled, these samples can be organized in a matrix D-by-T, where
    D is the number of detectors and T is the number of time samples. Following
    ``numpy.ndarray`` notation, we denote it as ``tod[row_index, col_index]``.

    Dense matrices (2-dimensional arrays) are stored in regular data
    structures: D chunks of memory stitched together, each made of T contiguous
    elements. 

    ``tod[3]`` gets the third row (the TOD of the third detector), ``tod[:, 3]`` is
    the third column. Both are views of the original memory allocations (no data
    copy nor movement) and both selections are quick: The first because 
    it is one contiguous chuck of memory at known location, the second because
    it is made of equally-spaced elements in memory. The first is much better than the
    second, but this is the best you can do if you want to access both many time samples
    for fixed detector and many detectors at fixed time. "Equally-spaced" may
    seem a detail to a human but makes a ton of difference for a CPU:
    Unpredictable memory access is a performance killer.

.. toggle-header::
 :header: We want to operate on the detector dimension as easily as we operate on the time dimension.

   Having the detectors as an axis of a 2-dimensional ``numpy.ndarray`` 
   (i.e.  one index of a TOD matrix) allow to perform easily operation that
   involve multiple detectors at fixed time.

   As mentioned above accessing all the detectors at the ``i`` th time sample is as
   easy as ``tod[:, i]``. Taking the time stream of the mean signal across the
   focal plane is simply ``tod.mean(axis=0)``, and taking the standard
   deviation is just as easy. In general, an extremely large number of
   operations can be expressed in terms of ``numpy`` functions and they can
   operate easily and efficiently over axes of multi-dimensional arrays.
   Suppose you have a cross-talk matrix ``C``, mixing the TODs accordingly is
   as easy as ``C @ tod``.
   
.. toggle-header::
 :header: We want to operate on the detector dimension efficiently

   The previous examples are not only easy to write, they are also (very likely)
   as fast as they can be (compared to different data layouts). This is
   particularly true for the last example. The matrix multiplication ``C @ tod``
   calls internally some highly optimized dense linear algebra routines.

   Even when there is not really communications between detectors involved,
   having data in a 2-dimensional array produces (arguably) cleaner code and
   sometimes faster code (never slower). Suppose you have a *list* of
   ``calibration_factors`` and a *list* of time streams ``tod``. You can apply the
   calibration factors with

   .. code-block:: python 

     for calibration_factor, det_tod in zip(calibration_factors, tod):
         det_tod *= calibration_factor

   but if ``calibration_factors`` is an *array* and ``tod`` a *matrix* you can
   achieve the same result with the easier and (typically) faster

   .. code-block:: python 

     tod *= calibration_factors[:, None]

Splitting the TOD matrix
------------------------
.. toggle-header::
 :header: When working on a laptop (or a single compute node) we can live in the ideal case discussed above
   
   We can benefit both from the API simplifications and the performance of
   the compiled codes that the numerical libraries typically call. Moreover,
   these libraries are often threaded (or easily threadable) and therefore we
   can in principle leverage on all the processors available.

.. toggle-header::
 :header: When working on clusters, we have to split this matrix into pieces.

   We resort to supercomputers either when we want more CPU power than a laptop
   or because the we need more memory to store our data. Both motivations apply
   to full-scale LiteBIRD simulations (4000 detectors, sampled at 30 GHz for 3
   years take approximately 15 TB). Therefore, we have to distribute the matrix
   and a compute node has access only to the portion of the matrix that is in its
   memory.

.. toggle-header::
 :header: We split the matrix into blocks that are stored as (sub-)matrices.

   At the present stage, there is no constraint on the shape of these blocks,
   they can also be a single row or a single column. The important thing is
   that, if the block spans more than a row, it is stored as a matrix (a
   2-dimensional array). This means that, for the time- and detector-chunk
   assigned to a compute node, all the benefits discussed in the previous
   section apply.

Choosing the blocks shape
-------------------------
The most convenient block shape depends on the application

.. toggle-header::
 :header: Some operations prefer to store an entire row

   For example, :math:`4\pi` beam convolution has a memory overhead of the order
   of the GB for each beam, which is in principle different for each detector.
   For this operation is therefore more convenient to hold a detector for the
   entire length of the mission.

.. toggle-header::
 :header: Some operations prefer to have an entire column

   For example, computing the common mode across the focal plane requires
   information from all the detectors at every time sample. This is trivial if
   the full detector column is in the memory, but it is very complicated if it
   is scattered across many compute nodes. Another example is cross-talks.
   Mixing the TOD of different detectors is trivial when everything is in
   memory, but otherwise it requires sending large messages across the network. Sending
   messages is not only a performance issue, it means that probably a custom
   algorithm has to be written (there are tons of shared-memory libraries but
   only a few distributed-memory libraries). This algorithm would require the
   MPI interface, which reduces drastically the number of people that can
   contribute to (or even understand) the code.

Since there is no one-size-fit-all solution, we keep general. 
The shape of the blocks will depend on the application and it will be also
possible to change it during the application (but it should be avoided as much
as possible).

Possible advantages of less general choices
-------------------------------------------

.. toggle-header::
 :header: Keeping the detectors independent in memory gives more flexibility but no substantial advantage in typical applications
   
   We realized that many collaborators (and codes) expected the TOD of each
   detector to be an independent vector. To our understanding, the main
   advantage is that it allows to easily add and remove detectors to
   an observation. However, we do not expect it to happen often in a given application.

.. toggle-header::
 :header: Assuming row-blocks seems natural to most people.

   Of course this is equivalent to the previous point and
   not having any matrix structure in the data. This assumption still suits
   several applications and for them it is more convenient to deal with a 
   length-T array rather than a 1-by-T matrix for API reasons
   (there is no memory nor performance advantage)
   However, it seems to us that the potential advantages of allowing non-trivial
   blocks largely outweighs the possible API advantages of imposing row-blocks.

Beyond TOD
----------
.. toggle-header::
 :header: Applying the same mindset to all the other detector quantities and properties, we store them in arrays that contain a given information for all the detectors
  
   Conceptually it may seem more natural to select a detector and have all its
   quantities as attributes. However, when we operate on the observations, we
   typically are interested only in one or few attributes of the detectors and
   we operate on all the detectors. Because of this pattern, it is more
   convenient from any computational point of view to stack a given attribute
   for every detector into an array. This is likely to save many for loops
   by exploiting ``numpy`` broadcasting rules in most of numerical operations,
   which results in both cleaner code and higher performance.

.. toggle-header::
 :header: Note that if the TOD matrix is chunked along the detector dimension, only the corresponding portion of the property array is detained in memory. 
  
   This implies that -- regardless if and how the TOD is distributed or not --
   ``obs.tod[i]`` and ``obs.wn_level[i]`` both refer to the same
   detector, ``obs.tod`` and ``obs.wn_level`` have the same length and
   ``obs.tod * obs.wn_level[:, None]`` is valid (and correct) operation.
   Compared to storying the full property array in every process, the main
   drawback is that, whenever the the block distribution of the
   TOD is changed, also the property arrays have to be redistributed. The pros
   are a higher memory efficiency and are more consistend serial/parallel
   experience for the user.
