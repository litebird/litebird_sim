.. _random-numbers:

Random numbers
==============

The LiteBIRD Simulation Framework can be used to generate random numbers,
used for example for producing noise timelines. In order to do so,
a seed and a Random Number Generator (RNG) are used.

Seed
----

The ``random_seed`` is used to control the behavior of the RNG. The
seed can be ``None`` or an integer number. If you are **not**
interested in the reproducibility of your results, you can set
``random_seed`` to ``None``. However, this is not recommended, as if
you run many times a function or method that uses an RNG, e.g., for
producing noise timelines, then the outputs will be **different**, and
you will not be able to re-obtain these results again. If, instead,
you are interested in the reproducibility of your results, you can set
``random_seed`` to an integer number. With this choice, if you run a
function or method multiple times using an RNG, then the outputs will
be **the same**, and you will obtain these results again by re-running
your script, as the random numbers produced by the generator will be
the same.

How should you set the ``random_seed``? This parameter **must** be
passed to the constructor of a :class:`.Simulation` class. If you
create a :class:`.Simulation` instance without passing the seed, the
constructor will raise an error, and your script will fail. We
implemented this feature to avoid automatic settings of the seed and
unclear behavior of the generation of random numbers. If you run a
parallel script using MPI, you do not have to worry about setting a
different seed for different MPI ranks. The random number generator
ensures that the random numbers generated by the ranks will produce
orthogonal sequences. We want this behavior as we do not want
repetitions of, e.g., the same noise TOD if we split their computation
on different MPI ranks. For example, in this way, if you split the TOD
matrix of an :class:`.Observation` class by the time, you will not
encounter the same noise after the samples generated by a certain
rank; if you split the TOD matrix of an :class:`.Observation` class by
the detectors, each one will have a different noise timestream.

Regarding the reproducibility of the results in a parallel code, there
is an **important thing** to bear in mind. If you set the seed to an
integer number but run your script with a different number of MPI
ranks, the outputs will be **different**! Think about a noise time
stream of 4 samples. If you use 2 MPI ranks, then the first two
samples will be generated by one RNG (rank 0), while the last two
samples will be generated by a different RNG (rank 1). If you then run
the same script with the same seed but with 4 MPI ranks, each of the
samples will be generated by a different RNG, and only the first
sample will be the same for the two runs, as it is always the first
one generated by rank 0’s RNG.

The setting of the ``random_seed`` is as simple as this::

  sim = lbs.Simulation(
      base_path="/storage/output",
      start_time=astropy.time.Time("2020-02-01T10:30:00"),
      duration_s=3600.0,
      name="My noise simulation",
      description="I want to generate some noise and be able to reproduce my results",
      random_seed=12345, # Here the seed for the random number generator is set
  )

The :class:`.Simulation` constructor runs the
:meth:`.Simulation.init_random` method, which builds the RNG and seeds
it with ``random_seed``. You can then use this RNG by calling
``sim.random``. There is also the possibility to change the random
seed after creating a :class:`.Simulation` instance. You can achieve
this by calling :meth:`.Simulation.init_random`::

  sim = lbs.Simulation(random_seed=12345)
  [...]
  sim.init_random(random_seed=42)

Random number generators
------------------------

The random number generator used by the :meth:`.Simulation.init_random`
method of the :class:`.Simulation` class is
`PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_.
After creating this RNG by calling :meth:`.Simulation.init_random`
(directly or from the :class:`.Simulation` constructor), you can use it
via `sim.random`::

  sim = lbs.Simulation(random_seed=12345)
  [...]
  sim.add_noise(noise_type='white', random=sim.random)

You can also use your own RNG with the functions and methods of
``litebird_sim``::

  sim = lbs.Simulation(random_seed=12345)
  [...]
  my_rng = ... # New RNG definition
  sim.add_noise(noise_type='white', random=my_rng)

You should just make sure that your custom RNG implements the
``normal`` method, so it can be used for white noise generation.

