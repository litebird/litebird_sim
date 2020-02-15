Simulations
===========

Describe how to use a :class:`litebird_sim.simulations.Simulation`
object.

Here is an example, just to show how Python code can be included in
RST files::

  import litebird_sim as ls
  import matplotlib.pylab as plt

  simul = ls.Simulation(
      base_path="/storage/mysimulations/adc-sim",
      name="Simulation of ADC non-linearities",
      use_mpi=False,
      description="""
  Simulate the acquisition of an ADC including non-linearities
      """,
  )

  # Create a figure
  plt.plot([1, 2, 4, 3])

  simul.append_to_report(
      markdown_text="""
  In this simulation I have computed the following values:

  ![](myplot.png)

  Nice!
  """,
      figures=[
          (plt.gcf(), "myplot.png"),
      ],
  )

  simul.flush()


API reference
-------------

.. automodule:: litebird_sim.simulations
    :members:
    :undoc-members:
    :show-inheritance:
