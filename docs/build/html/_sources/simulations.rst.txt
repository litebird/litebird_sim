Simulations
===========

Describe how to use a :class:`litebird_sim.simulations.Simulation`
object.


Generation of reports
---------------------

This section should explain how reports can be generated, first from
the perspective of a library user, and then describing how developers
can generate plots for their own modules.

Here is an example, showing several advanced topics like mathematical
formulae, plots, and value substitution::

    import litebird_sim as lbs
    import matplotlib.pylab as plt

    sim = lbs.Simulation(name="My simulation", base_path="output")
    data_points = [0, 1, 2, 3]

    plt.plot(data_points)
    fig = plt.gcf()

    sim.append_to_report('''
    Here is a formula for $`f(x)`$:

    ```math
    f(x) = \sin x
    ```

    And here is a completely unrelated plot:

    ![](myplot.png)

    The data points have the following values:
    {% for sample in data_points %}
    - {{ sample }}
    {% endfor %}
    ''', figures=[(fig, "myplot.png")],
         data_points=data_points)

    sim.flush()


API reference
-------------

.. automodule:: litebird_sim.simulations
    :members:
    :undoc-members:
    :show-inheritance:
