# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs
import pathlib


class MockPlot:
    def savefig(*args, **kwargs):
        pass


def test_healpix_map_write(tmp_path):
    sim = lbs.Simulation(base_path=tmp_path / "simulation_dir")
    output_file = sim.write_healpix_map(filename="test.fits.gz", pixels=np.zeros(12))

    assert isinstance(output_file, pathlib.Path)
    assert output_file.exists()

    sim.append_to_report(
        """Here is a plot:

 ![](myplot.png)
 """,
        [(MockPlot(), "myplot.png")],
    )

    sim.flush()


def test_markdown_report(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        name="My simulation",
        description="Lorem ipsum",
    )
    output_file = sim.write_healpix_map(filename="test.fits.gz", pixels=np.zeros(12))

    assert isinstance(output_file, pathlib.Path)
    assert output_file.exists()

    sim.append_to_report(
        """Here is a plot:

 ![](myplot.png)

And here are the data points:
{% for sample in data_points %}
- {{ sample }}
{% endfor %}
 """,
        figures=[(MockPlot(), "myplot.png")],
        data_points=[0, 1, 2],
    )

    reference = """# My simulation

Lorem ipsum

Here is a plot:

 ![](myplot.png)

And here are the data points:

- 0

- 1

- 2
"""

    print(sim.report)
    assert sim.report.strip() == reference.strip()

    sim.flush()
