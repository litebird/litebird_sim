# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs
import pathlib
from uuid import UUID


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
        start_time=1.0,
        duration_s=3600.0,
    )
    output_file = sim.write_healpix_map(filename="test.fits.gz", pixels=np.zeros(12))

    assert isinstance(output_file, pathlib.Path)
    assert output_file.exists()

    sim.append_to_report(
        """Here is a plot:

![](myplot.png)

And here are the data points:
{% for sample in data_points -%}
- {{ sample }}
{% endfor %}
 """,
        figures=[(MockPlot(), "myplot.png")],
        data_points=[0, 1, 2],
    )

    reference = """# My simulation

Lorem ipsum
The simulation starts at t0=1.0 and lasts 3600.0 seconds.

Here is a plot:

![](myplot.png)

And here are the data points:
- 0
- 1
- 2
"""

    print(sim.report)
    assert reference.strip() in sim.report.strip()

    sim.flush()


def test_imo_in_report(tmp_path):
    curpath = pathlib.Path(__file__).parent
    imo = lbs.Imo(flatfile_location=curpath / "mock_imo")

    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        name="My simulation",
        description="Lorem ipsum",
        imo=imo,
    )

    entity_uuid = UUID("dd32cb51-f7d5-4c03-bf47-766ce87dc3ba")
    _ = sim.imo.query(f"/entities/{entity_uuid}")

    quantity_uuid = UUID("e9916db9-a234-4921-adfd-6c3bb4f816e9")
    _ = sim.imo.query(f"/quantities/{quantity_uuid}")

    data_file_uuid = UUID("37bb70e4-29b2-4657-ba0b-4ccefbc5ae36")
    _ = sim.imo.query(f"/data_files/{data_file_uuid}")

    # This data file is an older version of 37bb70e4
    data_file_uuid = UUID("bd8e16eb-2e9d-46dd-a971-f446e953b9dc")
    _ = sim.imo.query(f"/data_files/{data_file_uuid}")

    html_file = sim.flush()
    assert isinstance(html_file, pathlib.Path)
    assert html_file.exists()
