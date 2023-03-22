# -*- encoding: utf-8 -*-

import os
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import numpy as np
import litebird_sim as lbs
import pathlib
from uuid import UUID

import astropy


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

[TOC]



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


def test_parameter_dict(tmp_path):
    from datetime import date

    sim = lbs.Simulation(
        parameters={
            "general": {
                "a": 10,
                "b": 20.0,
                "c": False,
                "subtable": {"d": date(2020, 7, 1), "e": "Hello, world!"},
            }
        }
    )

    assert not sim.parameter_file
    assert isinstance(sim.parameters, dict)

    assert "general" in sim.parameters
    assert sim.parameters["general"]["a"] == 10
    assert sim.parameters["general"]["b"] == 20.0
    assert not sim.parameters["general"]["c"]

    assert "subtable" in sim.parameters["general"]
    assert sim.parameters["general"]["subtable"]["d"] == date(2020, 7, 1)
    assert sim.parameters["general"]["subtable"]["e"] == "Hello, world!"

    try:
        sim = lbs.Simulation(parameter_file="dummy", parameters={"a": 10})
        assert False, "Simulation object should have asserted"
    except AssertionError:
        pass


def test_parameter_file():
    from datetime import date

    with NamedTemporaryFile(mode="wt", delete=False) as conf_file:
        conf_file_name = conf_file.name
        conf_file.write(
            """# test_parameter_file
[simulation]
start_time = "2020-01-01T00:00:00"
duration_s = 11.0
description = "Dummy description"

[general]
a = 10
b = 20.0
c = false

[general.subtable]
d = 2020-07-01
e = "Hello, world!"
"""
        )

    with TemporaryDirectory() as tmpdirname:
        sim = lbs.Simulation(base_path=tmpdirname, parameter_file=conf_file_name)

        assert isinstance(sim.parameter_file, pathlib.Path)
        assert isinstance(sim.parameters, dict)

        assert "simulation" in sim.parameters
        assert isinstance(sim.start_time, astropy.time.Time)
        assert sim.duration_s == 11.0
        assert sim.description == "Dummy description"

        assert "general" in sim.parameters
        assert sim.parameters["general"]["a"] == 10
        assert sim.parameters["general"]["b"] == 20.0
        assert not sim.parameters["general"]["c"]

        assert "subtable" in sim.parameters["general"]
        assert sim.parameters["general"]["subtable"]["d"] == date(2020, 7, 1)
        assert sim.parameters["general"]["subtable"]["e"] == "Hello, world!"

    # Check that the code does not complain if the output directory is
    # the same as the one containing the parameter file

    sim = lbs.Simulation(
        base_path=Path(conf_file_name).parent, parameter_file=conf_file_name
    )

    os.unlink(conf_file_name)


def test_duration_units_in_parameter_file():
    with NamedTemporaryFile(mode="wt", delete=False) as conf_file:
        conf_file_name = conf_file.name
        conf_file.write(
            """# test_duration_units_in_parameter_file
[simulation]
start_time = "2020-01-01T00:00:00"
duration_s = "1 day"
"""
        )

    with TemporaryDirectory() as tmpdirname:
        sim = lbs.Simulation(base_path=tmpdirname, parameter_file=conf_file_name)

        assert "simulation" in sim.parameters
        assert isinstance(sim.start_time, astropy.time.Time)
        assert sim.duration_s == 86400.0


def test_distribute_observation(tmp_path):
    for dtype in (np.float16, np.float32, np.float64, np.float128):
        sim = lbs.Simulation(
            base_path=tmp_path / "simulation_dir", start_time=1.0, duration_s=11.0
        )
        det = lbs.DetectorInfo("dummy", sampling_rate_hz=15)
        obs_list = sim.create_observations(
            detectors=[det], num_of_obs_per_detector=5, dtype_tod=dtype
        )

        assert len(obs_list) == 5
        assert int(obs_list[-1].get_times()[-1] - obs_list[0].get_times()[0]) == 10
        assert (
            sum([o.n_samples for o in obs_list])
            == sim.duration_s * det.sampling_rate_hz
        )


def test_distribute_observation_many_tods(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir", start_time=1.0, duration_s=11.0
    )
    det = lbs.DetectorInfo("dummy", sampling_rate_hz=15)
    sim.create_observations(
        detectors=[det],
        num_of_obs_per_detector=5,
        tods=[
            lbs.TodDescription(name="tod1", dtype=np.float32, description="TOD 1"),
            lbs.TodDescription(name="tod2", dtype=np.float64, description="TOD 2"),
        ],
    )

    assert sim.get_tod_names() == ["tod1", "tod2"]
    assert sim.get_tod_descriptions() == ["TOD 1", "TOD 2"]
    assert sim.get_tod_dtypes() == [np.float32, np.float64]

    for cur_obs in sim.observations:
        assert "tod1" in dir(cur_obs)
        assert "tod2" in dir(cur_obs)

        assert cur_obs.tod1.shape == cur_obs.tod2.shape
        assert cur_obs.tod1.dtype == np.float32
        assert cur_obs.tod2.dtype == np.float64

    assert len(sim.observations) == 5
    assert (
        int(sim.observations[-1].get_times()[-1] - sim.observations[0].get_times()[0])
        == 10
    )
    assert (
        sum([o.n_samples for o in sim.observations])
        == sim.duration_s * det.sampling_rate_hz
    )


def test_distribute_observation_astropy(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=astropy.time.Time("2020-01-01T00:00:00"),
        duration_s=11.0,
    )
    det = lbs.DetectorInfo("dummy", sampling_rate_hz=15)
    obs_list = sim.create_observations(detectors=[det], num_of_obs_per_detector=5)

    assert len(obs_list) == 5
    assert int(obs_list[-1].get_times()[-1] - obs_list[0].get_times()[0]) == 10
    assert sum([o.n_samples for o in obs_list]) == sim.duration_s * det.sampling_rate_hz


def test_describe_distribution(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=40.0,
    )
    det = lbs.DetectorInfo("dummy", sampling_rate_hz=10.0)

    sim.create_observations(
        detectors=[det],
        num_of_obs_per_detector=4,
        tods=[
            lbs.TodDescription(name="tod", dtype="float32", description="Signal"),
            lbs.TodDescription(
                name="fg_tod", dtype="float64", description="Foregrounds"
            ),
            lbs.TodDescription(
                name="dipole_tod", dtype="float32", description="Dipole"
            ),
        ],
    )

    for cur_obs in sim.observations:
        assert "tod" in dir(cur_obs)
        assert "fg_tod" in dir(cur_obs)
        assert "dipole_tod" in dir(cur_obs)

    descr = sim.describe_mpi_distribution()

    assert len(descr.detectors) == 1
    assert len(descr.mpi_processes) == lbs.MPI_COMM_WORLD.size

    for mpi_proc in descr.mpi_processes:
        for obs in mpi_proc.observations:
            assert obs.det_names == ["dummy"]
            assert obs.tod_names == ["tod", "fg_tod", "dipole_tod"]
            assert obs.tod_shape == (1, 100)
            assert obs.tod_dtype == ["float32", "float64", "float32"]
