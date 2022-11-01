# -*- encoding: utf-8 -*-
from typing import Any, List, Union

import astropy
import numpy as np
import astropy.units as u

import litebird_sim
import litebird_sim as lbs
from litebird_sim.madam import _sort_obs_per_det, _ObsInMpiProcess
from litebird_sim.simulations import (
    MpiDistributionDescr,
    MpiProcessDescr,
    MpiObservationDescr,
)


def test_sort_obs_per_det():
    # These are the values of "start_time" for each observation in "distribution":
    #
    #            Obs #0   Obs #1
    #  MPI #0     2.0      0.0
    #  MPI #1     1.0      3.0
    #
    # Obviously, the order must be:
    # 1. MPI #0, Obs #1 (start_time = 0.0)
    # 2. MPI #1, Obs #0 (start_time = 1.0)
    # 3. MPI #0, Obs #0 (start_time = 2.0)
    # 4. MPI #1, Obs #1 (start_time = 3.0)

    distribution = MpiDistributionDescr(
        num_of_observations=4,
        detectors=[lbs.DetectorInfo("A")],
        mpi_processes=[
            MpiProcessDescr(
                mpi_rank=0,
                observations=[
                    MpiObservationDescr(
                        det_names=["A"],
                        tod_shape=(0, 0),
                        tod_dtype="float32",
                        start_time=2.0,
                        duration_s=1.0,
                        num_of_samples=1,
                        num_of_detectors=1,
                    ),
                    MpiObservationDescr(
                        det_names=["A"],
                        tod_shape=(0, 0),
                        tod_dtype="float32",
                        start_time=0.0,
                        duration_s=1.0,
                        num_of_samples=1,
                        num_of_detectors=1,
                    ),
                ],
            ),
            MpiProcessDescr(
                mpi_rank=1,
                observations=[
                    MpiObservationDescr(
                        det_names=["A"],
                        tod_shape=(0, 0),
                        tod_dtype="float32",
                        start_time=1.0,
                        duration_s=1.0,
                        num_of_samples=1,
                        num_of_detectors=1,
                    ),
                    MpiObservationDescr(
                        det_names=["A"],
                        tod_shape=(0, 0),
                        tod_dtype="float32",
                        start_time=3.0,
                        duration_s=1.0,
                        num_of_samples=1,
                        num_of_detectors=1,
                    ),
                ],
            ),
        ],
    )

    rank0_list = _sort_obs_per_det(distribution=distribution, detector="A", mpi_rank=0)
    rank1_list = _sort_obs_per_det(distribution=distribution, detector="A", mpi_rank=1)

    assert len(rank0_list) == 2
    assert len(rank1_list) == 2

    # 1. MPI #0, Obs #1 (start_time = 0.0)
    assert rank0_list[0].obs_local_idx == 1
    assert rank0_list[0].mpi_rank == 0
    assert rank0_list[0].obs_global_idx == 0

    # 2. MPI #1, Obs #0 (start_time = 1.0)
    assert rank1_list[0].obs_local_idx == 0
    assert rank1_list[0].mpi_rank == 1
    assert rank1_list[0].obs_global_idx == 1

    # 3. MPI #0, Obs #0 (start_time = 2.0)
    assert rank0_list[1].obs_local_idx == 0
    assert rank0_list[1].mpi_rank == 0
    assert rank0_list[1].obs_global_idx == 2

    # 4. MPI #1, Obs #1 (start_time = 3.0)
    assert rank1_list[1].obs_local_idx == 1
    assert rank1_list[1].mpi_rank == 1
    assert rank1_list[1].obs_global_idx == 3


def is_sorted(x: List[Any], key=lambda x: x) -> bool:
    return all([key(x[i]) <= key(x[i + 1]) for i in range(len(x) - 1)])


def get_key_from_fits(filename: str, key: str) -> Any:
    from astropy.io import fits

    with fits.open(filename) as hdul:
        # We limit ourselves to HDU#1, which is the first tabular HDU in
        # Madam FITS files
        return hdul[1].header[key]


def _num_of_obs_per_detector(descr: lbs.MpiDistributionDescr, det_name: str) -> int:
    return sum(
        [
            1
            for mpi_proc in descr.mpi_processes
            for obs in mpi_proc.observations
            if det_name in obs.det_names
        ]
    )


def run_test_on_madam(
    tmp_path,
    n_blocks_det: int,
    n_blocks_time: int,
    start_time: Union[float, astropy.time.Time],
):
    sim = lbs.Simulation(
        base_path=tmp_path / "destriper_output",
        start_time=start_time,
        duration_s=86400.0,
        mpi_comm=lbs.MPI_COMM_WORLD,
    )

    sim.generate_spin2ecl_quaternions(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # seconds
            precession_rate_hz=1.0 / (4 * u.day).to("s").value,
        )
    )
    instr = lbs.InstrumentInfo(name="core", spin_boresight_angle_rad=np.deg2rad(65))
    detectors = [
        lbs.DetectorInfo(name="0A", sampling_rate_hz=10),
        lbs.DetectorInfo(
            name="0B", sampling_rate_hz=10, quat=lbs.quat_rotation_z(np.pi / 2)
        ),
    ]

    sim.create_observations(
        detectors=detectors,
        dtype_tod=np.float64,
        split_list_over_processes=False,
        num_of_obs_per_detector=2,
        n_blocks_det=n_blocks_det,
        n_blocks_time=n_blocks_time,
    )

    distribution = sim.describe_mpi_distribution()
    assert distribution is not None

    lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
    )

    for cur_obs in sim.observations:
        cur_obs.tod[:] = float(lbs.MPI_COMM_WORLD.rank)

    params = lbs.DestriperParameters(
        nside=16,
        nnz=3,
        baseline_length_s=100,
        iter_max=10,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
        return_npp=True,
        return_invnpp=True,
        return_rcond=True,
    )

    # Just check that all the files are saved without errors/exceptions:
    # to verify that the input files are ok, we should download and install
    # Madam…
    result = lbs.save_simulation_for_madam(sim, detectors=detectors, params=params)

    if lbs.MPI_COMM_WORLD.rank != 0:
        # The value of "result" is meaningful only for MPI process #0, so there
        # is no point in continuing
        return

    assert len(result["detectors"]) == 2

    for det_name in ("0A", "0B"):
        obs_per_detector = _num_of_obs_per_detector(
            descr=distribution, det_name=det_name
        )

        tod_files = sorted(
            [x["file_name"] for x in result["tod_files"] if x["det_name"] == det_name],
            key=lambda x: x.name,
        )
        assert is_sorted(
            [get_key_from_fits(filename=x, key="TIME0") for x in tod_files]
        )
        assert len(tod_files) == obs_per_detector

        pointing_files = sorted(
            [
                x["file_name"]
                for x in result["pointing_files"]
                if x["det_name"] == det_name
            ],
            key=lambda x: x.name,
        )
        assert is_sorted(
            [get_key_from_fits(filename=x, key="TIME0") for x in pointing_files]
        )
        assert len(pointing_files) == obs_per_detector


def run_mpi_test_on_madam(tmp_path, start_time):
    run_test_on_madam(
        tmp_path,
        n_blocks_det=lbs.MPI_COMM_WORLD.size,
        n_blocks_time=1,
        start_time=start_time,
    )
    run_test_on_madam(
        tmp_path,
        n_blocks_det=1,
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
        start_time=start_time,
    )


def test_madam(tmp_path):
    run_mpi_test_on_madam(tmp_path, start_time=0.0)


def test_madam_astropy_time(tmp_path):
    run_mpi_test_on_madam(tmp_path, start_time=astropy.time.Time("2030-01-01 00:00:00"))
