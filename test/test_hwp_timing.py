"""HWP phase must be independent of observation and MPI boundaries."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from astropy.time import Time

import litebird_sim as lbs
from litebird_sim.io import read_one_observation
from litebird_sim.observation_utilities import _get_hwp_angle, apply_hwp_to_obs


def _elapsed_s(time, epoch):
    delta = time - epoch
    return delta.to_value("s") if hasattr(delta, "to_value") else delta


def _assert_phase(actual, expected, atol=1e-10):
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (actual - expected))), 0.0, atol=atol
    )


def _make_simulation(
    tmp_path, start, count, nonideal=False, split_list=False, duration_s=12.25
):
    comm = lbs.MPI_COMM_WORLD
    sim = lbs.Simulation(
        base_path=tmp_path,
        start_time=start,
        duration_s=duration_s,
        random_seed=123,
    )
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name="000_test",
                orient="Q",
                pol="T",
                sampling_rate_hz=4.0,
                bandcenter_ghz=100.0,
                bandwidth_ghz=20.0,
            )
        ],
        num_of_obs_per_detector=count,
        split_list_over_processes=split_list,
        n_blocks_time=1 if split_list else comm.size,
    )
    if nonideal:
        hwp = lbs.NonIdealHWP(
            ang_speed_radpsec=0.73,
            start_angle_rad=0.37,
            harmonic_expansion=False,
            calculus=lbs.HWPFormalism.MUELLER,
        )
        for obs in sim.observations:
            obs.mueller_hwp = np.array(
                [{"0f": np.diag([1.0, 1.0, -1.0, -1.0])}], dtype=object
            )
    else:
        hwp = lbs.IdealHWP(ang_speed_radpsec=0.73, start_angle_rad=0.37)
    sim.set_hwp(hwp)
    sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=0.0))
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=0.0
        ),
        delta_time_s=1.0,
        append_to_report=False,
    )
    sim.prepare_pointings(append_to_report=False)
    return sim


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("nonideal", [False, True])
@pytest.mark.parametrize("scale", [None, "utc", "tdb"])
def test_hwp_phase_across_observations(tmp_path, count, nonideal, scale):
    start = Time("2026-01-01T00:00:00", scale=scale) if scale else 100.0
    sim = _make_simulation(tmp_path, start, count, nonideal)
    for obs in sim.observations:
        assert obs.start_time_simulation == sim.start_time
        elapsed = _elapsed_s(obs.start_time_local, sim.start_time)
        expected = 0.37 + 0.73 * (elapsed + np.arange(obs.n_samples_local) / 4.0)
        _assert_phase(obs.get_hwp_angle(), expected)
        _assert_phase(obs.get_pointings(0)[1], expected)
        _assert_phase(_get_hwp_angle(obs, hwp=sim.hwp), expected)
        # The deprecated API adds twice the mechanical angle to orientation.
        pointings = np.zeros((1, obs.n_samples_local, 3))
        with pytest.warns(DeprecationWarning):
            apply_hwp_to_obs(obs, sim.hwp, pointings)
        _assert_phase(pointings[0, :, 2], 2 * expected)
        obs.precompute_pointings()
        _assert_phase(obs.hwp_angle, expected)


def test_hwp_phase_distributed_observation_list(tmp_path):
    sim = _make_simulation(
        tmp_path, 100.0, 2 * lbs.MPI_COMM_WORLD.size + 1, split_list=True
    )
    for obs in sim.observations:
        expected = 0.37 + 0.73 * (
            obs.start_time_local - sim.start_time + np.arange(obs.n_samples_local) / 4.0
        )
        _assert_phase(obs.get_hwp_angle(), expected)


@pytest.mark.parametrize("use_astropy", [False, True])
def test_hwp_later_observation_roundtrip(tmp_path, use_astropy):
    start = (
        Time("2026-01-01T00:00:00.123456789", scale="tdb") if use_astropy else 100.125
    )
    # 17 samples per observation, so two MPI ranks get different local lengths.
    sim = _make_simulation(tmp_path, start, 3, duration_s=12.75)
    original = sim.observations[-1]
    files = lbs.write_list_of_observations(
        original,
        path=tmp_path,
        collective_mpi_call=False,
        file_name_mask="later_{mpi_rank}.h5",
    )
    loaded = read_one_observation(files[0])
    assert loaded is not None
    assert loaded.start_time_simulation == start
    assert loaded.start_time_observation == original.start_time_observation
    assert loaded.end_time_observation == original.end_time_observation
    assert loaded.start_time_local == original.start_time_local
    assert loaded.n_samples_observation == original.n_samples_observation
    assert loaded.n_samples_local == original.n_samples_local
    assert loaded.start_sample_observation == original.start_sample_observation
    assert loaded._get_tod_shape(1, 1) == loaded.tod.shape
    np.testing.assert_allclose(
        loaded.get_times(normalize=True), original.get_times(normalize=True)
    )
    if use_astropy:
        assert loaded.start_time_simulation.scale == start.scale
        assert loaded.start_time_local.scale == original.start_time_local.scale
    elapsed = _elapsed_s(loaded.start_time_local, start)
    expected = 0.37 + 0.73 * (elapsed + np.arange(loaded.n_samples_local) / 4.0)
    _assert_phase(loaded.get_hwp_angle(), expected)
    loaded_pointings, loaded_hwp = loaded.get_pointings(0)
    _assert_phase(loaded_hwp, expected)
    np.testing.assert_allclose(
        loaded_pointings, original.get_pointings(0)[0], atol=1e-10
    )
    # A second save/read must preserve the fragment's original sample interval.
    second_files = lbs.write_list_of_observations(
        loaded,
        path=tmp_path,
        collective_mpi_call=False,
        file_name_mask="later_again_{mpi_rank}.h5",
    )
    reloaded = read_one_observation(second_files[0])
    assert reloaded is not None
    assert reloaded.n_samples_observation == original.n_samples_observation
    assert reloaded.n_samples_local == original.n_samples_local
    assert reloaded.start_sample_observation == original.start_sample_observation
    np.testing.assert_allclose(reloaded.get_times(), original.get_times())
    np.testing.assert_array_equal(reloaded.tod, original.tod)


def test_standalone_observation_time_reference():
    obs = lbs.Observation(
        detectors=1,
        n_samples_observation=10,
        start_time_observation=25.0,
        sampling_rate_hz=4.0,
    )
    assert obs.start_time_simulation == 25.0
    assert obs.start_time_observation == 25.0
    assert obs.start_time_local == 25.0
    assert not hasattr(obs, "start_time")
    assert obs.end_time_observation == 27.5
    assert not hasattr(obs, "start_time_global")
    assert not hasattr(obs, "end_time_global")
    assert obs.n_samples_observation == obs.n_samples_local == 10
    assert obs.start_sample_observation == 0
    assert not hasattr(obs, "n_samples_global")
    assert not hasattr(obs, "n_samples")
    assert not hasattr(obs, "start_sample")


@pytest.mark.parametrize("use_astropy", [False, True])
def test_wedge_phase_across_observations(tmp_path, use_astropy):
    start = Time("2026-01-01T00:00:00", scale="tdb") if use_astropy else 100.0
    # The spacecraft grid is deliberately different from the detector grid.
    reference = _make_simulation(tmp_path / "reference", start, 1)
    split = _make_simulation(tmp_path / "split", start, 3)
    reference_obs = reference.observations[0]
    reference_sys = lbs.PointingSys(reference, reference_obs, reference.detectors)
    reference_sys.hwp.add_hwp_rot_disturb(0.01, 0.73, tilt_phase_rad=0.29)
    reference.prepare_pointings(append_to_report=False)
    # Use an independent analytic phase for the disturbance, including alpha_0.
    for sim in (reference, split):
        scanning_start = sim.spin2ecliptic_quats.start_time
        for obs in sim.observations:
            if sim is split:
                syst = lbs.PointingSys(sim, obs, sim.detectors)
                syst.hwp.add_hwp_rot_disturb(0.01, 0.73, tilt_phase_rad=0.29)
            assert sim.spin2ecliptic_quats.start_time == scanning_start
            elapsed = _elapsed_s(obs.start_time_local, start)
            phase = (
                0.37 + 0.73 * (elapsed + np.arange(obs.n_samples_local) / 4.0) + 0.29
            )
            expected_quats = lbs.RotQuaternion(
                quats=lbs.quat_rotation_x_brdcast(0.01 * np.cos(phase)),
                start_time=obs.start_time_local,
                sampling_rate_hz=4.0,
            ) * lbs.RotQuaternion(
                quats=lbs.quat_rotation_y_brdcast(0.01 * np.sin(phase)),
                start_time=obs.start_time_local,
                sampling_rate_hz=4.0,
            )
            actual = obs.quat[0].slerp(obs.start_time_local, 4.0, obs.n_samples_local)
            np.testing.assert_allclose(actual, expected_quats.quats, atol=1e-10)
            sim.prepare_pointings(append_to_report=False)
            pointings, hwp = obs.get_pointings(0)
            bore_samples = sim.observations[
                0
            ].pointing_provider.bore2ecliptic_quats.slerp(
                obs.start_time_local, 4.0, obs.n_samples_local
            )
            expected_sky_quats = (
                lbs.RotQuaternion(
                    quats=bore_samples,
                    start_time=obs.start_time_local,
                    sampling_rate_hz=4.0,
                )
                * expected_quats
            )
            expected_pointings = np.empty_like(pointings)
            lbs.all_compute_pointing_and_orientation(
                expected_pointings, expected_sky_quats.quats
            )
            np.testing.assert_allclose(pointings, expected_pointings, atol=1e-10)
            _assert_phase(hwp + 0.29, phase)


def test_hwp_phase_one_year_after_epoch():
    """Bound phase error with a high precision oracle, without simulating a year."""
    elapsed_s = 365.25 * 86400.0 + 0.125
    omega = 2 * np.pi * 0.65
    obs = lbs.Observation(
        detectors=[{"name": "det", "quat": lbs.RotQuaternion()}],
        n_samples_observation=4096,
        start_time_observation=elapsed_s,
        start_time_simulation=0.0,
        sampling_rate_hz=32.0,
        comm=None,
    )
    obs.prepare_pointings(
        lbs.InstrumentInfo(),
        lbs.RotQuaternion(),
        hwp=lbs.IdealHWP(omega, start_angle_rad=0.37),
    )
    with localcontext() as context:
        context.prec = 60
        two_pi = 2 * Decimal(
            "3.14159265358979323846264338327950288419716939937510582097494"
        )
        expected = np.array(
            [
                float(
                    (
                        Decimal.from_float(0.37)
                        + Decimal.from_float(omega)
                        * (Decimal.from_float(elapsed_s) + Decimal(i) / 32)
                    )
                    % two_pi
                )
                for i in range(obs.n_samples_local)
            ]
        )
    # ~1.3e8 rad accumulated phase: float64 argument rounding dominates.
    _assert_phase(obs.get_hwp_angle(), expected, atol=5e-8)
