"""Tests of ObservationPointings, the single way to get the pointings of the
detectors of an observation and its HWP angle"""

import numpy as np
import pytest
from ducc0.healpix import Healpix_Base

import litebird_sim as lbs
from litebird_sim import CoordinateSystem
from litebird_sim.observation_utilities import (
    ObservationPointings,
    observation_pointings,
)

N_DETS = 3


def _simulation(hwp: bool, precompute: bool = False):
    sim = lbs.Simulation(start_time=0.0, duration_s=300.0, random_seed=1)
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            precession_rate_hz=1 / 3600,
            spin_rate_hz=1 / 60,
        )
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=np.deg2rad(50.0))
    )
    dets = [
        lbs.DetectorInfo(
            name=f"d{i}",
            sampling_rate_hz=2.0,
            pol_angle_rad=0.4 * i,
            quat=lbs.RotQuaternion(
                np.array(lbs.quat_rotation_y(np.deg2rad(1.0 + 2.0 * i)))
            ),
        )
        for i in range(N_DETS)
    ]
    sim.create_observations(detectors=dets)
    if hwp:
        sim.set_hwp(lbs.IdealHWP(ang_speed_radpsec=2.0))
    sim.prepare_pointings()
    if precompute:
        sim.precompute_pointings()
    return sim


def _three_ways(hwp: bool, **kwargs):
    """The same pointings computed on the fly, precomputed, and passed by the
    caller as an array"""
    on_the_fly = _simulation(hwp)
    precomputed = _simulation(hwp, precompute=True)
    external = _simulation(hwp)
    ecliptic, _ = external.observations[0].get_pointings("all")
    return [
        ObservationPointings(on_the_fly.observations[0], **kwargs),
        ObservationPointings(precomputed.observations[0], **kwargs),
        ObservationPointings(external.observations[0], pointings=ecliptic, **kwargs),
    ]


def _assert_same_angles(actual, desired, atol=1e-9):
    diff = np.angle(np.exp(1j * (actual - desired)))  # Wrapped to (-π, π]
    np.testing.assert_allclose(diff, 0.0, rtol=0, atol=atol)


@pytest.mark.parametrize("hwp", [False, True])
@pytest.mark.parametrize(
    "coordinates", [CoordinateSystem.Ecliptic, CoordinateSystem.Galactic]
)
def test_parity_of_the_three_ways(hwp, coordinates):
    ways = _three_ways(hwp, output_coordinate_system=coordinates)
    reference = ways[0]
    for other in ways[1:]:
        for idet in range(N_DETS):
            # The on-the-fly pointings rotate the quaternions, the others the
            # angles: they agree up to rounding, and the angles may differ by 2π
            _assert_same_angles(other.detector(idet), reference.detector(idet))
        if hwp:
            np.testing.assert_allclose(other.hwp_angle, reference.hwp_angle)
        else:
            assert other.hwp_angle is None and reference.hwp_angle is None


def test_parity_with_centering():
    # In Ecliptic coordinates no rotation is applied, so the centered pointings
    # must be identical (a rounding difference could move a sample to the
    # nearby pixel)
    ways = _three_ways(
        False, output_coordinate_system=CoordinateSystem.Ecliptic, nside_centering=8
    )
    hpx_centers = ways[0].detector(0)[:, 0:2]
    for other in ways[1:]:
        np.testing.assert_array_equal(other.detector(0)[:, 0:2], hpx_centers)
    # ψ is not centered
    raw = ObservationPointings(_simulation(False).observations[0]).detector(0)
    np.testing.assert_array_equal(ways[0].detector(0)[:, 2], raw[:, 2])
    assert not np.array_equal(hpx_centers, raw[:, 0:2])


def test_detector_index_forms():
    for ptg in _three_ways(False):
        n_samples = ptg.observation.n_samples_process
        assert ptg.detector(1).shape == (n_samples, 3)
        both = ptg.detector([0, 2])
        assert both.shape == (2, n_samples, 3)
        np.testing.assert_allclose(both[1], ptg.detector(2), rtol=0, atol=1e-12)
        assert ptg.detector("all").shape == (N_DETS, n_samples, 3)


def test_no_writes_to_the_observation():
    sim = _simulation(False)
    obs = sim.observations[0]
    before = dict(vars(obs))
    ptg = ObservationPointings(obs, hwp=lbs.IdealHWP(ang_speed_radpsec=1.0))
    ptg.detector("all")
    assert ptg.hwp_angle is not None
    assert vars(obs).keys() == before.keys()
    assert obs.has_hwp is False


def test_hwp_precedence_and_cache():
    sim = _simulation(True)
    obs = sim.observations[0]

    own = ObservationPointings(obs)
    np.testing.assert_allclose(own.hwp_angle, obs.get_hwp_angle())
    assert own.hwp_angle is own.hwp_angle  # computed once

    # An external HWP overrides the one of the observation
    external = ObservationPointings(obs, hwp=lbs.IdealHWP(ang_speed_radpsec=0.5))
    assert not np.allclose(external.hwp_angle, own.hwp_angle)

    assert ObservationPointings(_simulation(False).observations[0]).hwp_angle is None


def test_errors_at_construction():
    obs = _simulation(False).observations[0]
    with pytest.raises(AssertionError, match="shape"):
        ObservationPointings(obs, pointings=np.zeros((N_DETS, 5, 3)))

    obs.mueller_hwp[0] = {"0f": np.eye(3), "2f": np.zeros((3, 3)), "4f": np.eye(3)}
    with pytest.raises(AssertionError, match="mueller_hwp"):
        ObservationPointings(obs)


def test_observation_pointings_on_lists():
    sims = [_simulation(False), _simulation(False)]
    obs_list = [s.observations[0] for s in sims]

    pairs = list(observation_pointings(obs_list))
    assert [p[0] for p in pairs] == obs_list
    assert all(isinstance(p[1], ObservationPointings) for p in pairs)

    single = list(observation_pointings(obs_list[0]))
    assert len(single) == 1 and single[0][0] is obs_list[0]

    ptg = [o.get_pointings("all")[0] for o in obs_list]
    with pytest.raises(AssertionError, match="elements"):
        list(observation_pointings(obs_list, pointings=ptg[:1]))


def test_coordinate_system_override():
    for ptg in _three_ways(False, output_coordinate_system=CoordinateSystem.Ecliptic):
        galactic = ObservationPointings(
            ptg.observation,
            pointings=ptg._ecliptic_array
            if ptg._ecliptic_array is not None
            and not hasattr(ptg.observation, "pointing_matrix")
            else None,
            output_coordinate_system=CoordinateSystem.Galactic,
        )
        _assert_same_angles(
            ptg.detector(1, output_coordinate_system=CoordinateSystem.Galactic),
            galactic.detector(1),
        )
        assert ptg.output_coordinate_system == CoordinateSystem.Ecliptic


@pytest.mark.parametrize("hwp", [False, True])
@pytest.mark.parametrize(
    "coordinates", [CoordinateSystem.Ecliptic, CoordinateSystem.Galactic]
)
def test_pixel_index_and_pol_angle(hwp, coordinates):
    nside = 16
    hpx = Healpix_Base(nside, "RING")
    ways = _three_ways(hwp)
    for ptg in ways:
        obs = ptg.observation
        for idet in range(N_DETS):
            angles = ptg.detector(idet, output_coordinate_system=coordinates)

            pixidx = ptg.pixel_index(idet, nside, output_coordinate_system=coordinates)
            assert pixidx.dtype == np.int32
            np.testing.assert_array_equal(pixidx, hpx.ang2pix(angles[:, 0:2]))

            if hwp:
                expected = 2 * ptg.hwp_angle - obs.pol_angle_rad[idet] + angles[:, 2]
            else:
                expected = obs.pol_angle_rad[idet] + angles[:, 2]
            pol_angle = ptg.pol_angle(idet, output_coordinate_system=coordinates)
            np.testing.assert_allclose(pol_angle, expected, rtol=0, atol=1e-12)

            both = ptg.pixel_index_and_pol_angle(
                idet, nside, output_coordinate_system=coordinates
            )
            np.testing.assert_array_equal(both[0], pixidx)
            np.testing.assert_array_equal(both[1], pol_angle)

    # The three ways agree (ψ up to 2π)
    for other in ways[1:]:
        for idet in range(N_DETS):
            np.testing.assert_array_equal(
                other.pixel_index(idet, nside, output_coordinate_system=coordinates),
                ways[0].pixel_index(idet, nside, output_coordinate_system=coordinates),
            )
            _assert_same_angles(
                other.pol_angle(idet, output_coordinate_system=coordinates),
                ways[0].pol_angle(idet, output_coordinate_system=coordinates),
            )


def test_pixel_index_uses_the_default_coordinates():
    (ptg,) = _three_ways(False, output_coordinate_system=CoordinateSystem.Galactic)[:1]
    np.testing.assert_array_equal(
        ptg.pixel_index(0, 8),
        ptg.pixel_index(0, 8, output_coordinate_system=CoordinateSystem.Galactic),
    )
    np.testing.assert_array_equal(
        ptg.pol_angle(0),
        ptg.pol_angle(0, output_coordinate_system=CoordinateSystem.Galactic),
    )


def test_pixel_index_and_pol_angle_of_more_detectors():
    nside = 8
    for ptg in _three_ways(True):
        pixidx, pol_angle = ptg.pixel_index_and_pol_angle("all", nside)
        assert (
            pixidx.shape
            == pol_angle.shape
            == (N_DETS, ptg.observation.n_samples_process)
        )
        assert pixidx.dtype == np.int32
        for idet in range(N_DETS):
            np.testing.assert_array_equal(pixidx[idet], ptg.pixel_index(idet, nside))
            np.testing.assert_array_equal(pol_angle[idet], ptg.pol_angle(idet))
        np.testing.assert_array_equal(ptg.pixel_index("all", nside), pixidx)
        np.testing.assert_array_equal(ptg.pol_angle([2, 0]), pol_angle[[2, 0]])
        np.testing.assert_array_equal(
            ptg.pixel_index_and_pol_angle([1], nside)[0], pixidx[[1]]
        )
