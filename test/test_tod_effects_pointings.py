"""The TOD effects get the pointings through ObservationPointings: the same
TOD whether the pointings are computed on the fly, precomputed, or passed by
the caller as an array"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

import litebird_sim as lbs
from litebird_sim.hwp_diff_emiss import add_2f_to_observations
from litebird_sim.hwp_harmonics.hwp_harmonics import fill_tod_with_hwp_harmonics
from litebird_sim.maps_and_harmonics import HealpixMap, SphericalHarmonics

N_DETS = 2
WAYS = ["on_the_fly", "precomputed", "external"]


def _simulation(way: str, hwp: lbs.HWP | None = None, tmp_path=None, mueller=None):
    sim = lbs.Simulation(
        base_path=tmp_path,
        start_time=Time("2025-01-01T00:00:00"),
        duration_s=120.0,
        random_seed=1,
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            precession_rate_hz=1 / 3600,
            spin_rate_hz=1 / 60,
            start_time=sim.start_time,
        )
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=np.deg2rad(50.0))
    )
    dets = [
        lbs.DetectorInfo(
            name=f"d{i}",
            sampling_rate_hz=2.0,
            bandcenter_ghz=140.0,
            pol_angle_rad=0.3 * i,
            pointing_theta_phi_psi_deg=[0.0, 0.0, 0.0],
            quat=lbs.RotQuaternion(
                np.array(lbs.quat_rotation_y(np.deg2rad(1.0 + 2.0 * i)))
            ),
        )
        for i in range(N_DETS)
    ]
    sim.create_observations(detectors=dets, tod_dtype=np.float64)
    if mueller is not None:
        for cur_obs in sim.observations:
            for idet in range(cur_obs.n_detectors):
                cur_obs.mueller_hwp[idet] = mueller
    if hwp is not None:
        sim.set_hwp(hwp)
    sim.prepare_pointings()
    if way == "precomputed":
        sim.precompute_pointings()

    pointings = None
    if way == "external":
        pointings = [obs.get_pointings("all")[0] for obs in sim.observations]
    return sim, pointings


def _alms(coordinates=lbs.CoordinateSystem.Galactic, lmax=16):
    rng = np.random.default_rng(12345)
    nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax)
    sky = SphericalHarmonics(
        values=rng.normal(size=(3, nalm)) + 1j * rng.normal(size=(3, nalm)),
        lmax=lmax,
        coordinates=coordinates,
    )
    sky.values[:, : lmax + 1] = sky.values[:, : lmax + 1].real
    return sky


def _tods(run):
    """Run `run(sim, pointings)` for the three ways and return the TODs"""
    tods = []
    for way in WAYS:
        sim, pointings = _simulation(way, hwp=lbs.IdealHWP(ang_speed_radpsec=2.0))
        run(sim, pointings)
        tods.append(sim.observations[0].tod.copy())
    return tods


def _assert_all_equal(tods, atol=1e-10):
    assert np.any(tods[0] != 0.0)
    for cur in tods[1:]:
        np.testing.assert_allclose(cur, tods[0], rtol=0, atol=atol)


def test_beam_convolution():
    lmax = 16
    sky = _alms(lmax=lmax)
    beam = lbs.gauss_beam_to_alm(lmax=lmax, mmax=4, fwhm_rad=np.deg2rad(5.0))
    params = lbs.BeamConvolutionParameters(
        lmax=lmax, mmax=4, single_precision=False, strict_typing=False
    )

    def run(sim, pointings):
        lbs.add_convolved_sky_to_observations(
            sim.observations,
            sky_alms=sky,
            beam_alms=beam,
            pointings=pointings,
            convolution_params=params,
        )

    _assert_all_equal(_tods(run))


@pytest.mark.parametrize("apply_convolution", [False, True])
def test_dipole(apply_convolution):
    beam = lbs.gauss_beam_to_alm(lmax=8, mmax=2, fwhm_rad=np.deg2rad(1.0))

    def run(sim, pointings):
        pos_vel = lbs.spacecraft_pos_and_vel(
            lbs.SpacecraftOrbit(sim.start_time),
            observations=sim.observations,
            delta_time_s=10.0,
        )
        lbs.add_dipole_to_observations(
            sim.observations,
            pos_and_vel=pos_vel,
            pointings=pointings,
            dipole_type=lbs.DipoleType.LINEAR,
            apply_convolution=apply_convolution,
            beam_alms=beam,
        )

    _assert_all_equal(_tods(run), atol=1e-12)


def test_hwp_diff_emiss():
    def run(sim, pointings):
        add_2f_to_observations(sim.observations, amplitude_2f_k=1.0)

    _assert_all_equal(_tods(run), atol=0)


def test_hwp_harmonics():
    nside = 16
    rng = np.random.default_rng(1)
    sky = HealpixMap(
        values=rng.normal(size=(3, 12 * nside**2)),
        nside=nside,
        coordinates=lbs.CoordinateSystem.Galactic,
    )
    hwp = lbs.NonIdealHWP(
        ang_speed_radpsec=2.0,
        harmonic_expansion=True,
        calculus=lbs.HWPFormalism.MUELLER,
    )
    mueller = {
        "0f": np.diag([1.0, 0.0, 0.0]),
        "2f": np.zeros((3, 3)),
        "4f": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float64),
    }

    mueller_phases = {"2f": np.zeros((3, 3)), "4f": np.zeros((3, 3))}

    tods = []
    for way in WAYS:
        sim, pointings = _simulation(way, hwp=hwp, mueller=mueller)
        lbs.scan_map_in_observations(
            sim.observations,
            maps=sky,
            pointings=pointings,
            mueller_phases=mueller_phases,
        )
        tods.append(sim.observations[0].tod.copy())

    # Called directly, it gets the pointings and the HWP angle by itself
    for way in ["on_the_fly", "precomputed"]:
        sim, _ = _simulation(way, hwp=hwp, mueller=mueller)
        (obs,) = sim.observations
        fill_tod_with_hwp_harmonics(
            hwp=hwp,
            observation=obs,
            tod=obs.tod,
            maps=sky,
            mueller_phases=mueller_phases,
        )
        tods.append(obs.tod.copy())

    _assert_all_equal(tods)


@pytest.mark.parametrize(
    "coordinates", [lbs.CoordinateSystem.Ecliptic, lbs.CoordinateSystem.Galactic]
)
def test_madam(tmp_path, coordinates):
    params = lbs.ExternalDestriperParameters(nside=16)
    columns = []
    for way in WAYS:
        sim, pointings = _simulation(
            way, hwp=lbs.IdealHWP(ang_speed_radpsec=2.0), tmp_path=tmp_path / way
        )
        result = lbs.save_simulation_for_madam(
            sim,
            params=params,
            pointings=pointings,
            output_coordinate_system=coordinates,
            save_tods=False,
        )
        assert len(result["pointing_files"]) == N_DETS
        cur = []
        for cur_file in sorted(result["pointing_files"], key=lambda x: x["det_name"]):
            with fits.open(cur_file["file_name"]) as inpf:
                cur.append(np.stack([inpf[1].data[x] for x in ("THETA", "PHI", "PSI")]))
        columns.append(np.array(cur))

    # The same angles, but ψ can differ by 2π
    for cur in columns[1:]:
        diff = np.angle(np.exp(1j * (cur - columns[0])))
        np.testing.assert_allclose(diff, 0.0, rtol=0, atol=1e-6)
