"""Test GLS mapmaking with pysanepic"""

import inspect
import tempfile
from unittest import mock

import astropy.time
import healpy as hp
import numpy as np
import pytest

import litebird_sim as lbs

pysanepic = pytest.importorskip(
    modname="pysanepic", reason="Couldn't import 'pysanepic' module"
)


def test_import_error():
    with mock.patch.dict("sys.modules", {"pysanepic": None}):
        with pytest.raises(ImportError, match="Could not import `pysanepic`"):
            lbs.mapmaking.make_sanepic_gls_map(nside=1, observations=[])


def test_sanepic_noiseless_matches_binner():
    # On noiseless data every least-squares map-maker returns the same map.
    # HWP, γ ≠ 1 and a chunk length that does not divide the data are included.
    nside = 16
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    with tempfile.TemporaryDirectory() as tmp:
        sim = lbs.Simulation(
            base_path=tmp,
            imo=imo,
            start_time=astropy.time.Time("2030-01-01T00:00:00"),
            duration_s=4 * 3600.0,
            random_seed=1,
            mpi_comm=lbs.MPI_COMM_WORLD,
        )
        sim.set_instrument(
            lbs.InstrumentInfo.from_imo(
                imo, "/releases/vPTEP/satellite/LFT/instrument_info"
            )
        )
        dets = [
            lbs.DetectorInfo.from_imo(
                url=f"/releases/vPTEP/satellite/LFT/L4-140/{d}/detector_info", imo=imo
            )
            for d in ["000_001_017_QB_140_T", "000_001_017_QB_140_B"]
        ]
        for d in dets:
            d.sampling_rate_hz = 5.0
        dets[1].pol_efficiency = 0.9
        sim.set_scanning_strategy(
            imo_url="/releases/vPTEP/satellite/scanning_parameters/"
        )
        sim.create_observations(
            detectors=dets,
            n_blocks_time=lbs.MPI_COMM_WORLD.size,
            split_list_over_processes=False,
        )
        sim.set_hwp(lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60))
        sim.prepare_pointings()
        ch = lbs.FreqChannelInfo.from_imo(
            url="/releases/vPTEP/satellite/LFT/L4-140/channel_info", imo=imo
        )
        sim.get_sky(
            parameters=lbs.SkyGenerationParams(
                make_cmb=True,
                make_fg=False,
                seed_cmb=1,
                apply_beam=False,
                units="K_CMB",
                output_type="map",
                nside=nside,
            ),
            channels=[ch],
            store_in_observation=True,
        )
        sim.fill_tods()

        binned = sim.make_binned_map(nside=nside).binned_map
        result = sim.make_sanepic_gls_map(
            nside=nside, params=pysanepic.GLSParameters(chunk_s=1700.0)
        )

    assert result.converged
    good = (
        (result.maps[1] != hp.UNSEEN)
        & np.isfinite(binned[1])
        & (binned[1] != hp.UNSEEN)
    )
    assert good.sum() > 0.3 * good.size
    # The PCG tolerance bounds the absolute error, set by the I amplitude
    scale = np.abs(binned[0][good]).max()
    for i in range(3):
        assert np.abs(result.maps[i][good] - binned[i][good]).max() < 1e-5 * scale


def test_sanepic_noise_model(monkeypatch):
    # The noise model reaches pysanepic.DetectorData; unknown models are rejected
    seen = []

    def fake_make_maps(data, *args, **kwargs):
        seen.extend(d.noise.model for d in data)

    monkeypatch.setattr(pysanepic, "make_maps", fake_make_maps)
    sim = lbs.Simulation(start_time=0, duration_s=100.0, random_seed=1)
    sim.set_instrument(lbs.InstrumentInfo(name="instr"))
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="d", sampling_rate_hz=1.0, net_ukrts=50.0)]
    )
    sim.set_scanning_strategy(lbs.SpinningScanningStrategy(0.785, 0.0, 0.01))
    sim.prepare_pointings()
    lbs.make_sanepic_gls_map(
        nside=8, observations=sim.observations, noise_model="keshner"
    )
    assert seen == ["keshner"]
    with pytest.raises(ValueError, match="Unknown noise model"):
        lbs.make_sanepic_gls_map(
            nside=8, observations=sim.observations, noise_model="x"
        )


def test_sanepic_flags(monkeypatch):
    # Per-detector and common flags are combined and passed to pysanepic
    seen = []
    monkeypatch.setattr(
        pysanepic, "make_maps", lambda data, *a, **k: seen.extend(d.flags for d in data)
    )
    sim = lbs.Simulation(start_time=0, duration_s=10.0, random_seed=1)
    sim.set_instrument(lbs.InstrumentInfo(name="instr"))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(name=n, sampling_rate_hz=1.0, net_ukrts=50.0)
            for n in ("a", "b")
        ]
    )
    sim.set_scanning_strategy(lbs.SpinningScanningStrategy(0.785, 0.0, 0.01))
    sim.prepare_pointings()
    obs = sim.observations[0]
    obs.local_flags = np.zeros((2, obs.n_samples_local), dtype=np.uint16)
    obs.local_flags[1, 3] = 4
    obs.global_flags = np.zeros(obs.n_samples_local, dtype=np.uint8)
    obs.global_flags[7] = 1
    lbs.make_sanepic_gls_map(
        nside=8, observations=sim.observations, flags=["local_flags", "global_flags"]
    )
    assert [list(np.flatnonzero(f)) for f in seen] == [[7], [3, 7]]
    seen.clear()
    lbs.make_sanepic_gls_map(nside=8, observations=sim.observations)
    assert seen == [None, None]


def test_sanepic_noise_estimation():
    # Spectra estimated from the residual of a first map, per detector name,
    # and passed back to make_sanepic_gls_map
    if not hasattr(pysanepic, "estimate_noise"):
        pytest.skip("pysanepic without estimate_noise")
    sim = lbs.Simulation(start_time=0, duration_s=86400.0, random_seed=2)
    sim.set_instrument(lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=0.87))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=n,
                sampling_rate_hz=2.0,
                net_ukrts=50.0,
                fknee_mhz=50.0,
                alpha=1.0,
                fmin_hz=1e-5,
                pol_angle_rad=a,
            )
            for n, a in (("a", 0.0), ("b", np.pi / 4), ("c", np.pi / 8))
        ],
        num_of_obs_per_detector=2,
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(0.785, 1 / 3600 / 3.2, 0.05 / 60),
        delta_time_s=60,
    )
    sim.prepare_pointings()
    sim.add_noise(noise_type="one_over_f")
    params = pysanepic.GLSParameters(chunk_s=43200.0, tol=1e-8)
    first = sim.make_sanepic_gls_map(nside=8, params=params, append_to_report=False)
    noise = sim.estimate_sanepic_noise(first, chunk_s=43200.0, nbins=20)
    assert sorted(noise) == ["a", "b", "c"]
    sigma = 50.0 * np.sqrt(2.0) * 1e-6
    for est in noise.values():
        # high frequencies: the white level 2 sigma^2 / fs
        high = est.freqs_hz > 0.5
        assert abs(np.mean(est.psd[high]) / (2 * sigma**2 / 2.0) - 1) < 0.1
    second = sim.make_sanepic_gls_map(
        nside=8, params=params, noise=noise, append_to_report=False
    )
    assert second.converged


def test_sanepic_correlated_noise():
    # Common-mode noise simulated by add_noise(noise_type="correlated"): the
    # map with the same model of the correlation is less noisy than ignoring it
    if not hasattr(pysanepic, "DetectorGroup"):
        pytest.skip("pysanepic without DetectorGroup")
    sim = lbs.Simulation(start_time=0, duration_s=86400.0, random_seed=3)
    sim.set_instrument(lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=0.87))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=n,
                sampling_rate_hz=2.0,
                net_ukrts=50.0,
                fknee_mhz=100.0,
                alpha=1.5,
                fmin_hz=1e-4,
                pol_angle_rad=a,
            )
            for n, a in (("a", 0.0), ("b", np.pi / 4), ("c", np.pi / 8), ("d", 0.3))
        ],
        num_of_obs_per_detector=2,
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(0.785, 1 / 3600 / 3.2, 0.05 / 60),
        delta_time_s=60,
    )
    sim.prepare_pointings()
    correlation = {"groups": [0, 0, 0, 0], "rho": 0.8}
    sim.add_noise(noise_type="correlated", correlation=correlation)
    params = pysanepic.GLSParameters(chunk_s=43200.0, tol=1e-10)
    rms = {}
    results = {}
    for label, corr in (("independent", None), ("correlated", correlation)):
        r = sim.make_sanepic_gls_map(
            nside=8, params=params, correlation=corr, append_to_report=False
        )
        assert r.converged
        seen = r.maps[1] != hp.UNSEEN
        rms[label] = np.std(r.maps[1:, seen])
        results[label] = r
    assert rms["correlated"] < 0.8 * rms["independent"]

    # the common mode estimated from the residual of the first map
    if "n_modes" in inspect.signature(pysanepic.estimate_noise).parameters:
        est = sim.estimate_sanepic_noise(
            results["independent"], chunk_s=43200.0, nbins=20, correlation=correlation
        )
        (mode,) = est[("a", "b", "c", "d")]
        assert np.allclose(mode.coupling, 1.0, atol=0.05)  # same rho for all
        r = sim.make_sanepic_gls_map(
            nside=8,
            params=params,
            noise=est,
            correlation=correlation,
            append_to_report=False,
        )
        seen = r.maps[1] != hp.UNSEEN
        assert np.std(r.maps[1:, seen]) < 1.05 * rms["correlated"]
        with pytest.raises(ValueError, match="no common modes"):
            sim.make_sanepic_gls_map(
                nside=8,
                params=params,
                noise={"a": est["a"]},
                correlation=correlation,
            )
    with pytest.raises(ValueError, match="corr_matrix"):
        sim.make_sanepic_gls_map(
            nside=8, params=params, correlation={"corr_matrix": np.eye(4)}
        )


def _calibration_sim(n_days=30, lmax=None):
    """Noiseless: a sky at nside 8 (a map, or with `lmax` alm interpolated at
    the pointing), 4 detectors at 0.05 Hz, n_days daily observations, with a
    "dipole" TOD component; returns (sim, sky maps or alm)."""
    sim = lbs.Simulation(
        start_time=astropy.time.Time("2030-01-01T00:00:00"),
        duration_s=n_days * 86400.0,
        random_seed=5,
    )
    sim.set_instrument(lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=0.87))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=n,
                sampling_rate_hz=0.05,
                net_ukrts=50.0,
                fknee_mhz=0.5,
                alpha=1.0,
                fmin_hz=1e-5,
                pol_angle_rad=a,
                bandcenter_ghz=140.0,
            )
            for n, a in (("a", 0.0), ("b", np.pi / 4), ("c", np.pi / 8), ("d", 0.3))
        ],
        num_of_obs_per_detector=n_days,
        tods=[
            lbs.TodDescription(
                name=c, dtype=np.float64, units=lbs.Units.K_CMB, description=c
            )
            for c in ("tod", "dipole")
        ],
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(0.785, 1 / 3600 / 3.2, 0.05 / 60),
        delta_time_s=60,
    )
    sim.prepare_pointings()
    rng = np.random.default_rng(5)
    sky = hp.smoothing(rng.normal(0, 1e-4, (3, 12 * 8**2)), fwhm=np.radians(10))
    gal = lbs.CoordinateSystem.Galactic
    if lmax is None:
        maps = lbs.HealpixMap(sky, coordinates=gal)
    else:
        sky = np.array(hp.map2alm(sky, lmax=lmax))
        maps = lbs.SphericalHarmonics(values=sky, lmax=lmax, coordinates=gal)
    sim.fill_tods(maps=maps, append_to_report=False)
    return sim, sky


def _apply_gains(sim, seed=7):
    """Multiply the TOD by a gain per detector and day; returns them."""
    rng = np.random.default_rng(seed)
    true = {}
    for obs in sim.observations:
        for i, name in enumerate(obs.name):
            true[name, obs.start_time_local.jd] = 1 + 0.01 * rng.normal()
            obs.tod[i] *= true[name, obs.start_time_local.jd]
    return true


def _check_gains(result, sim, true, tol):
    k = 0
    for obs in sim.observations:
        for name in obs.name:
            assert abs(result.gains[k][0] - true[name, obs.start_time_local.jd]) < tol
            k += 1


def test_sanepic_calibration():
    # Gains changing every day, noiseless: recovered from the dipole in the
    # "dipole" component (filled by add_dipole, as the data)
    if not hasattr(pysanepic, "CalibrationParameters"):
        pytest.skip("pysanepic without calibration")
    sim, sky = _calibration_sim()
    sim.add_dipole(append_to_report=False)
    sim.add_dipole(component="dipole", append_to_report=False)
    true = _apply_gains(sim)
    params = pysanepic.GLSParameters(chunk_s=86400.0, tol=1e-20, min_pol_rcond=0.0)
    cal = pysanepic.CalibrationParameters(gain_period_s=86400.0)
    r = sim.make_sanepic_gls_map(
        nside=8,
        params=params,
        calibration=cal,
        dipole_component="dipole",
        append_to_report=False,
    )
    _check_gains(r, sim, true, 1e-6)
    seen = r.maps[1] != hp.UNSEEN
    err = (r.maps - sky)[:, seen]
    err[0] -= err[0].mean()  # I monopole unconstrained with 1/f weights
    assert np.abs(err).max() < 1e-3 * np.abs(sky).max()
    with pytest.raises(ValueError, match="dipole_component"):
        sim.make_sanepic_gls_map(nside=8, calibration=cal, append_to_report=False)


def test_sanepic_calibration_keep_solar_dipole():
    # The solar dipole kept in the maps: only the orbital dipole in the
    # component, the solar one in the sky model (alm); the maps are the sky
    # plus the solar dipole
    if not hasattr(pysanepic, "CalibrationParameters"):
        pytest.skip("pysanepic without sky_model")
    sim, sky_alm = _calibration_sim(lmax=16)
    # LINEAR: the dipole of solar + orbital velocity is exactly the sum of the two
    linear = lbs.DipoleType.LINEAR
    sim.add_dipole(dipole_type=linear, append_to_report=False)  # solar + orbital
    orbital = lbs.spacecraft_pos_and_vel(
        lbs.SpacecraftOrbit(sim.start_time, solar_velocity_km_s=0.0),
        observations=sim.observations,
        delta_time_s=86400.0,
    )
    lbs.add_dipole_to_observations(
        sim.observations, orbital, dipole_type=linear, component="dipole"
    )
    true = _apply_gains(sim)
    # the solar dipole (linear), from the Planck direction in Galactic coordinates
    v = hp.ang2vec(
        np.pi / 2 - lbs.constants.SOLAR_VELOCITY_GAL_LAT_RAD,
        lbs.constants.SOLAR_VELOCITY_GAL_LON_RAD,
    )
    amp = (
        lbs.constants.T_CMB_K
        * lbs.constants.SOLAR_VELOCITY_KM_S
        / lbs.constants.C_LIGHT_KM_OVER_S
    )
    solar_map = amp * (np.array(hp.pix2vec(64, np.arange(12 * 64**2))).T @ v)
    solar_alm = np.zeros_like(sky_alm)
    solar_alm[0] = hp.map2alm(solar_map, lmax=16, iter=3)
    params = pysanepic.GLSParameters(chunk_s=86400.0, tol=1e-20, min_pol_rcond=0.0)
    r = sim.make_sanepic_gls_map(
        nside=8,
        params=params,
        calibration=pysanepic.CalibrationParameters(gain_period_s=86400.0),
        dipole_component="dipole",
        sky_model=sky_alm + solar_alm,
        append_to_report=False,
    )
    _check_gains(r, sim, true, 1e-6)
    # expected: the binned map of the signal (sky + solar dipole) of the samples
    for obs in sim.observations:
        for i, name in enumerate(obs.name):
            obs.tod[i] = (
                obs.tod[i] / true[name, obs.start_time_local.jd] - obs.dipole[i]
            )
    binned = sim.make_binned_map(nside=8).binned_map
    seen = r.maps[1] != hp.UNSEEN
    assert np.abs((r.maps - binned)[:, seen]).max() < 1e-6 * amp


def test_sanepic_sky_model_and_templates():
    # A sky model as zero alm (pointings passed as angles) or as the sky map
    # leaves the maps unchanged; a foreground seen with a different amplitude
    # by each detector is fitted with a sky template, and a TOD component
    # with a coupling per detector with a TOD template
    if not hasattr(pysanepic, "Template"):
        pytest.skip("pysanepic without templates")
    sim, sky = _calibration_sim(n_days=4)
    params = pysanepic.GLSParameters(chunk_s=86400.0, tol=1e-20, min_pol_rcond=0.0)
    ref = sim.make_sanepic_gls_map(nside=8, params=params, append_to_report=False)
    seen = ref.maps[1] != hp.UNSEEN
    zero = lbs.SphericalHarmonics(
        values=np.zeros((3, hp.Alm.getsize(4)), complex), lmax=4
    )
    for model in (zero, sky):
        r = sim.make_sanepic_gls_map(
            nside=8, params=params, sky_model=model, append_to_report=False
        )
        err = (r.maps - ref.maps)[:, seen]
        err[0] -= err[0].mean()
        assert np.abs(err).max() < 1e-6 * np.abs(sky).max()  # the PCG tolerance
    # the pointings as angles give the same maps; alm need them
    r = sim.make_sanepic_gls_map(
        nside=8, params=params, pointing_form="angles", append_to_report=False
    )
    assert np.allclose(r.maps, ref.maps, rtol=0, atol=1e-6 * np.abs(sky).max())
    with pytest.raises(ValueError, match="need the angles"):
        sim.make_sanepic_gls_map(
            nside=8, params=params, sky_model=zero, pointing_form="pixels"
        )
    with pytest.raises(ValueError, match="pointing_form"):
        sim.make_sanepic_gls_map(nside=8, params=params, pointing_form="pix")

    fg = hp.smoothing(
        np.random.default_rng(9).normal(0, 3e-4, (3, 12 * 8**2)), fwhm=np.radians(15)
    )
    amp = {"a": 0.04, "b": -0.01, "c": 0.02, "d": -0.03}
    mean = np.mean(list(amp.values()))  # the common part goes into the map
    for obs in sim.observations:
        signal = obs.tod.copy()
        obs.tod[:] = 0.0
        lbs.scan_map_in_observations(
            obs, maps=lbs.HealpixMap(fg, coordinates=lbs.CoordinateSystem.Galactic)
        )
        for i, name in enumerate(obs.name):
            obs.tod[i] = signal[i] + (1 + amp[name]) * obs.tod[i]
    fg_tmpl = pysanepic.Template(sky=fg, sum_zero=True)
    r = sim.make_sanepic_gls_map(
        nside=8, params=params, templates={"fg": fg_tmpl}, append_to_report=False
    )
    assert all(abs(r.templates["fg"][k] - (v - mean)) < 1e-6 for k, v in amp.items())
    err = (r.maps - ref.maps - (1 + mean) * fg)[:, seen]
    err[0] -= err[0].mean()
    assert np.abs(err).max() < 1e-6 * np.abs(sky).max()

    # a TOD component (e.g. a thermometer) seen with a coupling per detector
    rng = np.random.default_rng(11)
    coupling = {"a": 2e-4, "b": -1e-4, "c": 5e-5, "d": 3e-4}
    for obs in sim.observations:
        obs.thermal = rng.normal(size=obs.tod.shape)
        for i, name in enumerate(obs.name):
            obs.tod[i] += coupling[name] * obs.thermal[i]
    r = sim.make_sanepic_gls_map(
        nside=8,
        params=params,
        templates={"fg": fg_tmpl, "thermal": pysanepic.Template()},
        template_covariance=True,
        append_to_report=False,
    )
    assert all(abs(r.templates["thermal"][k] - v) < 1e-9 for k, v in coupling.items())
    assert all(abs(r.templates["fg"][k] - (v - mean)) < 1e-6 for k, v in amp.items())
    assert r.template_covariance["labels"][-1] == ("thermal", "d")


def _beam_sim(quats, hwp, lmax=16, n_days=4):
    """Noiseless, as _calibration_sim, with detector quaternions `quats` and
    optionally an ideal HWP; returns (sim, sky as SphericalHarmonics)."""
    sim = lbs.Simulation(
        start_time=astropy.time.Time("2030-01-01T00:00:00"),
        duration_s=n_days * 86400.0,
        random_seed=5,
    )
    sim.set_instrument(lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=0.87))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=n,
                sampling_rate_hz=0.05,
                net_ukrts=50.0,
                fknee_mhz=0.5,
                alpha=1.0,
                fmin_hz=1e-5,
                pol_angle_rad=a,
                bandcenter_ghz=140.0,
                quat=q,
            )
            for (n, a), q in zip(
                (("a", 0.0), ("b", np.pi / 4), ("c", np.pi / 8), ("d", 0.3)), quats
            )
        ],
        num_of_obs_per_detector=n_days,
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(0.785, 1 / 3600 / 3.2, 0.05 / 60),
        delta_time_s=60,
    )
    if hwp:
        sim.set_hwp(lbs.IdealHWP(ang_speed_radpsec=2 * np.pi / 600))
    sim.prepare_pointings()
    sky = hp.smoothing(
        np.random.default_rng(5).normal(0, 1e-4, (3, 12 * 8**2)), fwhm=np.radians(10)
    )
    alm = np.array(hp.map2alm(sky, lmax=lmax))
    return sim, lbs.SphericalHarmonics(
        values=alm, lmax=lmax, coordinates=lbs.CoordinateSystem.Galactic
    )


def test_sanepic_beam_templates():
    # Detectors whose quaternions are off by 0.3-0.6 deg, mapped with the
    # nominal ones: the beam templates recover the offsets. A rotation of the
    # quaternion about x (y) moves the beam along the axis at psi (psi + 90
    # deg), so the offsets in the frame of the templates, at psi +
    # pol_angle, are those rotated by -pol_angle; with and without HWP
    if not hasattr(pysanepic, "beam_tods"):
        pytest.skip("pysanepic without template helpers")
    from litebird_sim.quaternions import quat_rotation_x, quat_rotation_y

    rho = np.radians(0.3)
    quats = [
        np.array(quat_rotation_x(rho)),
        np.array(quat_rotation_y(rho)),
        np.array(quat_rotation_x(-rho)),
        np.array(quat_rotation_y(-2 * rho)),
    ]
    uv = {"a": (rho, 0), "b": (0, rho), "c": (-rho, 0), "d": (0, -2 * rho)}
    pols = {"a": 0.0, "b": np.pi / 4, "c": np.pi / 8, "d": 0.3}
    params = pysanepic.GLSParameters(chunk_s=86400.0, tol=1e-20, min_pol_rcond=0.0)
    for hwp, ellipticity in ((False, False), (True, True)):
        data, sky = _beam_sim(quats, hwp)
        data.fill_tods(maps=sky, append_to_report=False)
        sim, _ = _beam_sim([np.array([0.0, 0.0, 0.0, 1.0])] * 4, hwp)
        for obs, obs_data in zip(sim.observations, data.observations):
            obs.tod[:] = obs_data.tod
        templates = sim.compute_sanepic_beam_templates(
            sky_model=sky, ellipticity=ellipticity
        )
        names = ["offset_x", "offset_y"] + (
            ["ell_plus", "ell_cross"] if ellipticity else []
        )
        assert list(templates) == names
        assert not any(t.sum_zero for t in templates.values())  # offsets; HWP
        assert sim.observations[0].offset_x.shape == sim.observations[0].tod.shape
        r = sim.make_sanepic_gls_map(
            nside=8,
            params=params,
            sky_model=sky,
            templates=templates,
            append_to_report=False,
        )
        par = pysanepic.beam_parameters(r, templates)
        for name, (u, v) in uv.items():
            c, s = np.cos(pols[name]), np.sin(pols[name])
            expect = (c * u + s * v, -s * u + c * v)
            got = (r.templates["offset_x"][name], r.templates["offset_y"][name])
            assert np.hypot(*np.subtract(got, expect)) < 0.02 * rho, (hwp, name, got)
            assert par[name]["offset_x"] == got[0]
    # a TOD template needs its component in the observations
    del sim.observations[0].offset_x
    with pytest.raises(ValueError, match="compute_sanepic_beam_templates"):
        sim.make_sanepic_gls_map(
            nside=8, params=params, sky_model=sky, templates=templates
        )
