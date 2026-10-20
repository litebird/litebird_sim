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
    obs.local_flags = np.zeros((2, obs.n_samples), dtype=np.uint16)
    obs.local_flags[1, 3] = 4
    obs.global_flags = np.zeros(obs.n_samples, dtype=np.uint8)
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
