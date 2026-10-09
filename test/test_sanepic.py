"""Test GLS mapmaking with pysanepic"""

import tempfile
from unittest import mock

import astropy.time
import healpy as hp
import numpy as np
import pytest

import litebird_sim as lbs

pytest.importorskip(modname="pysanepic", reason="Couldn't import 'pysanepic' module")


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
        result = sim.make_sanepic_gls_map(nside=nside, chunk_s=1700.0)

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
