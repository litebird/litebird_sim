# -*- encoding: utf-8 -*-

"Test the interface to the TOAST destriper"

from pathlib import Path
import numpy as np
import astropy.units as u
import litebird_sim as lbs
import healpy
from numpy.random import MT19937, RandomState, SeedSequence


def test_destriper(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "destriper_output", start_time=0, duration_s=86400.0
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
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(name="0A", sampling_rate_hz=10),
            lbs.DetectorInfo(
                name="0B", sampling_rate_hz=10, quat=lbs.quat_rotation_z(np.pi / 2)
            ),
        ],
        # num_of_obs_per_detector=lbs.MPI_COMM_WORLD.size,
        dtype_tod=np.float64,
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
        distribute=False,
    )

    # Generate some white noise
    rs = RandomState(MT19937(SeedSequence(123456789)))
    for curobs in sim.observations:
        curobs.tod *= 0.0
        curobs.tod += rs.randn(*curobs.tod.shape)

    params = lbs.DestriperParameters(
        nside=16,
        nnz=3,
        baseline_length=100,
        iter_max=10,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
        return_npp=True,
        return_invnpp=True,
        return_rcond=True,
    )

    results = lbs.destripe(sim, instr, params=params)

    ref_map_path = Path(__file__).parent / "destriper_reference"

    hit_map_filename = ref_map_path / "destriper_hit_map.fits.gz"
    # healpy.write_map(hit_map_filename, results.hit_map, dtype=np.int32, overwrite=True)
    assert np.allclose(
        results.hit_map,
        healpy.read_map(hit_map_filename, field=None, verbose=False, dtype=np.int32),
    )

    binned_map_filename = ref_map_path / "destriper_binned_map.fits.gz"
    # healpy.write_map(
    #     binned_map_filename,
    #     results.binned_map,
    #     dtype=list((np.float32 for i in range(3))),
    #     overwrite=True,
    # )
    assert np.allclose(
        results.binned_map,
        healpy.read_map(
            binned_map_filename,
            field=None,
            verbose=False,
            dtype=list((np.float32 for i in range(3))),
        ),
    )

    destriped_map_filename = ref_map_path / "destriper_destriped_map.fits.gz"
    # healpy.write_map(
    #     destriped_map_filename,
    #     results.destriped_map,
    #     dtype=list((np.float32 for i in range(3))),
    #     overwrite=True,
    # )
    assert np.allclose(
        results.destriped_map,
        healpy.read_map(
            destriped_map_filename,
            field=None,
            verbose=False,
            dtype=list((np.float32 for i in range(3))),
        ),
    )

    npp_filename = ref_map_path / "destriper_npp.fits.gz"
    # healpy.write_map(
    #     npp_filename,
    #     results.npp,
    #     dtype=list((np.float32 for i in range(6))),
    #     overwrite=True,
    # )
    assert np.allclose(
        results.npp,
        healpy.read_map(
            npp_filename,
            field=None,
            verbose=False,
            dtype=list((np.float32 for i in range(6))),
        ),
    )

    invnpp_filename = ref_map_path / "destriper_invnpp.fits.gz"
    # healpy.write_map(
    #     invnpp_filename,
    #     results.invnpp,
    #     dtype=list((np.float32 for i in range(6))),
    #     overwrite=True,
    # )
    assert np.allclose(
        results.invnpp,
        healpy.read_map(
            invnpp_filename,
            field=None,
            verbose=False,
            dtype=list((np.float32 for i in range(6))),
        ),
    )

    rcond_filename = ref_map_path / "destriper_rcond.fits.gz"
    # healpy.write_map(
    #     rcond_filename,
    #     results.rcond,
    #     dtype=np.float32,
    #     overwrite=True,
    # )
    assert np.allclose(
        results.rcond,
        healpy.read_map(rcond_filename, field=None, verbose=False, dtype=np.float32),
    )
