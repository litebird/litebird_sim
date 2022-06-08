# -*- encoding: utf-8 -*-

"Test the interface to the TOAST destriper"

from pathlib import Path
import numpy as np
import astropy.units as u
import litebird_sim as lbs
import healpy
from numpy.random import MT19937, RandomState, SeedSequence

from litebird_sim import CoordinateSystem


COORDINATE_SYSTEM_STR = {
    CoordinateSystem.Ecliptic: "ecliptic",
    CoordinateSystem.Galactic: "galactic",
}


def run_destriper_tests(tmp_path, coordinates: CoordinateSystem):
    coordinates_str = COORDINATE_SYSTEM_STR[coordinates]

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
        split_list_over_processes=False,
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
        coordinate_system=coordinates,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
        return_npp=True,
        return_invnpp=True,
        return_rcond=True,
    )

    results = lbs.destripe(sim, instr, params=params)
    assert results.coordinate_system == coordinates

    ref_map_path = Path(__file__).parent / "destriper_reference"

    hit_map_filename = ref_map_path / f"destriper_hit_map_{coordinates_str}.fits.gz"
    # healpy.write_map(hit_map_filename, results.hit_map, dtype="int32", overwrite=True)
    np.testing.assert_allclose(
        results.hit_map, healpy.read_map(hit_map_filename, field=None, dtype=np.int32)
    )

    binned_map_filename = (
        ref_map_path / f"destriper_binned_map_{coordinates_str}.fits.gz"
    )
    # healpy.write_map(
    #     binned_map_filename,
    #     results.binned_map,
    #     dtype=list((np.float32 for i in range(3))),
    #     overwrite=True,
    # )
    ref_binned = healpy.read_map(
        binned_map_filename, field=None, dtype=list((np.float32 for i in range(3)))
    )
    assert results.binned_map.shape == ref_binned.shape
    np.testing.assert_allclose(results.binned_map, ref_binned, rtol=1e-2, atol=1e-3)

    destriped_map_filename = (
        ref_map_path / f"destriper_destriped_map_{coordinates_str}.fits.gz"
    )
    # healpy.write_map(
    #     destriped_map_filename,
    #     results.destriped_map,
    #     dtype=list((np.float32 for i in range(3))),
    #     overwrite=True,
    # )
    ref_destriped = healpy.read_map(
        destriped_map_filename, field=None, dtype=list((np.float32 for i in range(3)))
    )
    assert results.destriped_map.shape == ref_destriped.shape
    np.testing.assert_allclose(
        results.destriped_map, ref_destriped, rtol=1e-2, atol=1e-3
    )

    npp_filename = ref_map_path / f"destriper_npp_{coordinates_str}.fits.gz"
    # healpy.write_map(
    #     npp_filename,
    #     results.npp,
    #     dtype=list((np.float32 for i in range(6))),
    #     overwrite=True,
    # )
    ref_npp = healpy.read_map(
        npp_filename, field=None, dtype=list((np.float32 for i in range(6)))
    )
    assert results.npp.shape == ref_npp.shape
    np.testing.assert_allclose(results.npp, ref_npp, rtol=1e-2, atol=1e-3)

    invnpp_filename = ref_map_path / f"destriper_invnpp_{coordinates_str}.fits.gz"
    # healpy.write_map(
    #     invnpp_filename,
    #     results.invnpp,
    #     dtype=list((np.float32 for i in range(6))),
    #     overwrite=True,
    # )
    ref_invnpp = healpy.read_map(
        invnpp_filename, field=None, dtype=list((np.float32 for i in range(6)))
    )
    assert results.invnpp.shape == ref_invnpp.shape
    np.testing.assert_allclose(results.invnpp, ref_invnpp, rtol=1e-2, atol=1e-3)

    rcond_filename = ref_map_path / f"destriper_rcond_{coordinates_str}.fits.gz"
    # healpy.write_map(
    #     rcond_filename,
    #     results.rcond,
    #     dtype=np.float32,
    #     overwrite=True,
    # )
    assert np.allclose(
        results.rcond, healpy.read_map(rcond_filename, field=None, dtype=np.float32)
    )


def test_destriper_ecliptic(tmp_path):
    run_destriper_tests(tmp_path=tmp_path, coordinates=CoordinateSystem.Ecliptic)


def test_destriper_galactic(tmp_path):
    run_destriper_tests(tmp_path=tmp_path, coordinates=CoordinateSystem.Galactic)


def test_destriper_coordinate_consistency(tmp_path):
    # Here we check that MBS uses the same coordinate system as the destriper
    # in «Galactic» mode: specifically, we create a noiseless TOD from a CMB
    # map in Galactic coordinates and run the destriper asking to use Galactic
    # coordinates again. Since the TOD was noiseless, the binned map should be
    # the same as the input map, except for two features:
    #
    # 1. It does not cover the whole sky
    # 2. A few pixels at the border of the observed region are not properly
    #    constrained and their value will be set to zero
    #
    # This test checks that «most» of the two maps agree, i.e., the 5% and 95%
    # percentiles are smaller than some (small) threshold.

    sim = lbs.Simulation(
        base_path="destriper_output",
        start_time=0,
        duration_s=10 * 86400.0,
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
    instr = lbs.InstrumentInfo(
        name="mock_instrument",
        spin_boresight_angle_rad=np.deg2rad(65),
    )

    detectors = [
        lbs.DetectorInfo(
            name="noiseless_detector",
            sampling_rate_hz=10.0,
            fwhm_arcmin=60.0,
            bandcenter_ghz=40.0,
            bandwidth_ghz=12.0,
        ),
    ]

    # We create two detectors, whose polarization angles are separated by π/2
    sim.create_observations(
        detectors=detectors,
        dtype_tod=np.float64,  # Needed if you use the TOAST destriper
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
        split_list_over_processes=False,
    )

    for obs in sim.observations:
        lbs.get_pointings(
            obs,
            spin2ecliptic_quats=sim.spin2ecliptic_quats,
            detector_quats=None,
            bore2spin_quat=instr.bore2spin_quat,
        )

    params = lbs.MbsParameters(
        make_cmb=True,
    )
    mbs = lbs.Mbs(
        simulation=sim,
        parameters=params,
        detector_list=detectors,
    )
    (healpix_maps, file_paths) = mbs.run_all()

    lbs.scan_map_in_observations(obs=sim.observations, maps=healpix_maps)

    params = lbs.DestriperParameters(
        nside=healpy.npix2nside(len(healpix_maps[detectors[0].name][0])),
        coordinate_system=lbs.CoordinateSystem.Galactic,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
    )

    result = lbs.destripe(sim, instr, params)

    inp = healpix_maps[detectors[0].name]  # Input CMB map in Galactic coordinates
    out = result.binned_map  # The binned map produced by the destriper
    hit = result.hit_map  # The hit map produced by the destriper

    # We do not consider unseen pixels nor pixels that have not been properly
    # constrained by our mock detector
    mask = (hit == 0) | (out == 0.0)
    inp[mask] = np.NaN
    out[mask] = np.NaN

    # Ideally this should be ≈0
    diff = inp - out

    # We check the closeness of `diff` to zero through the 5% and 95% percentiles
    low_limit, high_limit = np.percentile(diff[np.isfinite(diff)], (0.05, 0.95))
    assert np.abs(low_limit) < 1e-9
    assert np.abs(high_limit) < 1e-9
