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


def test_basic_functionality(tmp_path):
    # This tests checks that the destriper «does the right thing» with a
    # very simple TOD that is created by hand, i.e., where pointings and
    # TOD samples are created manually one by one instead of resorting
    # to objects like "SpinningScanningStrategy" and noise generators.
    #
    # The TOD is just the summation of constant baselines and a sky map,
    # and there is no white noise added. The map is at very low resolution,
    # and only a handful of samples are considered. Plus, we consider here
    # just temperature, not polarization.

    # Resolution of the output map
    nside = 1

    # The fake sky map we are going to observe: just a sequence of numbers
    sky_map = np.arange(healpy.nside2npix(nside))

    # Samples per each 1/f baseline
    samples_per_baseline = 3

    # Value of each baseline
    baselines = np.array([1.0, 4.0, -2.0], dtype=np.float32)

    # Number of samples we're going to have in the TOD
    num_of_samples = samples_per_baseline * len(baselines)

    # Sampling frequency
    sampling_frequency_hz = 1.0

    # Duration of the simulation
    duration_s = num_of_samples * sampling_frequency_hz

    # Initializing the simulation
    sim = lbs.Simulation(
        base_path=tmp_path,
        start_time=0.0,
        duration_s=duration_s,
        random_seed=12345,
    )

    detectors = [
        lbs.DetectorInfo(name="mock_detector", sampling_rate_hz=sampling_frequency_hz)
    ]

    # creating one observation
    sim.create_observations(
        detectors=detectors,
        n_blocks_det=1,
        n_blocks_time=1,
    )

    # Now create a simple TOD

    # Let's start from the pixels we're going to observe. Note that we
    # want many repetitions for the destriper to work correctly!
    pixidx = np.array([0, 0, 1, 0, 1, 2, 2, 0, 2], dtype=np.int8)
    assert len(pixidx) == num_of_samples

    # Now generate 1/f baselines with the same number of samples as the TOD
    expected_baselines = np.repeat(baselines, samples_per_baseline).reshape(1, -1)

    # The sky TOD is just the sky map unrolled over the observed pixels
    sky_tod = sky_map[pixidx]

    theta, phi = healpy.pix2ang(nside, pixidx, nest=True)

    # Let's create the TOD, pointings, and polarization angles with our
    # new simple values. We write the TOD in `full_tod` instead of the
    # default `tod` because we want to test that PR#242 works
    sim.observations[0].full_tod = expected_baselines + sky_tod
    sim.observations[0].pointings = np.empty((1, num_of_samples, 2))
    sim.observations[0].psi = np.empty((1, num_of_samples))
    sim.observations[0].pointings[0, :, 0] = theta
    sim.observations[0].pointings[0, :, 1] = phi
    sim.observations[0].pointing_coords = lbs.CoordinateSystem.Ecliptic
    sim.observations[0].psi[0, :] = np.linspace(
        start=0.0,
        stop=np.pi,
        num=num_of_samples,
    )

    param_noise_madam = lbs.DestriperParameters(
        nside=nside,
        nnz=1,  # Compute just I
        baseline_length_s=samples_per_baseline / sampling_frequency_hz,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
        coordinate_system=lbs.coordinates.CoordinateSystem.Ecliptic,
        iter_max=10,
    )

    # The call to round(10) means that we clip to zero those samples whose
    # value is negligible (e.g., 4e-16). As the destriper is going to
    # overwrite the TOD, we keep a copy in "input_tod"
    input_tod = np.copy(sim.observations[0].full_tod).round(10)

    # Run the destriper and modify the TOD in place
    result = lbs.destripe(
        sim=sim,
        params=param_noise_madam,
        component="full_tod",
    )

    # Let's retrieve the TOD and clip small values as above
    output_tod = np.copy(sim.observations[0].full_tod).round(10)

    # These are the baselines computed by the destriper (we must compute
    # them manually, because unfortunately TOAST2 does not save them)
    computed_baselines = input_tod - output_tod

    # The expected difference between the input baselines and the computed
    # ones is not necessarily zero, as the offset of the baselines is not
    # constrained by the destriping equations. We just check that this
    # mismatch is constant.
    mismatch = computed_baselines - expected_baselines
    assert np.allclose(mismatch, mismatch[0])

    # Compute the expected hit map and check that the destriper got it
    # right
    expected_hit_map = np.bincount(pixidx, minlength=len(result.hit_map))
    assert np.allclose(expected_hit_map, result.hit_map)

    # Compute the difference between the input map and the destriped map,
    # and check that their difference is a constant (see above the
    # discussion for "mismatch").
    hit_mask = result.hit_map > 0
    map_mismatch = sky_map[hit_mask] - result.destriped_map[hit_mask]
    assert np.allclose(map_mismatch, map_mismatch[0])


def run_destriper_tests(tmp_path, coordinates: CoordinateSystem):
    coordinates_str = COORDINATE_SYSTEM_STR[coordinates]

    sim = lbs.Simulation(
        base_path=tmp_path / "destriper_output",
        start_time=0,
        duration_s=86400.0,
        random_seed=12345,
    )

    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # seconds
            precession_rate_hz=1.0 / (4 * u.day).to("s").value,
        )
    )

    sim.set_instrument(
        lbs.InstrumentInfo(name="core", spin_boresight_angle_rad=np.deg2rad(65)),
    )
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

    sim.compute_pointings()

    # Generate some white noise
    rs = RandomState(MT19937(SeedSequence(123456789)))
    for cur_obs in sim.observations:
        cur_obs.tod *= 0.0
        cur_obs.tod += rs.randn(*cur_obs.tod.shape)

    params = lbs.DestriperParameters(
        nside=16,
        nnz=3,
        baseline_length_s=100,
        iter_max=10,
        coordinate_system=coordinates,
        return_hit_map=True,
        return_binned_map=True,
        return_destriped_map=True,
        return_npp=True,
        return_invnpp=True,
        return_rcond=True,
    )

    results = lbs.destripe(sim, params=params)
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
        base_path=Path(tmp_path) / "destriper_output",
        start_time=0,
        duration_s=3 * 86400.0,
        random_seed=12345,
    )

    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(30),  # CORE-specific parameter
            spin_rate_hz=0.5 / 60,  # Ditto
            # We use astropy to convert the period (4 days) in
            # seconds
            precession_rate_hz=1.0 / (4 * u.day).to("s").value,
        )
    )
    sim.set_instrument(
        lbs.InstrumentInfo(
            name="mock_instrument",
            spin_boresight_angle_rad=np.deg2rad(65),
        ),
    )

    detectors = [
        lbs.DetectorInfo(
            name="noiseless_detector",
            sampling_rate_hz=5.0,
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

    sim.compute_pointings()

    params = lbs.MbsParameters(
        make_cmb=True,
        nside=8,
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

    result = lbs.destripe(sim, params)

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
