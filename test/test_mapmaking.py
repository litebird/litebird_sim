# -*- encoding: utf-8 -*-

import numpy as np

from litebird_sim.mapmaking.common import cholesky, solve_cholesky, estimate_cond_number


def _generate_random_positive_definite_matrix(
    N: int, random: np.random.Generator
) -> np.ndarray:
    """Return a random 3×3 matrix that is symmetric and positive definite"""

    # Use Eq. 10 in KurkiSuonio2009 to create an M matrix for a
    # pixel that has been observed 10 times with random ψ angles
    angles = random.random(10) * np.pi

    # We pick a random value for σ (we need to rescale each
    # coefficient by 1/σ²)
    sigma = random.random() + 0.5
    A = np.zeros((N, N))
    for cur_angle in angles:
        A[0, 0] += 1.0
        A[0, 1] += np.cos(2 * cur_angle)
        A[0, 2] += np.sin(2 * cur_angle)
        A[1, 1] += np.cos(2 * cur_angle) ** 2
        A[1, 2] += np.cos(2 * cur_angle) * np.sin(2 * cur_angle)
        A[2, 2] += np.sin(2 * cur_angle) ** 2

        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]

    return A / sigma**2


def test_cholesky_and_solve_random():
    # To check the correctness of the two methods `cholesky` and `solve_cholesky`,
    # we generate a high number of random 3×3 matrices that satisfy the two
    # properties of (1) symmetry, and (2) positive definiteness, and we check
    # that the results of our functions match the ones returned by the (slower)
    # numpy.linalg.cholesky and numpy.linalg.solve

    random = np.random.Generator(np.random.PCG64(12345))

    # Run the test on 1000 matrices
    for i in range(1000):
        A = _generate_random_positive_definite_matrix(3, random)
        assert np.allclose(np.transpose(A), A)

        L = np.empty(6)
        cholesky(
            a00=A[0, 0],
            a10=A[1, 0],
            a11=A[1, 1],
            a20=A[2, 0],
            a21=A[2, 1],
            a22=A[2, 2],
            dest_L=L,
        )
        numpy_L = np.linalg.cholesky(A)

        # L[0] = l₀₀
        np.testing.assert_almost_equal(actual=L[0], desired=numpy_L[0][0])
        # L[1] = l₁₀
        np.testing.assert_almost_equal(actual=L[1], desired=numpy_L[1][0])
        # L[2] = l₁₁
        np.testing.assert_almost_equal(actual=L[2], desired=numpy_L[1][1])
        # L[3] = l₂₀
        np.testing.assert_almost_equal(actual=L[3], desired=numpy_L[2][0])
        # L[4] = l₂₁
        np.testing.assert_almost_equal(actual=L[4], desired=numpy_L[2][1])
        # L[5] = l₂₂"""
        np.testing.assert_almost_equal(actual=L[5], desired=numpy_L[2][2])

        v = np.array([1.0, 2.0, 3.0])
        x = np.empty(3)
        x[:] = solve_cholesky(L=L, v0=v[0], v1=v[1], v2=v[2])
        np.testing.assert_allclose(actual=x, desired=np.linalg.solve(A, v))


def test_estimate_cond_number():
    random = np.random.Generator(np.random.PCG64(12345))

    # Run the test on 1000 matrices
    for i in range(1000):
        A = _generate_random_positive_definite_matrix(3, random)
        assert np.allclose(np.transpose(A), A)

        (cond, found) = estimate_cond_number(
            a00=A[0][0],
            a10=A[1][0],
            a11=A[1][1],
            a20=A[2][0],
            a21=A[2][1],
            a22=A[2][2],
        )

        if found:
            np.testing.assert_almost_equal(actual=cond, desired=np.linalg.cond(A))


def test_is_well_conditioned():
    # Singular matrices (1-2 hits per pixel) must be rejected whatever the
    # scale of the weights; estimate_cond_number accepts some of them
    from litebird_sim.mapmaking.common import _is_well_conditioned

    def nobs_matrix(psi_list, weight):
        vecs = [np.array([1, np.cos(2 * psi), np.sin(2 * psi)]) for psi in psi_list]
        return weight * sum(np.outer(v, v) for v in vecs)

    rng = np.random.default_rng(1)
    for weight in (1e-8, 1.0, 1e8):
        for num_of_hits in (1, 2):
            for _ in range(200):
                m = nobs_matrix(rng.uniform(0, np.pi, num_of_hits), weight)
                assert not _is_well_conditioned(m)

        # Three well-separated angles give a good matrix
        assert _is_well_conditioned(
            nobs_matrix([0.0, np.pi / 3, 2 * np.pi / 3], weight)
        )
        assert not _is_well_conditioned(np.zeros((3, 3)))


def test_flags_in_binner_and_destriper():
    # Flagged samples must not contribute to the maps: corrupting them must
    # not change the result, unless flag_mask selects other bits
    import pytest

    import litebird_sim as lbs

    if lbs.MPI_COMM_WORLD.size > 1:
        pytest.skip("This test uses one observation, which is not split over MPI")

    sim = lbs.Simulation(start_time=0, duration_s=3600.0, random_seed=1)
    sim.set_instrument(lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=0.87))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=name, sampling_rate_hz=5.0, net_ukrts=50.0, pol_angle_rad=angle
            )
            for name, angle in (("a", 0.0), ("b", np.pi / 4))
        ]
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.785, precession_rate_hz=1 / 600, spin_rate_hz=1 / 60
        )
    )
    sim.prepare_pointings()
    obs = sim.observations[0]
    obs.tod[:] = np.random.default_rng(1).normal(size=obs.tod.shape)

    flags = lbs.get_local_flags(obs)
    flags[0, 100:300] |= lbs.FLAG_PLANET
    flags[1, 2000:2500] |= lbs.FLAG_PLANET
    lbs.get_global_flags(obs)[5000:5100] |= 2
    bad = (flags != 0) | (obs.global_flags != 0)

    params = lbs.DestriperParameters(samples_per_baseline=500)

    def maps(**kwargs):
        binned = lbs.make_binned_map(nside=4, observations=sim.observations, **kwargs)
        destriped = lbs.make_destriped_map(
            nside=4, observations=sim.observations, params=params, **kwargs
        )
        return binned.binned_map, destriped.destriped_map

    names = ["local_flags", "global_flags"]
    clean = maps(flags=names)
    obs.tod[bad] = 1e10
    for actual, desired in zip(maps(flags=names), clean):
        np.testing.assert_array_equal(actual, desired)

    # Only the bits in flag_mask exclude samples: the corrupted samples of
    # the global flags (bit 2) are now used
    obs.tod[:, 5000:5100] = 1e10
    planet_only = maps(flags=names, flag_mask=lbs.FLAG_PLANET)
    assert np.nanmax(np.abs(planet_only[0])) > 1e8
