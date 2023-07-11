# -*- encoding: utf-8 -*-

import numpy as np

from litebird_sim.mapmaking.common import cholesky, solve_cholesky, estimate_cond_number


# A = np.array([[300.0, -100.0, 173.205], [-100.0, 150.0, 0.0], [173.205, 0.0, 150.0]])


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
        solve_cholesky(L, v, x)
        np.testing.assert_allclose(actual=x, desired=np.linalg.solve(A, v))


def test_estimate_cond_number():
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
