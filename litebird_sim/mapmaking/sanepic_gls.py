"""
GLS map-maker with correlated (1/f) noise using pysanepic

pysanepic does not depend on litebird_sim: this module converts the
observations into `pysanepic.DetectorData` (pointings, HWP angle, TOD and the
1/f noise parameters used to simulate the noise) and calls
`pysanepic.make_maps`.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from litebird_sim.coordinates import CoordinateSystem
from litebird_sim.hwp import HWP
from litebird_sim.mpi import MPI_COMM_WORLD, MPI_ENABLED
from litebird_sim.observation_utilities import (
    _get_hwp_angle,
    _get_pointings_array,
    _normalize_observations_and_pointings,
)

from litebird_sim.utilities import resolve_nthreads

from .common import get_pol_efficiency

if TYPE_CHECKING:
    import pysanepic

_COORDINATES = {CoordinateSystem.Galactic: "G", CoordinateSystem.Ecliptic: "E"}


def _sum_components(obs, components, det_idx):
    """TOD of one detector summed over `components`, without a copy if there is one."""
    tod = getattr(obs, components[0])[det_idx]
    for c in components[1:]:
        tod = tod + getattr(obs, c)[det_idx]
    return tod


def make_sanepic_gls_map(
    nside: int,
    observations,
    pointings: npt.NDArray | list[npt.NDArray] | None = None,
    hwp: HWP | None = None,
    components: str | list[str] = "tod",
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    chunk_s: float = 3600.0,
    pad_s: float = 0.0,
    pad_fill: str = "zeros",
    pol: bool = True,
    preconditioner: str = "block",
    min_pol_rcond: float = 1e-2,
    tol: float = 1e-12,
    maxiter: int = 2000,
    pointings_dtype=np.float64,
    nthreads: int | None = None,
) -> "pysanepic.MapResult":
    """
    GLS map-maker with 1/f noise using pysanepic (SANEPIC algorithm).

    The noise covariance of each detector is modelled from the same
    parameters used by :func:`.add_noise_to_observations` (``net_ukrts``,
    ``fknee_mhz``, ``alpha``, ``fmin_hz``) and applied in Fourier space on
    chunks of ``chunk_s`` seconds.

    Parameters
    ----------
    nside : int
        Nside of the output map
    observations : Observation | list[Observation]
        Observations to use; with MPI, every process must call this function
    pointings : np.ndarray | list[np.ndarray], optional
        Pointings (θ, φ, ψ); by default they are taken from `observations`
    hwp : HWP, optional
        An instance of :class:`.HWP`; by default the one in the observations
    components : str | list[str], optional
        TOD components summed before map-making, by default "tod"
    output_coordinate_system : CoordinateSystem, optional
        Coordinate system of the output map, by default Galactic
    chunk_s : float, optional
        Duration of the chunks on which N^-1 is applied, by default 3600 s
    pad_s : float, optional
        Padding added to both sides of each chunk to avoid the FFT
        wrap-around (seconds), by default 0 (no padding)
    pad_fill : str, optional
        "zeros" (default) or "extrapolate"
    pol : bool, optional
        If True (default) solve for I, Q, U, otherwise for I only
    preconditioner : str, optional
        "block" (default, 3×3 I/Q/U block per pixel) or "jacobi" (original SANEPIC)
    min_pol_rcond : float, optional
        Q/U are solved only where the polarization-angle coverage gives a
        reciprocal condition number >= this value, by default 1e-2
    tol : float, optional
        PCG stops when |r|²/|b|² < tol, by default 1e-12
    maxiter : int, optional
        Maximum number of PCG iterations, by default 2000
    pointings_dtype : dtype, optional
        dtype used to compute pointings on the fly, by default `np.float64`
    nthreads : int, optional
        Threads per process; by default resolved with :func:`.resolve_nthreads`

    Returns
    -------
    pysanepic.MapResult
        Output maps (``maps``, ``hp.UNSEEN`` where not solved), ``hit_map``
        and the PCG convergence information
    """
    try:
        import pysanepic
    except ImportError:
        raise ImportError(
            "Could not import `pysanepic`. Make sure that the package "
            "`pysanepic` is installed in the same environment as `litebird_sim`"
        )

    if isinstance(components, str):
        components = [components]

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )

    def detector_data():
        # a generator: pointings are computed for one detector at a time, and
        # pysanepic keeps only pixels and angles (no copy of the TODs)
        for obs, cur_ptg in zip(obs_list, ptg_list):
            hwp_angle = _get_hwp_angle(obs=obs, hwp=hwp, pointing_dtype=pointings_dtype)
            gamma = get_pol_efficiency(obs)
            net = np.broadcast_to(obs.net_ukrts, (obs.n_detectors,))
            for det_idx in range(obs.n_detectors):
                ptg_det, hwp_angle = _get_pointings_array(
                    detector_idx=det_idx,
                    pointings=cur_ptg,
                    hwp_angle=hwp_angle,
                    output_coordinate_system=output_coordinate_system,
                    pointings_dtype=pointings_dtype,
                )
                yield pysanepic.DetectorData(
                    tod=_sum_components(obs, components, det_idx),
                    theta=ptg_det[:, 0],
                    phi=ptg_det[:, 1],
                    psi=ptg_det[:, 2],
                    hwp_angle=hwp_angle,
                    coordinates=_COORDINATES[output_coordinate_system],
                    sampling_rate_hz=obs.sampling_rate_hz,
                    net_ukrts=net[det_idx],
                    fknee_hz=obs.fknee_mhz[det_idx] / 1e3,
                    alpha=obs.alpha[det_idx],
                    fmin_hz=obs.fmin_hz[det_idx],
                    pol_angle_rad=obs.pol_angle_rad[det_idx],
                    pol_efficiency=gamma[det_idx],
                )

    comm = MPI_COMM_WORLD if MPI_ENABLED and MPI_COMM_WORLD.size > 1 else None
    return pysanepic.make_maps(
        detector_data(),
        nside,
        coordinates=_COORDINATES[output_coordinate_system],
        chunk_s=chunk_s,
        pad_s=pad_s,
        pad_fill=pad_fill,
        pol=pol,
        preconditioner=preconditioner,
        min_pol_rcond=min_pol_rcond,
        tol=tol,
        maxiter=maxiter,
        comm=comm,
        nthreads=resolve_nthreads(nthreads),
    )
